import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response):
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    if match is None:
        # If pattern doesn't match, return an empty list
        return []
    return json.loads(match.group())["named_entities"]


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self._max_completion_tokens = None
        self._debug_timing = os.getenv("OPENIE_DEBUG_TIMING", "0") == "1"
        env_max = os.getenv("OPENIE_MAX_COMPLETION_TOKENS")
        if env_max:
            try:
                self._max_completion_tokens = max(1, int(env_max))
            except ValueError:
                self._max_completion_tokens = None

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            infer_kwargs = {"messages": ner_input_message, "response_format": {"type": "json_object"}}
            if self._max_completion_tokens is not None:
                infer_kwargs["max_completion_tokens"] = self._max_completion_tokens
            raw_response, metadata, cache_hit = self.llm_model.infer(**infer_kwargs)
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is None:
                # If pattern doesn't match, return an empty list
                return []
            return json.loads(match.group())["triples"]

        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            infer_kwargs = {"messages": messages, "response_format": {"type": "json_object"}}
            if self._max_completion_tokens is not None:
                infer_kwargs["max_completion_tokens"] = self._max_completion_tokens
            raw_response, metadata, cache_hit = self.llm_model.infer(**infer_kwargs)
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        t1 = time.perf_counter()
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        t2 = time.perf_counter()
        if self._debug_timing:
            print(
                f"[TIMING] openie_chunk chunk_id={chunk_key} ner_s={t1-t0:.3f} "
                f"triple_s={t2-t1:.3f} total_s={t2-t0:.3f}",
                flush=True,
            )
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        max_workers_env = os.getenv("OPENIE_MAX_WORKERS")
        max_workers = None
        if max_workers_env:
            try:
                max_workers = max(1, int(max_workers_env))
            except ValueError:
                max_workers = None
        if max_workers is None:
            max_workers = min(len(chunk_passages), (os.cpu_count() or 4))

        parallel_mode = os.getenv("OPENIE_PARALLEL_MODE", "per_chunk").lower()

        ner_results_list = []
        triple_results_list = []
        t_batch_start = time.perf_counter()
        if self._debug_timing:
            print(
                f"[TIMING] openie_batch_start chunks={len(chunk_passages)} mode={parallel_mode} max_workers={max_workers}",
                flush=True,
            )

        if parallel_mode == "staged":
            # Original staged mode: NER for all chunks, then triple extraction.
            total_prompt_tokens = 0
            total_completion_tokens = 0
            num_cache_hit = 0
            
            if self._debug_timing:
                print(f"[TIMING] batch_openie STAGED start. chunks={len(chunk_passages)} max_workers={max_workers}")

            ner_results_map = {} # temp storage for triple step
            
            t_ner_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                ner_futures = {
                    executor.submit(self.ner, chunk_key, passage): chunk_key
                    for chunk_key, passage in chunk_passages.items()
                }

                pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
                for future in pbar:
                    result = future.result()
                    if self._debug_timing:
                        print(f"[TIMING] NER chunk done: {result.chunk_id}")
                    ner_results_list.append(result)
                    ner_results_map[result.chunk_id] = result
                    metadata = result.metadata
                    total_prompt_tokens += metadata.get('prompt_tokens', 0)
                    total_completion_tokens += metadata.get('completion_tokens', 0)
                    if metadata.get('cache_hit'):
                        num_cache_hit += 1

                    pbar.set_postfix({
                        "total_tokens": total_prompt_tokens + total_completion_tokens,
                        "num_cache_hit": num_cache_hit
                    })
            t_ner_end = time.perf_counter()
            if self._debug_timing:
                print(f"[TIMING] batch_openie NER phase done. elapsed={t_ner_end-t_ner_start:.3f}s")
            
            t_triple_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                 triple_futures = []
                 for chunk_key, Passage in chunk_passages.items():
                     if chunk_key in ner_results_map:
                         ner_out = ner_results_map[chunk_key]
                         triple_futures.append(executor.submit(self.triple_extraction, chunk_key, Passage, ner_out.unique_entities))
                 
                 pbar = tqdm(as_completed(triple_futures), total=len(triple_futures), desc="Triples")
                 for future in pbar:
                     result = future.result()
                     if self._debug_timing:
                         print(f"[TIMING] Triple chunk done: {result.chunk_id}")
                     triple_results_list.append(result)
                     metadata = result.metadata
                     total_prompt_tokens += metadata.get('prompt_tokens', 0)
                     total_completion_tokens += metadata.get('completion_tokens', 0)
                     if metadata.get('cache_hit'):
                         num_cache_hit += 1
            t_triple_end = time.perf_counter()
            
            t_total_end = time.perf_counter()
            
            if self._debug_timing:
                print(f"[TIMING] batch_openie Triples phase done. elapsed={t_triple_end-t_triple_start:.3f}s")
                print(f"[TIMING] batch_openie TOTAL. chunks={len(chunk_passages)} total_s={t_total_end-t_batch_start:.3f}s")

            return {r.chunk_id: r for r in ner_results_list}, {r.chunk_id: r for r in triple_results_list}

        # Fallback to non-staged or per-chunk mode (currently defaulting to staged in our usage)
        else:
            # Per-chunk mode: run NER + triple sequentially per chunk, but fully parallel across chunks.
            total_prompt_tokens = 0
            total_completion_tokens = 0
            num_cache_hit = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.openie, chunk_key, passage): chunk_key
                    for chunk_key, passage in chunk_passages.items()
                }
                pbar = tqdm(as_completed(futures), total=len(futures), desc="OpenIE")
                for future in pbar:
                    result = future.result()
                    ner_result = result["ner"]
                    triple_result = result["triplets"]
                    ner_results_list.append(ner_result)
                    triple_results_list.append(triple_result)

                    for md in (ner_result.metadata, triple_result.metadata):
                        total_prompt_tokens += md.get('prompt_tokens', 0)
                        total_completion_tokens += md.get('completion_tokens', 0)
                        if md.get('cache_hit'):
                            num_cache_hit += 1

                    pbar.set_postfix({
                        'total_prompt_tokens': total_prompt_tokens,
                        'total_completion_tokens': total_completion_tokens,
                        'num_cache_hit': num_cache_hit
                    })
            if self._debug_timing:
                t_batch_end = time.perf_counter()
                print(
                    f"[TIMING] openie_batch_done chunks={len(ner_results_list)} total_s={t_batch_end - t_batch_start:.3f}",
                    flush=True,
                )

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict
