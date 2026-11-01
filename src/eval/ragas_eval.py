"""Section 10: RAGAS scores on a small sample, judged by the free Gemini lite models.

RAGAS makes several LLM calls per sample, so it is run on ``ragas.samples_per_dataset`` questions
per dataset (not the whole evaluation set). Every judge call goes through ``call_llm`` (role
``ragas_judge``), so rate limits, the daily stop and the on-disk cache all apply; embeddings are a
local sentence-transformers model (no Gemini embeddings).

Samples are scored one at a time and appended to ``results/eval/ragas_<dataset>_<system>.jsonl``,
so a daily-limit stop resumes where it ended. Answers that abstain, and answers with no contexts,
are not scored (faithfulness of a refusal means nothing).
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.outputs import Generation, LLMResult
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM

from src.eval.harness import chunk_text_lookup, load_eval_set, results_dir
from src.ingest.io_utils import append_jsonl, read_jsonl
from src.llm import DailyLimitReached, call_llm

ROLE = "ragas_judge"


@dataclass
class CallLLMRagas(BaseRagasLLM):
    """RAGAS judge that sends every prompt through ``call_llm`` with the ``ragas_judge`` role."""

    role: str = ROLE
    limit_error: Optional[Exception] = field(default=None, repr=False)

    def _complete(self, prompt: Any, n: int) -> LLMResult:
        """One reply per requested completion (identical prompts hit the cache and repeat)."""
        texts = []
        for _ in range(max(n, 1)):
            try:
                texts.append(call_llm(prompt.to_string(), role=self.role).text)
            except DailyLimitReached as error:
                self.limit_error = error
                raise
        return LLMResult(generations=[[Generation(text=text) for text in texts]])

    def generate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult:
        """Synchronous RAGAS hook."""
        return self._complete(prompt, n)

    async def agenerate_text(self, prompt, n=1, temperature=0.01, stop=None, callbacks=None) -> LLMResult:
        """Async RAGAS hook; the call itself is blocking, which is fine with one worker."""
        return self._complete(prompt, n)

    def is_finished(self, response: LLMResult) -> bool:
        """Replies come back complete."""
        return True


class LocalRagasEmbeddings(BaseRagasEmbeddings):
    """RAGAS embeddings computed by the project's local sentence-transformers ``Embedder`` (no API)."""

    def __init__(self, embedder: Any):
        """Wrap an object with ``encode(texts)`` returning a 2-D array."""
        super().__init__()
        self.embedder = embedder

    def embed_query(self, text: str) -> List[float]:
        """Embedding of one text."""
        return self.embedder.encode([text])[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeddings of several texts."""
        return [row.tolist() for row in self.embedder.encode(list(texts))]

    async def aembed_query(self, text: str) -> List[float]:
        """Async hook; the local model is fast, so it just calls the sync version."""
        return self.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async hook for several texts."""
        return self.embed_documents(texts)


def sample_query_ids(dataset: str, config: Dict[str, Any]) -> List[str]:
    """The first ``ragas.samples_per_dataset`` answerable questions of the evaluation set."""
    answerable = [question for question in load_eval_set(dataset, config) if "insufficient information" not in " ".join(question["references"]).lower()]
    return [question["query_id"] for question in answerable[: config["ragas"]["samples_per_dataset"]]]


def scorable(record: Dict[str, Any]) -> bool:
    """A record RAGAS can judge: it retrieved something and did not abstain."""
    return bool(record["retrieved_chunk_ids"]) and not record["abstained"]


def build_metrics(config: Dict[str, Any]) -> List[Any]:
    """The configured RAGAS metric objects."""
    from ragas.metrics import AnswerRelevancy, Faithfulness

    factories = {
        "faithfulness": Faithfulness,
        "answer_relevancy": lambda: AnswerRelevancy(strictness=config["ragas"]["answer_relevancy_strictness"]),
    }
    return [factories[name]() for name in config["ragas"]["metrics"]]


def score_sample(record: Dict[str, Any], texts: Dict[str, Dict[str, Any]], metrics: List[Any], llm: CallLLMRagas, embeddings: Any) -> Dict[str, Optional[float]]:
    """RAGAS scores of one answer; a metric that could not be computed is None."""
    from ragas import EvaluationDataset, RunConfig, evaluate

    sample = {
        "user_input": record["query"],
        "response": record["answer"],
        "retrieved_contexts": [texts[chunk_id]["text"] for chunk_id in record["retrieved_chunk_ids"] if chunk_id in texts],
    }
    result = evaluate(
        EvaluationDataset.from_list([sample]), metrics=metrics, llm=llm, embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=300), show_progress=False, raise_exceptions=False,
    )
    row = result.to_pandas().iloc[0]
    scores: Dict[str, Optional[float]] = {}
    for metric in metrics:
        value = row.get(metric.name)
        scores[metric.name] = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
    return scores


def run_ragas(config: Dict[str, Any], dataset: str, system: str, limit: Optional[int] = None) -> int:
    """Score the sampled answers of one (dataset, system); return how many samples were newly scored."""
    from src.index.embedder import Embedder

    directory = results_dir(config)
    source = directory / f"{dataset}_{system}.jsonl"
    if not source.is_file():
        return 0
    wanted = set(sample_query_ids(dataset, config))
    records = [record for record in read_jsonl(source) if record["query_id"] in wanted and scorable(record)]
    output = directory / f"ragas_{dataset}_{system}.jsonl"
    done = {record["query_id"] for record in read_jsonl(output)}
    todo = [record for record in records if record["query_id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        return 0

    texts = chunk_text_lookup(dataset, config)
    llm = CallLLMRagas()
    embeddings = LocalRagasEmbeddings(Embedder.from_config(config))
    metrics = build_metrics(config)
    scored = 0
    for record in todo:
        scores = score_sample(record, texts, metrics, llm, embeddings)
        if llm.limit_error is not None:
            raise llm.limit_error
        append_jsonl(output, [{"query_id": record["query_id"], "dataset": dataset, "system": system, **scores}])
        scored += 1
    return scored


def summarize_ragas(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mean RAGAS score per (dataset, system) over the scored samples; datasets are never averaged."""
    summary: Dict[str, Any] = {}
    for dataset in ("arxiv", "hotpotqa", "multihoprag"):
        for system in config["ragas"]["systems"]:
            path = results_dir(config) / f"ragas_{dataset}_{system}.jsonl"
            if not path.is_file():
                continue
            rows = list(read_jsonl(path))
            entry: Dict[str, Any] = {"n": len(rows)}
            for name in config["ragas"]["metrics"]:
                values = [row[name] for row in rows if row.get(name) is not None]
                entry[name] = sum(values) / len(values) if values else None
                entry[f"{name}_n"] = len(values)
            summary.setdefault(dataset, {})[system] = entry
    return summary
