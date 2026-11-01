"""Run the vanilla RAG baseline on 10 sample questions and save results/baseline_sample.json.

Needs the Qdrant server (scripts/start_qdrant.ps1), the built collections
(scripts/build_index.py) and Ollama running with the generator model.

Run from the project root:
    python scripts/run_baseline.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.vanilla_rag import vanilla_rag  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.index.embedder import Embedder  # noqa: E402
from src.index.qdrant_store import QdrantStore, make_client  # noqa: E402

SAMPLE_QUESTIONS = [
    "What methods are proposed to reduce hallucination in large language models?",
    "How can retrieval-augmented generation be applied to multi-hop question answering?",
    "What approaches exist for efficient fine-tuning of language models?",
    "How are language models evaluated on long-context tasks?",
    "What techniques improve the robustness of dense retrieval?",
    "How is reinforcement learning used to improve language model reasoning?",
    "What are recent approaches to machine unlearning in language models?",
    "How do researchers benchmark retrieval of scientific papers?",
    "What methods compress or select evidence for multimodal retrieval-augmented generation?",
    "What is the capital of France?",  # general-knowledge question: retrieval should not help
]

OUTPUT_FILE = "results/baseline_sample.json"


def main() -> None:
    """Answer each sample question and write the results file."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    vector_config = config["vector_store"]
    top_k = config["retrieval"]["top_k"]

    embedder = Embedder.from_config(config)
    store = QdrantStore(
        make_client(vector_config),
        config["baseline"]["collection"],
        embedder,
        vector_config,
        use_quantization=False,
    )

    results = []
    for number, question in enumerate(SAMPLE_QUESTIONS, start=1):
        output = vanilla_rag(question, store, top_k)
        record = {"question": question, **output}
        record["contexts"] = [
            {"chunk_id": chunk["chunk_id"], "title": chunk["title"], "score": chunk["score"]}
            for chunk in output["contexts"]
        ]
        results.append(record)
        print(f"{number}. {question}")
        print(f"   answer: {output['answer'][:200]}")
        print(f"   latency {output['latency']:.2f}s, tokens {output['tokens']}, cached={output['cached']}\n")

    output_path = PROJECT_ROOT / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(results)} results to {output_path}")
    print(f"Usage log: {PROJECT_ROOT / config['llm']['usage_log_path']}")


if __name__ == "__main__":
    main()
