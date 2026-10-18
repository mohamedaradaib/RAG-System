import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

from vector_db import QdrantStorage


EMBED_MODEL = "text-embedding-3-large"


def load_dataset(path: Path) -> list[dict[str, str | list[str]]]:
    with path.open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("The evaluation dataset must be a non-empty JSON array.")

    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} must be a JSON object.")
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"Item {index} must include a non-empty question.")
        expected_sources = item.get("expected_sources")
        if (
            not isinstance(expected_sources, list)
            or not expected_sources
            or not all(isinstance(source, str) and source.strip() for source in expected_sources)
        ):
            raise ValueError(f"Item {index} must include a non-empty expected_sources list.")

    return dataset


def calculate_retrieval_metrics(
    expected_sources: list[str], retrieved_sources: list[str]
) -> tuple[bool, float, int | None]:
    expected = set(expected_sources)
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected:
            return True, 1 / rank, rank
    return False, 0.0, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate source retrieval against a labeled JSON dataset.")
    parser.add_argument("dataset", type=Path, help="Path to a JSON evaluation dataset")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        dataset = load_dataset(args.dataset)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    questions = [item["question"] for item in dataset]
    client = OpenAI()
    embeddings = client.embeddings.create(model=EMBED_MODEL, input=questions).data
    store = QdrantStorage()

    hits = 0
    reciprocal_ranks = []
    search_times = []
    print(f"Evaluating {len(dataset)} questions at k={args.top_k}")

    for item, embedding in zip(dataset, embeddings):
        started = time.perf_counter()
        result = store.search(
            embedding.embedding,
            args.top_k,
            query_text=item["question"],
        )
        search_times.append(time.perf_counter() - started)
        retrieved_sources = [citation["source"] for citation in result["citations"]]
        hit, reciprocal_rank, rank = calculate_retrieval_metrics(
            item["expected_sources"],
            retrieved_sources,
        )
        hits += int(hit)
        reciprocal_ranks.append(reciprocal_rank)
        rank_label = str(rank) if rank is not None else "miss"
        print(f"- {item['question']} | first relevant rank: {rank_label}")

    question_count = len(dataset)
    hit_rate = hits / question_count
    mean_reciprocal_rank = sum(reciprocal_ranks) / question_count
    mean_search_ms = sum(search_times) / question_count * 1000
    print(f"Hit@{args.top_k}: {hit_rate:.3f} ({hits}/{question_count})")
    print(f"MRR@{args.top_k}: {mean_reciprocal_rank:.3f}")
    print(f"Mean vector search latency: {mean_search_ms:.1f} ms")


if __name__ == "__main__":
    main()