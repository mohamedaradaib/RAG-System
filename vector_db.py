import os
import re
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, PayloadSchemaType, PointIdsList


def lexical_overlap_score(query_text: str, document_text: str) -> float:
    if not query_text or not document_text:
        return 0.0

    query_tokens = [token for token in re.findall(r"\b[\w'-]+\b", query_text.lower()) if len(token) > 2]
    doc_tokens = [token for token in re.findall(r"\b[\w'-]+\b", document_text.lower()) if len(token) > 2]

    if not query_tokens or not doc_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    doc_counts = Counter(doc_tokens)
    overlap = sum(min(query_counts[token], doc_counts[token]) for token in set(query_counts))
    return overlap / max(1, len(query_counts))


class QdrantStorage:
    def __init__(self, url=None, collection="docs", dim=3072, client=None):
        qdrant_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = client or QdrantClient(
            url=qdrant_url,
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30,
        )
        self.collection = collection
        self.dim = dim
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        payload_schema = self.client.get_collection(self.collection).payload_schema or {}
        if "source" not in payload_schema:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def replace_source(self, ids, vectors, payloads, source_id):
        if not ids:
            raise ValueError("Cannot replace a source with no chunks.")

        self.upsert(ids, vectors, payloads)
        retained_ids = {str(point_id) for point_id in ids}
        source_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_id))]
        )
        stale_ids = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=source_filter,
                limit=1000,
                offset=offset,
                with_payload=False,
            )
            stale_ids.extend(
                point.id for point in points if str(point.id) not in retained_ids
            )
            if offset is None:
                break

        if stale_ids:
            self.client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=stale_ids),
            )

    def list_sources(self):
        points, _ = self.client.scroll(collection_name=self.collection, with_payload=True, limit=1000)
        sources = set()

        for point in points:
            payload = getattr(point, "payload", None) or {}
            source = payload.get("source")
            if source:
                sources.add(source)

        return sorted(sources)

    def delete_source(self, source_id):
        filter_ = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_id))]
        )
        self.client.delete(collection_name=self.collection, points_selector=filter_)

    def search(self, query_vector, top_k: int = 5, query_text: str | None = None):
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=max(top_k * 4, top_k),
        )

        ranked = []
        for r in response.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if not text:
                continue

            vector_score = float(getattr(r, "score", 0.0) or 0.0)
            lexical_score = lexical_overlap_score(query_text or "", text) if query_text else 0.0
            combined_score = vector_score + (lexical_score * 0.5)
            ranked.append({
                "text": text,
                "source": source,
                "chunk_index": payload.get("chunk_index"),
                "page": payload.get("page"),
                "score": combined_score,
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)

        contexts = []
        sources = set()
        citations = []
        for citation_id, item in enumerate(ranked[:top_k], start=1):
            contexts.append(item["text"])
            if item["source"]:
                sources.add(item["source"])
            citations.append({"id": citation_id, **item})

        return {
            "contexts": contexts,
            "sources": sorted(sources),
            "citations": citations,
        }