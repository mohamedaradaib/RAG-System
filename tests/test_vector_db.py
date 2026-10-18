import unittest
import uuid
from unittest.mock import patch

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from vector_db import QdrantStorage


class ReplaceSourceTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(":memory:")
        self.storage = QdrantStorage(client=self.client, dim=2)
        self.source = "guide.pdf"

    def _replace(self, texts):
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.source}:{index}"))
            for index in range(len(texts))
        ]
        vectors = [[float(index + 1), 1.0] for index in range(len(texts))]
        payloads = [
            {"source": self.source, "text": text, "chunk_index": index}
            for index, text in enumerate(texts)
        ]
        self.storage.replace_source(ids, vectors, payloads, self.source)

    def _source_points(self):
        points, _ = self.client.scroll(
            collection_name=self.storage.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=self.source))]
            ),
            with_payload=True,
            limit=100,
        )
        return points

    def test_replacing_with_fewer_chunks_removes_old_trailing_chunks(self):
        self._replace(["old first", "old second", "old third"])
        self._replace(["new first"])

        points = self._source_points()
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].payload["text"], "new first")

    def test_failed_upsert_preserves_existing_chunks(self):
        self._replace(["existing first", "existing second"])

        with patch.object(self.storage, "upsert", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                self.storage.replace_source(
                    [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.source}:0"))],
                    [[1.0, 1.0]],
                    [{"source": self.source, "text": "replacement", "chunk_index": 0}],
                    self.source,
                )

        texts = {point.payload["text"] for point in self._source_points()}
        self.assertEqual(texts, {"existing first", "existing second"})


if __name__ == "__main__":
    unittest.main()