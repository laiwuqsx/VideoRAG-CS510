import json
import os
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import EMBED_MODEL, TOP_K_CHUNKS


_EMBED_MODEL_INSTANCE = None


def _get_embed_model():
    global _EMBED_MODEL_INSTANCE
    if _EMBED_MODEL_INSTANCE is None:
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL_INSTANCE = SentenceTransformer(EMBED_MODEL)
    return _EMBED_MODEL_INSTANCE


class JsonKVStorage:
    """
    Persistent key-value storage backed by a single JSON file.
    """

    def __init__(self, namespace: str, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.namespace = namespace
        self.directory = directory
        self.path = os.path.join(directory, f"{namespace}.json")
        self._data: Dict[str, object] = {}

        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                self._data = json.load(fh)

    def get(self, key: str):
        return self._data.get(key)

    def set(self, key: str, value):
        self._data[key] = value
        self._save()

    def delete(self, key: str):
        if key in self._data:
            del self._data[key]
            self._save()

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, ensure_ascii=False)


class SimpleVectorStore:
    """
    In-memory vector store with optional JSON persistence.
    """

    def __init__(self):
        self._records: List[Dict] = []

    def add(self, id: str, text: str, metadata: Optional[dict] = None):
        """
        Embed text and insert or overwrite a record.
        """
        embedding = _get_embed_model().encode(text)
        record = {
            "id": id,
            "text": text,
            "embedding": np.asarray(embedding, dtype=float),
            "metadata": metadata or {},
        }

        for i, rec in enumerate(self._records):
            if rec["id"] == id:
                self._records[i] = record
                return

        self._records.append(record)

    def search(self, query: str, top_k: int = TOP_K_CHUNKS) -> List[Dict]:
        """
        Return the most similar stored records for the query.
        """
        if not self._records:
            return []

        query_embedding = np.asarray(_get_embed_model().encode(query), dtype=float).reshape(1, -1)
        matrix = np.stack([rec["embedding"] for rec in self._records], axis=0)
        scores = cosine_similarity(query_embedding, matrix)[0]

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Dict] = []
        for idx in ranked_indices:
            rec = self._records[int(idx)]
            results.append(
                {
                    "id": rec["id"],
                    "text": rec["text"],
                    "metadata": dict(rec.get("metadata", {})),
                    "score": float(scores[int(idx)]),
                }
            )
        return results

    def save(self, path: str):
        """
        Serialize records to JSON, converting embeddings to lists.
        """
        payload = []
        for rec in self._records:
            payload.append(
                {
                    "id": rec["id"],
                    "text": rec["text"],
                    "embedding": rec["embedding"].tolist(),
                    "metadata": rec.get("metadata", {}),
                }
            )

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def load(self, path: str):
        """
        Restore records from a JSON file.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._records = []
        for rec in payload:
            self._records.append(
                {
                    "id": rec["id"],
                    "text": rec["text"],
                    "embedding": np.asarray(rec["embedding"], dtype=float),
                    "metadata": rec.get("metadata", {}),
                }
            )


def build_text_chunks(enriched_segments: List[Dict], max_chars: int = 1500) -> List[Dict]:
    """
    Combine transcript and caption into indexable text chunks.
    """
    overlap = 200
    step = max(1, max_chars - overlap)
    chunks: List[Dict] = []

    for segment in enriched_segments:
        parts = []
        transcript = (segment.get("transcript") or "").strip()
        caption = (segment.get("caption") or "").strip()

        if transcript:
            parts.append(f"[Transcript]: {transcript}")
        if caption:
            parts.append(f"[Visual]: {caption}")

        combined = "\n".join(parts).strip()
        if not combined:
            combined = "[Transcript]:"

        windows = [combined]
        if len(combined) > max_chars:
            windows = []
            for start in range(0, len(combined), step):
                window = combined[start : start + max_chars].strip()
                if window:
                    windows.append(window)
                if start + max_chars >= len(combined):
                    break

        for chunk_idx, chunk_text in enumerate(windows):
            chunks.append(
                {
                    "id": f"seg{segment['index']}_chunk{chunk_idx}",
                    "text": chunk_text,
                    "segment_index": segment["index"],
                    "segment_name": segment["name"],
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                }
            )

    return chunks
