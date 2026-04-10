from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # Khởi tạo chromadb client và collection
            client = chromadb.Client()
            try:
                client.delete_collection(name=self._collection_name)
            except Exception:
                pass
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Biến đổi doc thành record với embedding tương ứng
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata or {},
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # Tìm kiếm độ tương đồng trong bộ nhớ cục bộ
        query_emb = self._embedding_fn(query)
        scored = []
        for r in records:
            score = compute_similarity(query_emb, r["embedding"])
            r_copy = r.copy()
            r_copy["score"] = score
            scored.append((score, r_copy))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # Embed và lưu vào database
        import uuid
        records = [self._make_record(doc) for doc in docs]
        if self._use_chroma and self._collection:
            self._collection.add(
                ids=[str(uuid.uuid4()) for _ in records],
                documents=[r["content"] for r in records],
                embeddings=[r["embedding"] for r in records],
                metadatas=[{"doc_id": r["id"], **(r["metadata"] or {})} for r in records]
            )
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.
        ...
        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # Tìm kiếm dựa trên vector nhúng (có hỗ trợ chromadb hoặc in-memory)
        if self._use_chroma and self._collection:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
            formatted = []
            if results and results.get("documents"):
                for i in range(len(results["documents"][0])):
                    formatted.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0
                    })
            return formatted
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # Trả về số lượng chunk đang được lưu trữ
        if self._use_chroma and self._collection:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # Lọc thông qua metadata filter rồi thực hiện tìm kiếm
        if self._use_chroma and self._collection:
            query_emb = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_emb], 
                n_results=top_k, 
                where=metadata_filter or None,
                include=["documents", "metadatas", "distances"]
            )
            formatted = []
            if results and results.get("documents"):
                for i in range(len(results["documents"][0])):
                    formatted.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "score": 1.0 - results["distances"][0][i] if results.get("distances") else 0.0
                    })
            return formatted
            
        filtered = self._store
        if metadata_filter:
            filtered = [
                r for r in self._store
                if all(r.get("metadata", {}).get(k) == v for k, v in metadata_filter.items())
            ]
        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # Xóa các record thuộc về doc_id
        if self._use_chroma and self._collection:
            initial_count = self._collection.count()
            self._collection.delete(where={"doc_id": doc_id})
            return self._collection.count() < initial_count
            
        initial_len = len(self._store)
        self._store = [r for r in self._store if r.get("id") != doc_id and r.get("metadata", {}).get("doc_id") != doc_id]
        return len(self._store) < initial_len
