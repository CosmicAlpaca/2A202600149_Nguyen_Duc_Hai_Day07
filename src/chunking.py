from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class MarkdownChunker:
    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Tách theo cả # Header 1 và ## Header 2 để chunk nhỏ và tập trung hơn
        import re
        # Tìm các vị trí bắt đầu bằng # hoặc ## ở đầu dòng
        parts = re.split(r'\n(?=#+ )', "\n" + text)
        chunks = [p.strip() for p in parts if p.strip()]
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.


    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        # Cắt thành các câu và nhóm lại theo kích thước tối đa
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])\s+|\.\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunks.append(" ".join(sentences[i : i + self.max_sentences_per_chunk]))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # Bắt đầu chia đệ quy với toàn bộ separator
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Hàm đệ quy chia nhỏ văn bản
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [current_text[i:i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        splits = current_text.split(separator) if separator else list(current_text)
        
        chunks = []
        current = ""
        for s in splits:
            part = s + separator if separator and s != splits[-1] else s
            if len(current) + len(part) <= self.chunk_size:
                current += part
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)

        result = []
        for c in chunks:
            if len(c) > self.chunk_size:
                result.extend(self._split(c, remaining_separators[1:]))
            else:
                result.append(c)
        return result


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # Tính cosine similarity giữa 2 vector
    dot_product = _dot(vec_a, vec_b)
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Chạy và so sánh các bộ chunker
        fixed = FixedSizeChunker(chunk_size=chunk_size).chunk(text)
        sentence = SentenceChunker().chunk(text)
        recursive = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def avg_len(chunks):
            return sum(len(c) for c in chunks) / len(chunks) if chunks else 0

        return {
            "fixed_size": {"count": len(fixed), "avg_length": avg_len(fixed), "chunks": fixed},
            "by_sentences": {"count": len(sentence), "avg_length": avg_len(sentence), "chunks": sentence},
            "recursive": {"count": len(recursive), "avg_length": avg_len(recursive), "chunks": recursive},
        }
