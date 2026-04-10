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


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i : i + self.max_sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)
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
        return self._split(text, self.separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            return FixedSizeChunker(chunk_size=self.chunk_size, overlap=0).chunk(text)

        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        final_chunks: list[str] = []
        current_chunk = ""
        
        for part in splits:
            if not part and separator != "":
                continue
                
            if len(part) > self.chunk_size:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                res = self._split(part, remaining_separators)
                final_chunks.extend(res)
            else:
                potential_chunk = (current_chunk + separator + part) if current_chunk else part
                if len(potential_chunk) <= self.chunk_size:
                    current_chunk = potential_chunk
                else:
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                    current_chunk = part

        if current_chunk:
            final_chunks.append(current_chunk.strip())

        return [c for c in final_chunks if c]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0
    
    dot_prod = _dot(vec_a, vec_b)
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0
        
    return dot_prod / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        results = {}
        
        from typing import Union
        strategies: dict[str, Union[FixedSizeChunker, SentenceChunker, RecursiveChunker]] = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=2), # Arbitrary choice for comparison
            "recursive": RecursiveChunker(chunk_size=chunk_size)
        }
        
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0
            results[name] = {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks
            }
            
        return results


class DocumentStructureChunker:
    """
    Split text based on document structure (Markdown headers).
    Identifies lines starting with #, ##, or ### as section boundaries.
    """

    def __init__(self, header_patterns: list[str] | None = None) -> None:
        self.header_patterns = header_patterns or [r"^#\s+", r"^##\s+", r"^###\s+"]

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        lines = text.split("\n")
        chunks: list[str] = []
        current_chunk: list[str] = []

        header_regex = "|".join(self.header_patterns)

        for line in lines:
            if re.match(header_regex, line):
                if current_chunk:
                    chunks.append("\n".join(current_chunk).strip())
                current_chunk = [line]
            else:
                current_chunk.append(line)

        if current_chunk:
            chunks.append("\n".join(current_chunk).strip())

        return [c for c in chunks if c]

