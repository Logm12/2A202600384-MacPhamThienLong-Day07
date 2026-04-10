import re

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
