import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)
from chunking import RecursiveChunker
import re

# 1. Declare DocumentStructureChunker
class DocumentStructureChunker:
    """
    Chunking based on document structure.
    """
    def __init__(self, header_patterns: list[str] | None = None):
        self.header_patterns = header_patterns or [r'^#\s+', r'^##\s+', r'^###\s+']

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
            
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        
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


# 2. Define compare function
def compare_strategies():
    files = [
        "aids-4699.md",
        "am-anh-so-hai-4678.md",
        "ap-xe-nao-3205.md"
    ]
    data_dir = os.path.join(current_dir, "data")
    
    baseline_chunker = RecursiveChunker(chunk_size=500)
    my_chunker = DocumentStructureChunker()
    
    print("===== STRATEGY COMPARISON =====")
    for filename in files:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        print(f"\n--- Document: {filename} ---")
        
        baseline_chunks = baseline_chunker.chunk(text)
        baseline_count = len(baseline_chunks)
        baseline_avg = sum(len(c) for c in baseline_chunks) / baseline_count if baseline_count > 0 else 0
        print(f"Best baseline (Recursive): Count = {baseline_count}, Avg length = {baseline_avg:.2f}")
        
        my_chunks = my_chunker.chunk(text)
        my_count = len(my_chunks)
        my_avg = sum(len(c) for c in my_chunks) / my_count if my_count > 0 else 0
        print(f"My strategy (Doc-structure): Count = {my_count}, Avg length = {my_avg:.2f}")

if __name__ == "__main__":
    compare_strategies()
