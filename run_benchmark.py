import os
import sys

from src.models import Document
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from dotenv import load_dotenv
import re
import builtins

load_dotenv()

class DocumentStructureChunker:
    def __init__(self, header_patterns=None):
        self.header_patterns = header_patterns or [r'^#\s+', r'^##\s+', r'^###\s+']

    def chunk(self, text: str) -> list[str]:
        if not text: return []
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        header_regex = "|".join(self.header_patterns)
        for line in lines:
            if re.match(header_regex, line):
                if current_chunk: chunks.append("\n".join(current_chunk).strip())
                current_chunk = [line]
            else:
                current_chunk.append(line)
        if current_chunk: chunks.append("\n".join(current_chunk).strip())
        return [c for c in chunks if c]

def run_benchmarks():
    data_dir = "data"
    
    # Init real OpenAI embedder
    try:
        from src.embeddings import OpenAIEmbedder
        embedder = OpenAIEmbedder()
        print("Using OpenAIEmbedder initialized via API key.")
    except Exception as e:
        print("Failed to initialize embedder", e)
        return

    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=embedder)
    chunker = DocumentStructureChunker()
    docs = []
    
    print("Chunking documents...")
    for filename in os.listdir(data_dir):
        if not filename.endswith(".md"): continue
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = chunker.chunk(text)
            for i, chunk in enumerate(chunks):
                docs.append(Document(id=f"{filename}_{i}", content=chunk, metadata={"source": filename}))
                
    print(f"Adding {len(docs)} chunks to Vector Store...")
    store.add_documents(docs)
    
    # LLM function using OpenAI Chat Completion
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def openai_llm_fn(prompt: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ngắn gọn trả lời câu hỏi dựa trên Context được cho."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
            
    agent = KnowledgeBaseAgent(store=store, llm_fn=openai_llm_fn)
    
    queries = [
        "Bệnh Amip ăn não (Naegleria fowleri) lây nhiễm vào cơ thể qua đường nào và thời gian ủ bệnh thường là bao lâu?",
        "Những nguyên nhân nào khiến tình trạng áp xe hậu môn dễ bị tái phát sau khi đã tiến hành điều trị?",
        "Tam chứng Fontam điển hình trong bệnh áp xe gan do amip bao gồm những biểu hiện lâm sàng nào?",
        "Bệnh Alkapton niệu (nước tiểu sẫm màu) hình thành do đột biến gen nào và tại sao nước tiểu của người bệnh lại chuyển màu đen?",
        "Các phương pháp chính được sử dụng để điều trị hội chứng rối loạn ám ảnh sợ hãi là gì?"
    ]
    
    print("\n--- RESULTS ---")
    for i, q in enumerate(queries):
        top_chunks = store.search(q, top_k=3)
        top1_content = top_chunks[0]['content'] if top_chunks else "N/A"
        top1_short = top1_content[:150].replace('\n', ' ') + "..."
        score = top_chunks[0]['score'] if top_chunks else 0.0
        
        ans = agent.answer(q, top_k=3)
        
        print(f"\nQ{i+1}: {q}")
        print(f"Top 1 Score: {score:.3f}")
        print(f"Top 1 Chunk: {top1_short}")
        print(f"Agent Ans : {ans}")

if __name__ == "__main__":
    run_benchmarks()
