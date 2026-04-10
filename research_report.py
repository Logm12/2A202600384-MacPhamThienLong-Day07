import os
import sys
from dotenv import load_dotenv

# Đảm bảo load biến môi trường từ .env
load_dotenv(os.path.join(os.getcwd(), ".env"))
sys.path.append(os.getcwd())

from src.chunking import RecursiveChunker, compute_similarity, DocumentStructureChunker
from src.store import EmbeddingStore
from src.models import Document
from src.embeddings import OpenAIEmbedder

def analyze_data():
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Thư mục {data_dir} không tồn tại!")
        return
    files = [f for f in os.listdir(data_dir) if f.endswith('.md')]
    print("--- DATA INVENTORY ---")
    for f_name in files:
        with open(os.path.join(data_dir, f_name), 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"{f_name}|{len(content)}")

def similarity_tests():
    try:
        embedder = OpenAIEmbedder()
        print("\n--- SIMILARITY PREDICTIONS ---")
    except Exception as e:
        print(f"\nLỗi khởi tạo OpenAIEmbedder: {e}")
        return

    pairs = [
        ("Bệnh nhân bị sốt cao và đau bụng vùng hạ sườn phải.", "Triệu chứng phổ biến nhất là sốt và đau ở hạ sườn bên phải."),
        ("Amip xâm nhập vào cơ thể qua niêm mạc mũi.", "Ký sinh trùng đi vào người bằng đường hô hấp qua mũi."),
        ("Lãi suất tiền gửi tiết kiệm đang giảm mạnh.", "Tỷ lệ lạm phát đang có dấu hiệu hạ nhiệt."),
        ("Cần phẫu thuật để nạo vét hết các ổ mủ.", "Điều trị bằng thuốc kháng sinh liều cao."),
        ("Nước tiểu sẽ chuyển sang màu đen khi để ngoài không khí.", "Màu sắc của nước tiểu thay đổi do phản ứng oxy hóa với khí quyển.")
    ]
    
    for a, b in pairs:
        vec_a = embedder(a)
        vec_b = embedder(b)
        score = compute_similarity(vec_a, vec_b)
        print(f"A: {a}\nB: {b}\nScore: {score:.4f}\n")

def debug_q2():
    print("\n--- DEBUG Q2 (ÁP XE HẬU MÔN) ---")
    file_path = "data/ap-xe-hau-mon-3332.md"
    if not os.path.exists(file_path):
        print(f"File {file_path} không tồn tại!")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunker = DocumentStructureChunker()
    chunks = chunker.chunk(content)
    
    query = "Những nguyên nhân nào khiến tình trạng áp xe hậu môn dễ bị tái phát sau khi đã tiến hành điều trị?"
    try:
        embedder = OpenAIEmbedder()
        q_vec = embedder(query)
    except Exception as e:
        print(f"Lỗi: {e}")
        return
    
    scored_chunks = []
    for c in chunks:
        score = compute_similarity(q_vec, embedder(c))
        scored_chunks.append((score, c))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Total chunks from ap-xe-hau-mon-3332.md: {len(chunks)}")
    for i in range(min(3, len(scored_chunks))):
        score, text = scored_chunks[i]
        print(f"Top {i+1} Score: {score:.4f}")
        print(f"Text Snippet: {text[:200]}...\n")
        
    print("Checking keyword 'tái phát' in chunks:")
    for i, c in enumerate(chunks):
        if "tái phát" in c.lower():
            print(f"Chunk {i} contains keywords! Snippet: {c[:200]}...")

if __name__ == "__main__":
    analyze_data()
    similarity_tests()
    debug_q2()
