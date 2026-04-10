import os
import glob
from dotenv import load_dotenv
import chromadb.utils.embedding_functions as emb_fns
from src.chunking import MarkdownChunker
from src.store import EmbeddingStore
from src.models import Document

# Cấu hình môi trường
load_dotenv()
default_ef = emb_fns.DefaultEmbeddingFunction()

def embed_fn(text: str) -> list[float]:
    return default_ef([text])[0]

def main():
    # 1. Load và Chunk dữ liệu bằng Strategy của bạn
    markdown_files = glob.glob("markdown/*.md")
    chunker = MarkdownChunker()
    docs = []
    idx = 0
    
    for file in markdown_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = chunker.chunk(content)
            for c in chunks:
                idx += 1
                docs.append(Document(id=f"doc_{idx}", content=c, metadata={"source": file}))

    # 2. Lưu vào Store
    store = EmbeddingStore(collection_name="eval_store", embedding_fn=embed_fn)
    store.add_documents(docs)

    # 3. Benchmark Queries
    queries = [
        "Sau khi kiện trả hàng tới nhà bán thì bao lâu Tiki xử lý hoàn tiền?",
        "Nếu tôi gửi sản phẩm bảo hành về Tiki thì bao lâu nhận lại?",
        "Đơn giao từ nước ngoài mà giao không thành công thì Tiki giao lại mấy lần, giữ kho bao lâu?",
        "Muốn dùng Tiki Xu và mã giảm giá thì có điều kiện gì?",
        "Tôi có thể yêu cầu giao vào giờ cụ thể hoặc hẹn lại chủ nhật không?"
    ]

    print("\n" + "-"*30)
    print("KIỂM TRA HIỆU SUẤT TRUY XUẤT (RETRIEVAL)")
    print("-"*30)

    relevant_hits = 0
    # Ngưỡng score chấp nhận được cho model local (thường > 0.3 là có liên quan)
    THRESHOLD = 0.35

    for i, q in enumerate(queries, 1):
        results = store.search(q, top_k=3) # Kiểm tra xem có nằm trong Top-3 không
        is_relevant = False
        
        if results:
            # Nếu Top-1 có score tốt, ta tạm coi là Hit
            if results[0].get('score', 0) > THRESHOLD:
                is_relevant = True
                relevant_hits += 1
        
        status = "✅ YES" if is_relevant else "❌ NO"
        score_val = results[0].get('score', 0) if results else 0
        print(f"Query {i}: {status} (Top-1 Score: {score_val:.4f})")

    # Tính điểm trên thang 10 (mỗi câu đúng được 2 điểm)
    retrieval_score = relevant_hits * 2

    print("\n" + "="*50)
    print(f"🔥 KẾT QUẢ CUỐI CÙNG:")
    print(f"Strategy: MarkdownChunker")
    print(f"Retrieval Score: {retrieval_score}/10")
    print("="*50)
    
    print(f"\n[Dòng cho Report]: | MarkdownChunker | {retrieval_score}/10 |")

if __name__ == "__main__":
    main()
