import os
import glob
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb.utils.embedding_functions as emb_fns
from src.models import Document
from src.chunking import MarkdownChunker
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent

# Load ENVs
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))

default_ef = emb_fns.DefaultEmbeddingFunction()

def embed_fn(text: str) -> list[float]:
    return default_ef([text])[0]

def llm_fn(prompt: str) -> str:
    # Set default to lite due to API limits
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Lỗi API Gemini - Quota hoặc Network] {e}"

def main() -> int:
    markdown_files = glob.glob("markdown/*.md")
    
    if not markdown_files:
        print("Không tìm thấy file nào trong thư mục 'markdown/'. Vui lòng kiểm tra lại!")
        return 1
        
    docs = []
    chunker = MarkdownChunker()
    idx = 0
    
    print(f"Đang đọc {len(markdown_files)} file markdown...")
    for file in markdown_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = chunker.chunk(content)
            for c in chunks:
                idx += 1
                docs.append(Document(
                    id=f"doc_{idx}",
                    content=c,
                    metadata={"source": file}
                ))

    store = EmbeddingStore(collection_name="tiki_docs_hf_main", embedding_fn=embed_fn)
    print(f"Đang thêm {len(docs)} chunks vào ChromaDB (Local Embeddings)...")
    store.add_documents(docs)
    
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    
    print("\n" + "="*80)
    print("CHẾ ĐỘ HỘI THOẠI NGHIỆP VỤ TIKI (RAG)")
    print("Hệ thống đã nạp 10 file tài liệu. Bạn có thể tự đặt câu hỏi để test.")
    print("Gõ 'q' hoặc 'exit' để dừng và lấy mẫu kết quả cuối cùng.")
    print("="*80 + "\n")
    
    query_count = 0
    while True:
        try:
            q = input("\n👤 Bạn hỏi: ")
            if q.strip().lower() in ['q', 'exit', 'quit']:
                break
            if not q.strip():
                continue
            
            query_count += 1
            print("🔍 Đang tìm kiếm tài liệu liên quan (Top-5)...")
            chunks = store.search(q, top_k=5)
            
            # Hiển thị chunk tốt nhất để user biết search trúng file nào
            if chunks:
                best = chunks[0]
                print(f"[*] File gốc: {best['metadata'].get('source')}")
                print(f"[*] Score: {best.get('score', 0.0):.4f}")
                print(f"[*] Đoạn trích tiêu biểu: {best['content'][:150]}...\n")
            
            print("🤖 Đang suy luận câu trả lời...")
            # Tăng top_k lên 5 để AI có nhiều context hơn, tránh việc 'không tìm thấy thông tin'
            ans = agent.answer(q, top_k=5)
            print("-" * 40)
            print(f"TRỢ LÝ: {ans}")
            print("-" * 40)
            
            # Format sẵn một dòng cho Table Report để user tiện copy nếu muốn
            chunk_summary = chunks[0]['content'][:120].replace('\n', ' ') if chunks else "N/A"
            ans_summary = ans[:150].replace('\n', ' ')
            print(f"\n[Dòng cho Table]: | {query_count} | {q} | {chunk_summary}... | {chunks[0].get('score', 0):.4f} | Yes | {ans_summary}... |")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Có lỗi: {e}")

    print("\nKết thúc phiên hội thoại. Cảm ơn bạn!")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
