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
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

default_ef = emb_fns.DefaultEmbeddingFunction()

def embed_fn(text: str) -> list[float]:
    return default_ef([text])[0]

def llm_fn(prompt: str) -> str:
    # Use gemini-1.5-flash or gemini-2.5-flash since they exist
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Fallback to gemini-2.5-flash-lite")
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        return response.text

def main():
    # Load and chunk documents
    markdown_files = glob.glob("markdown/*.md")
    docs = []
    chunker = MarkdownChunker()
    idx = 0
    
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

    # Store
    store = EmbeddingStore(collection_name="tiki_docs_hf", embedding_fn=embed_fn)
    print(f"Total chunks to add: {len(docs)}")
    store.add_documents(docs)
    
    # Agent
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm_fn)
    
    print("\n--- TIKI RAG TƯ VẤN CHĂM SÓC KHÁCH HÀNG ---")
    print("Hệ thống kho trí thức AI đã load thành công. (Gõ 'q' hoặc 'exit' để thoát)")
    
    while True:
        try:
            q = input("\n👤 Bạn hỏi: ")
            if q.strip().lower() in ['q', 'quit', 'exit']:
                print("Tạm biệt!")
                break
            if not q.strip():
                continue
                
            # Test Retrieval
            chunks = store.search(q, top_k=1)
            if chunks:
                chunk = chunks[0]
                print(f"[*] Text Context (tóm tắt): {chunk['content'][:200]}...")
                print(f"[*] Score (độ tương đồng): {chunk.get('score', 'N/A'):.4f}\n")
            
            # Test Agent
            print("Đang phân tích...")
            ans = agent.answer(q, top_k=3)
            print(f"🤖 Trợ lý: {ans}")
            
        except KeyboardInterrupt:
            print("\nTạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi xảy ra: {e}")

if __name__ == "__main__":
    main()
