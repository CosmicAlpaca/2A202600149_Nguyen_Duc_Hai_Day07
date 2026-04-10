import chromadb.utils.embedding_functions as emb_fns
from src.chunking import compute_similarity

# Khởi tạo mô hình embed cục bộ để encode văn bản (giống hệt RAG pipeline)
default_ef = emb_fns.DefaultEmbeddingFunction()

def embed_fn(text: str) -> list[float]:
    return default_ef([text])[0]

print("\n--- TEST ĐỘ TƯƠNG ĐỒNG COSINE (BÀI 5) ---")
print("Bạn có thể nhập cặp câu A và B tự do. Gõ 'q' hoặc 'exit' ở ô nhập để thoát.\n")

while True:
    try:
        sent_a = input("\n📝 Nhập Câu A: ")
        if sent_a.strip().lower() in ['q', 'quit', 'exit']:
            print("Tạm biệt!")
            break
            
        sent_b = input("📝 Nhập Câu B: ")
        if sent_b.strip().lower() in ['q', 'quit', 'exit']:
            print("Tạm biệt!")
            break
            
        if not sent_a.strip() or not sent_b.strip():
            print("❌ Vui lòng bảo đảm câu nhập không để trống!")
            continue

        print("Đang tính toán vector...")
        vec_a = embed_fn(sent_a)
        vec_b = embed_fn(sent_b)
        
        score = compute_similarity(vec_a, vec_b)
        print(f"👉 Actual Score (Độ tương đồng): {score:.4f}")
        
        if score > 0.6:
            print("💡 Dự đoán NLP: Nhóm High (Cao)")
        else:
            print("💡 Dự đoán NLP: Nhóm Low (Thấp)")
            
    except KeyboardInterrupt:
        print("\nTạm biệt!")
        break
    except Exception as e:
        print(f"❌ Lỗi: {e}")
