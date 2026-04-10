from src.chunking import ChunkingStrategyComparator
import glob

def main():
    files_to_test = glob.glob("markdown/*.md")[:2]
    comparator = ChunkingStrategyComparator()
    
    with open("results.txt", "w", encoding="utf-8") as out:
        for file_path in files_to_test:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            stats = comparator.compare(text)
            
            file_name = file_path.replace("\\", "/")
            out.write(f"| {file_name} | FixedSizeChunker (`fixed_size`) | {stats['fixed_size']['count']} | {stats['fixed_size']['avg_length']:.1f} | Khong (de bi cat) |\n")
            out.write(f"| | SentenceChunker (`by_sentences`) | {stats['by_sentences']['count']} | {stats['by_sentences']['avg_length']:.1f} | Tot (Giap nguyen cau) |\n")
            out.write(f"| | RecursiveChunker (`recursive`) | {stats['recursive']['count']} | {stats['recursive']['avg_length']:.1f} | Rat tot (Theo noi dung) |\n")

if __name__ == "__main__":
    main()
