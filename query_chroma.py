# 조회 전용: query_chroma.py
import argparse
import chromadb
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default="./chroma_db")
    parser.add_argument("--collection", default="qna_collection")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--query", required=True, help="사용자 질문(검색어)")
    args = parser.parse_args()

    # 1) 질의 임베딩용 모델 (bge / query 프리픽스)
    model = SentenceTransformer("BAAI/bge-m3")
    query_vec = model.encode([f"query: {args.query}"], normalize_embeddings=True).tolist()

    # 2) Chroma 클라이언트 & 컬렉션 가져오기
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection(name=args.collection)

    # 3) 검색 실행 (임베딩 기반)
    res = collection.query(query_embeddings=query_vec, n_results=args.top_k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not docs:
        print("검색 결과가 없습니다.")
        return

    # 4) 결과 출력 (distance -> similarity %)
    for i, (doc, meta, _id, dist) in enumerate(zip(docs, metas, ids, dists), start=1):
        sim = max(0.0, min(1.0, 1 - dist))
        sim_pct = round(sim * 100, 1)

        print(f"\n[{i}] id={_id}  similarity={sim_pct}%  (distance={dist:.4f})")
        print(doc)

        # 메타데이터 보조 출력 (sources/tags가 직렬화된 문자열일 수 있음)
        q = meta.get("question")
        a = meta.get("answer")
        sources = meta.get("sources")
        tags = meta.get("tags")

        print("-> Q:", q)
        print("-> A:", a)
        print("-> sources:", sources)
        print("-> tags:", tags)
        print("-> last_verified:", meta.get("last_verified"))
        print("-> source_file:", meta.get("source_file"))

if __name__ == "__main__":
    main()
