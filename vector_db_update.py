# vector_db.py
import json, uuid, glob, os
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import EmbeddingFunction

DB_PATH = "./chroma_db"
COLLECTION_NAME = "qna_collection"

# 1) 아래 둘 중 하나로 사용하세요
# (A) 글롭 패턴: 하위 폴더 포함 모든 jsonl
# (B) 명시적 리스트: 특정 파일만
JSONL_FILES = []  # ["generated_qa_people.jsonl", "more_qa.jsonl"]

TOP_K = 3
BATCH_SIZE = 1000  # 메모리/성능 균형용 배치 업서트

class BGEPassageEmbedding(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-m3", normalize=True, device=None):
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
    def __call__(self, texts):
        texts = [f"passage: {t}" for t in texts]
        embs = self.model.encode(texts, normalize_embeddings=self.normalize)
        return embs.tolist()

bge_model = SentenceTransformer("BAAI/bge-m3")
NORMALIZE = True

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=BGEPassageEmbedding("BAAI/bge-m3", normalize=NORMALIZE),
    metadata={"hnsw:space": "cosine"}
)

# 👇 리스트/딕셔너리를 문자열로 안전 변환
def to_meta_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)

def iter_jsonl_files():
    files = []
    if JSONL_FILES:
        files.extend(JSONL_FILES)
    if JSONL_GLOB:
        files.extend(glob.glob(JSONL_GLOB, recursive=True))
    # 중복 제거 + 존재하는 파일만
    seen = set()
    for p in files:
        p = os.path.abspath(p)
        if p not in seen and os.path.isfile(p):
            seen.add(p)
            yield p

def upsert_batch(docs, metadatas, ids):
    if not docs:
        return 0
    collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
    return len(ids)

def ingest_files():
    total = 0
    docs, ids, metadatas = [], [], []

    for filepath in iter_jsonl_files():
        print(f"\n📄 파일 처리 시작: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  ⚠️ JSON 파싱 실패 (line {ln}): {e}")
                    continue

                q = obj.get("question", "")
                a = obj.get("answer", "")
                _id = obj.get("id") or str(uuid.uuid4())

                raw_meta = {
                    "sources": obj.get("sources"),
                    "tags": obj.get("tags"),
                    "last_verified": obj.get("last_verified"),
                    "source_file": obj.get("source_file") or os.path.basename(filepath),
                }
                meta = {k: to_meta_value(v) for k, v in raw_meta.items()}

                docs.append(f"Q: {q}\nA: {a}")
                ids.append(_id)
                metadatas.append(meta)

                # 배치 업서트
                if len(docs) >= BATCH_SIZE:
                    n = upsert_batch(docs, metadatas, ids)
                    total += n
                    print(f"  ↳ 배치 업서트 완료: +{n} (누적 {total})")
                    docs, ids, metadatas = [], [], []

    # 잔여분 업서트
    n = upsert_batch(docs, metadatas, ids)
    total += n
    if n:
        print(f"  ↳ 마지막 배치 업서트: +{n} (누적 {total})")

    print(f"\n✅ 전체 업서트 완료: {total}개")
    return total

if __name__ == "__main__":
    # 데이터 적재
    ingest_files()

    # 조회 예시 (유사도 %로 표시)
    user_query = "CardDAV API에서 연락처를 업데이트하는 방법은?"
    query_emb = bge_model.encode([f"query: {user_query}"], normalize_embeddings=NORMALIZE).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=TOP_K)

    for i, (doc, meta, _id, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["ids"][0],
        results["distances"][0],
    ), start=1):
        similarity = max(0.0, min(1.0, 1 - dist))
        sim_pct = round(similarity * 100, 1)
        print(f"\n[{i}] id={_id}  similarity={sim_pct}%  (distance={dist:.4f})")
        print(doc)
        print("-> sources:", meta.get("sources"))
        print("-> tags:", meta.get("tags"))
        print("-> last_verified:", meta.get("last_verified"))
        print("-> source_file:", meta.get("source_file"))
