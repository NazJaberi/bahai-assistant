import os
from openai import OpenAI
from pymilvus import connections, Collection

# Try to import search request helpers available in some 2.6 builds
HAVE_SR=False
try:
    from pymilvus.search_requests import AnnSearchRequest, SparseSearchRequest, RRFRanker
    HAVE_SR=True
except Exception:
    pass

client = OpenAI()
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
assert ZILLIZ_URI and ZILLIZ_TOKEN, "Set ZILLIZ_URI and ZILLIZ_TOKEN"

def embed(q):
    return client.embeddings.create(model="text-embedding-3-large", input=[q]).data[0].embedding

def print_hits(label, hits):
    print(f"\n=== {label} ===")
    for i, hit in enumerate(hits[0], start=1):
        print(f"{i}. id={hit.id} score={hit.distance:.4f}")
        print("   work_id   :", hit.fields.get("work_id"))
        print("   source_url:", hit.fields.get("source_url"))
        snippet = (hit.fields.get("text") or "")[:200].replace("\n"," ")
        print("   text      :", snippet, "...\n")

def main():
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    col = Collection("brl_chunks"); col.load()

    q = "What is the Most Great Peace?"
    e = embed(q)

    # Dense only (COSINE)
    dense = col.search(
        data=[e],
        anns_field="text_dense",
        param={"metric_type":"COSINE","params":{"nprobe":16}},
        limit=5,
        output_fields=["work_id","source_url","text"]
    )
    print_hits("Dense only (COSINE)", dense)

    if HAVE_SR:
        # Server-side BM25 and Hybrid with RRF
        bm25_req = SparseSearchRequest(field_name="text", data=q, params={"type":"bm25","limit":20})
        dense_req = AnnSearchRequest([e], "text_dense", {"metric_type":"COSINE","params":{"nprobe":16}}, limit=20)

        bm25 = col.hybrid_search(
            reqs=[bm25_req],
            rerank=RRFRanker(),
            limit=5,
            output_fields=["work_id","source_url","text"]
        )
        print_hits("BM25 only", bm25)

        fused = col.hybrid_search(
            reqs=[dense_req, bm25_req],
            rerank=RRFRanker(),
            limit=5,
            output_fields=["work_id","source_url","text"]
        )
        print_hits("Hybrid (RRF)", fused)
    else:
        print("\n(No SparseSearchRequest in this wheel; dense works. If you want, I can add a client-side BM25+RRF fallback.)")

if __name__=="__main__":
    main()
