import os
from pymilvus import connections, Collection
from openai import OpenAI

client = OpenAI()
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
assert ZILLIZ_URI and ZILLIZ_TOKEN, "Set ZILLIZ_URI and ZILLIZ_TOKEN in your env/.env"

def embed(q):
    return client.embeddings.create(model="text-embedding-3-large", input=[q]).data[0].embedding

def main():
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    col = Collection("brl_chunks")
    col.load()

    q = "What is the Most Great Peace?"
    e = embed(q)

    # NOTE: your index expects COSINE (earlier error showed IP vs COSINE mismatch)
    res = col.search(
        data=[e],
        anns_field="text_dense",
        param={"metric_type":"COSINE","params":{"nprobe":16}},
        limit=5,
        output_fields=["work_id","source_url","text"]
    )

    print("=== Dense (COSINE) ===")
    # PyMilvus 2.6: iterate hits; use hit.fields to access output_fields
    for i, hit in enumerate(res[0], start=1):
        print(f"{i}. id={hit.id} score={hit.distance:.4f}")
        print("   work_id   :", hit.fields.get("work_id"))
        print("   source_url:", hit.fields.get("source_url"))
        snippet = (hit.fields.get("text") or "")[:200].replace("\n"," ")
        print("   text      :", snippet, "...\n")

if __name__=="__main__":
    main()
