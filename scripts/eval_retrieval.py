import csv, json, os, statistics, time
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import connections, Collection
from openai import OpenAI

load_dotenv()
ZILLIZ_URI=os.getenv("ZILLIZ_URI"); ZILLIZ_TOKEN=os.getenv("ZILLIZ_TOKEN")
assert ZILLIZ_URI and ZILLIZ_TOKEN, "ZILLIZ_URI/ZILLIZ_TOKEN missing"
client = OpenAI()

def embed(q):
    return client.embeddings.create(model="text-embedding-3-large", input=[q]).data[0].embedding

def dense_search(col, q, k=10):
    e = embed(q)
    return col.search([e], "text_dense", {"metric_type":"COSINE","params":{"nprobe":16}}, limit=k,
                      output_fields=["work_id","text","source_url","work_title","paragraph_id"])

def hit_at_k(expected, hits, k):
    expected_set=set(x.strip() for x in expected.split("|"))
    for i, hit in enumerate(hits[0][:k], start=1):
        if hit.fields.get("work_id") in expected_set:
            return 1
    return 0

def mrr_at_k(expected, hits, k):
    expected_set=set(x.strip() for x in expected.split("|"))
    for i, hit in enumerate(hits[0][:k], start=1):
        if hit.fields.get("work_id") in expected_set:
            return 1.0/i
    return 0.0

def main():
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
    col = Collection("brl_chunks"); col.load()

    rows=list(csv.DictReader(open("eval/golden_set.csv", newline="", encoding="utf-8")))
    results=[]
    for r in rows:
        q=r["question"]; expected=r["expected_work_id"]
        hits=dense_search(col,q,k=10)
        h5=hit_at_k(expected,hits,5); h10=hit_at_k(expected,hits,10)
        m5=mrr_at_k(expected,hits,5);  m10=mrr_at_k(expected,hits,10)
        results.append({"question":q,"expected":expected,"hit@5":h5,"hit@10":h10,"mrr@5":m5,"mrr@10":m10})

    rpt={
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n": len(rows),
        "hit@5": statistics.mean([r["hit@5"] for r in results]) if results else 0.0,
        "hit@10": statistics.mean([r["hit@10"] for r in results]) if results else 0.0,
        "mrr@5": statistics.mean([r["mrr@5"] for r in results]) if results else 0.0,
        "mrr@10": statistics.mean([r["mrr@10"] for r in results]) if results else 0.0,
        "details": results
    }
    Path("eval").mkdir(exist_ok=True, parents=True)
    Path("eval/report.json").write_text(json.dumps(rpt, indent=2), encoding="utf-8")
    print(json.dumps(rpt, indent=2))

if __name__=="__main__":
    main()
