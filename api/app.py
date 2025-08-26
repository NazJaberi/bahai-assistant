import os, glob, json
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from api.fusion_generic import pick_with_fusion
from api.synthesis_rules import system_hint_for
from pydantic import BaseModel
from openai import OpenAI
from pymilvus import connections, Collection
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Bahai Assistant API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Optional hybrid helpers (if pymilvus has them)
HAVE_SR=False
try:
    from pymilvus.search_requests import AnnSearchRequest, SparseSearchRequest, RRFRanker
    HAVE_SR=True
except Exception:
    pass

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROMPT_ID = os.getenv("PROMPT_ID")
ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
assert OPENAI_API_KEY and PROMPT_ID and ZILLIZ_URI and ZILLIZ_TOKEN, "Missing required env vars"

client = OpenAI()

# Connect to Zilliz
connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN, timeout=30)
COL = Collection("brl_chunks")
COL.load()

# Load parents into memory for expansion
PARENTS: Dict[str, Dict[str, Any]] = {}
for path in glob.glob("data/exports/*_parents.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            PARENTS[r["id"]] = r

class SearchRequest(BaseModel):
    query: str
    k: int = 6
    work_id: str | None = None

class Passage(BaseModel):
    id: str
    parent_id: str | None = None
    work_id: str
    work_title: str | None = None
    paragraph_id: str | None = None
    text: str
    source_url: str | None = None
    score: float | None = None

class SearchResponse(BaseModel):
    results: List[Passage]
    used_mode: str

def embed(text: str) -> List[float]:
    return client.embeddings.create(model="text-embedding-3-large", input=[text]).data[0].embedding

def _hits_to_passages(hits, limit=6):
    out=[]
    for i, hit in enumerate(hits[0][:limit], start=1):
        out.append(Passage(
            id=hit.id,
            parent_id=hit.fields.get("parent_id"),
            work_id=hit.fields.get("work_id"),
            work_title=hit.fields.get("work_title"),
            paragraph_id=hit.fields.get("paragraph_id"),
            text=hit.fields.get("text") or "",
            source_url=hit.fields.get("source_url"),
            score=float(hit.distance) if hasattr(hit, "distance") else None,
        ))
    return out

def dense_search(q: str, k: int, expr: str | None):
    e = embed(q)
    res = COL.search(
        data=[e],
        anns_field="text_dense",
        param={"metric_type":"COSINE","params":{"nprobe":16}},
        limit=k,
        output_fields=["parent_id","work_id","work_title","paragraph_id","text","source_url"],
        expr=expr
    )
    return _hits_to_passages(res, limit=max(120, k))

def hybrid_rrf(q: str, k: int, expr: str | None):
    e = embed(q)
    dense_req = AnnSearchRequest([e], "text_dense", {"metric_type":"COSINE","params":{"nprobe":16}}, limit=max(k*3, 20), expr=expr)
    bm25_req = SparseSearchRequest("text", q, params={"type":"bm25","limit":max(k*3, 20)}, expr=expr)
    fused = COL.hybrid_search(
        reqs=[dense_req, bm25_req],
        rerank=RRFRanker(),
        limit=k,
        output_fields=["parent_id","work_id","work_title","paragraph_id","text","source_url"]
    )
    return _hits_to_passages(fused, limit=max(120, k))

def build_expr(work_id: str | None):
    if not work_id:
        return None
    return f'work_id == "{work_id}"'

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    expr = build_expr(req.work_id)
    if HAVE_SR:
        try:
            results = hybrid_rrf(req.query, req.k, expr)
            return SearchResponse(results=results, used_mode="hybrid_rrf")
        except Exception:
            results = dense_search(req.query, req.k, expr)
            return SearchResponse(results=results, used_mode="dense_only")
    else:
        results = dense_search(req.query, req.k, expr)
        return SearchResponse(results=results, used_mode="dense_only")

class AnswerRequest(BaseModel):
    query: str
    k: int = 6
    work_id: str | None = None

class Citation(BaseModel):
    work_title: str
    paragraph_id: str | None
    source_url: str
    work_id: str

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    context_preview: List[str]
    used_mode: str

DISCLAIMER = (
    "This assistant retrieves and cites passages from the Bahá’í writings. "
    "It does not issue rulings or speak with authority. See bahai.org/legal."
)

# Inline fallback system instructions
SYSTEM_INSTRUCTIONS = """
---  # 📖 Bahá’í Study Assistant – System Instructions

You are an **AI-powered study assistant** designed to help users explore the Bahá’í writings warmly and supportively. Your purpose is to provide **referenced source texts**, approachable explanations, and resource guidance.  

⚠️ You are **not an official Bahá’í authority, clergy, or representative**. Always make this clear to users. Present yourself as a **friendly companion for learning**.

---  

## 🔹 Core Guidelines

* **Identity:**
   * Introduce yourself as a *study assistant*.
   * Clarify you do not represent Bahá’í institutions.
   * Encourage users to continue their own investigation.
* **Referencing:**
   * Always provide direct **quotes from the Bahá’í writings** (or related resources).
   * Always include **source details**: book title, author, and page/section if available—but do not mention file types (such as PDF, TXT, etc) in your citations.
   * If metadata is incomplete in retrieved text chunks, give what is available and note this transparently.
   * **If a user asks for longer passages, extended selections, tablets, or full prayers (such as for a multiple-reading program or devotional), you must check for, and provide, the entire text, confirming that the portion you deliver is complete. If a passage continues in additional retrieved text chunks, you should gather and combine them to present the full, contiguous quotation or prayer as requested, with proper referencing.**
* **Explanations:**
   * After citing, give a **friendly, clear summary** of what the text means in simple language.
   * Note when explanations are interpretive and encourage the user to consult the full text.
   * Avoid dogmatic statements, personal interpretations, or definitive judgments.
* **Tone:**
   * Neutral, respectful, and approachable.
   * Encourage open-minded inquiry and warmth.
   * Never dismiss questions—if insufficient material exists, acknowledge limits and suggest further study.
* **Scope Handling:**
   * If multiple sources are relevant, present several key references prioritizing clarity and directness.
   * If sources differ in emphasis, gently note this and encourage users to consider the broader context.
   * If no references are found, state this kindly and suggest alternative exploration paths.
   * You should **only answer questions or discuss topics specifically relating to the Bahá’í Faith**. If asked about unrelated areas (such as math, science, coding, weather, etc.), politely decline to answer unless they directly connect to Bahá’í teachings or context.

---

## 🔹 Response Structure

For **every user question**, format your response as follows:

1. **Referenced Source(s):**
   * Quote(s) directly from the Bahá’í writings or other provided references.
   * Give full bibliographic details in this format:
     > *“The earth is but one country, and mankind its citizens.”*
     > — Bahá’u’lláh, *Gleanings from the Writings of Bahá’u’lláh*, p. 250
   * **If the user requests longer selections, a program, or asks for a full tablet, prayer, or passage (e.g., for devotionals or study), you must provide the entire text as completely as possible by gathering and combining any contiguous, relevant chunks, with proper citation. Always verify that the passage is complete.**

2. **Explanation:**
   * A plain-language, friendly explanation of the passage(s) in as much detail as needed for clarity.
   * Note when explanation is based on general understanding rather than explicit text.
   * Encourage the user to read the source in context for their own understanding.

3. **Disclaimer:**
   * Always include:
     > *I am an AI study assistant, not a representative of official Bahá’í institutions or clergy. Please view my responses as supportive guidance and continue your own exploration of the texts.*

---

## 🔹 Special Capability: Devotional/Program Planning

* Upon request, you can help **plan devotionals or study programs (e.g. Jy, Ruhi)** focused on a particular theme.
* **Programs should be detailed and, where appropriate, provide extended, full texts of prayers, tablets, or passages (not only excerpts), ensuring each text is complete by checking and assembling all associated text chunks.**
* Compile a list of prayers, suitable quotations, and brief related stories from Bahá’í history or texts.
* Arrange the content to suit the theme or specific needs stated by the user (e.g., children's class, youth, or community gathering).
* When assembling, ensure all sources are referenced and the selections are relevant to the requested theme.
* **If the user requests a longer or more comprehensive program, include extended passages, grouped texts, or longer paragraphs as appropriate to fulfill their request, verifying completeness as above.**

---

## 🔹 Final Notes

* **No strict length limit**—respond freely and use as much detail as is helpful, balancing referencing, explanation, and supportive guidance.
* Stay within the subject area of the Bahá’í Faith.
* Your ultimate role: **help users discover and reflect** on the Bahá’í writings, not to provide final answers.
"""

@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    sresp = search(SearchRequest(query=req.query, k=req.k, work_id=req.work_id))

    parent_texts = []
    for psg in sresp.results:
        if psg.parent_id and psg.parent_id in PARENTS:
            parent_texts.append(PARENTS[psg.parent_id]["text"])
        else:
            parent_texts.append(psg.text)

    citations = []
    context_snippets = []
    for psg in sresp.results:
        context_snippets.append(psg.text)
        if psg.source_url and psg.work_title:
            citations.append(Citation(
                work_title=psg.work_title,
                paragraph_id=psg.paragraph_id,
                source_url=psg.source_url,
                work_id=psg.work_id,
            ))

    prompt_vars = {
        "user_query": req.query,
        "disclaimer": DISCLAIMER,
        "passages": [
            {
                "text": psg.text,
                "work_title": psg.work_title,
                "paragraph_id": psg.paragraph_id,
                "source_url": psg.source_url,
                "work_id": psg.work_id,
            }
            for psg in sresp.results
        ],
        "parent_context": parent_texts[: req.k],
    }

    try:
        # Try PROMPT_ID path
        resp = client.responses.create(
            model="gpt-4.1",
            temperature=0.15,
            max_output_tokens=32000,
            prompt_id=PROMPT_ID,
            input=prompt_vars
        )
        answer_text = resp.output_text
    except TypeError:
        # Inline fallback
        USER = (
            f"User Query: {req.query}\n\n"
            "Passages:\n" +
            "\n\n".join(
                f"- {d['work_title']} ¶{d.get('paragraph_id') or ''} {d['source_url']}\n{d['text']}"
                for d in prompt_vars["passages"]
            ) +
            "\n\nParent Context (for background only):\n" +
            "\n\n---\n\n".join(parent_texts[: req.k])
        )

        resp = client.responses.create(
            model="gpt-4.1",
            temperature=0.15,
            max_output_tokens=32000,
            input=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": USER},
            ],
        )
        answer_text = resp.output_text
    except Exception:
        # Last resort fallback
        lines = [f"{DISCLAIMER}\n", f"**Query:** {req.query}\n"]
        if not sresp.results:
            lines.append("No strong matches were found.")
        else:
            lines.append("**Quoted passages:**")
            for psg in sresp.results[: req.k]:
                q = (psg.text or "").strip().replace("\n", " ")
                if len(q) > 400: q = q[:400] + "…"
                cite = f" — *{psg.work_title}*" + (f", ¶{psg.paragraph_id}" if psg.paragraph_id else "")
                link = f" ({psg.source_url})" if psg.source_url else ""
                lines.append(f"“{q}”{cite}{link}")
        answer_text = "\n".join(lines)

    return AnswerResponse(
        answer=answer_text,
        citations=citations,
        context_preview=context_snippets,
        used_mode=sresp.used_mode,
    )

@app.get("/healthz")
def healthz():
    return {"ok": True}
