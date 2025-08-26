# 1. Full Project `README.md`

```markdown
# Bahá’í Assistant – Retrieval-Augmented Generation (RAG) API

This repository implements a study assistant that retrieves from the Bahá’í Writings (stored in a Milvus/Zilliz vector DB) and generates grounded, cited answers using OpenAI’s `gpt-4.1`.

---

## 🚀 Project Overview

- **Backend:** FastAPI app exposing `/search` and `/answer` endpoints.
- **Vector DB:** Milvus/Zilliz Cloud storing chunked texts of the Bahá’í writings.
- **Embedding model:** `text-embedding-3-large` (1536-d).
- **Generation model:** `gpt-4.1` with low temperature and long context window (32k tokens).
- **Evaluation:** Golden set of queries + retrieval metrics.
- **Deployment:** Cloudflare Tunnel (ephemeral or named) for secure public API access.

---

## 📂 Project Structure

```

api/
app.py              # FastAPI app (main API logic)
fusion\_generic.py   # reranking / fusion helpers
synthesis\_rules.py  # system prompt helpers (optional)
data/
exports/            # JSONL exports (parents + children chunks)
scripts/
embed.py            # bulk embedding into Zilliz
test\_search.py      # test queries
test\_dense\_only.py  # dense-only search
eval\_retrieval.py   # retrieval evaluation loop
remove\_work.py      # delete a work from Milvus
embed\_one.py        # re-ingest a single work\_id
run\_api.sh          # run API via uvicorn
eval/
golden\_set.csv      # golden evaluation set

````

---

## ⚙️ Setup

```bash
# clone + enter
git clone <repo-url>
cd bahai-assistant

# create venv
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
````

### Required Environment Variables

Create `.env`:

```env
OPENAI_API_KEY=sk-...
PROMPT_ID=bahai_assistant_prompt   # or any saved prompt in OpenAI
ZILLIZ_URI=...
ZILLIZ_TOKEN=...
R2_BUCKET=bahai-texts
```

---

## 🔎 Ingest Data into Zilliz

```bash
# Embed all children JSONL into Milvus/Zilliz
python3 scripts/embed.py
```

---

## 🧪 Testing Retrieval

```bash
# Dense only test
python3 scripts/test_dense_only.py

# Hybrid test (if SparseSearchRequest available)
python3 scripts/test_search.py
```

---

## 📊 Evaluation (Phase 9)

Golden set defined in `eval/golden_set.csv`.
Run evaluation:

```bash
python3 scripts/eval_retrieval.py | tee eval/last_run.txt
```

Sync to Cloudflare R2:

```bash
rclone copy eval "r2:${R2_BUCKET}/eval" --create-empty-src-dirs
```

Produces `eval/report.json` with hit\@k and MRR metrics.

---

## 🛡️ Safety & Provenance (Phase 10)

* Each record carries `source_url` and license reference (`bahai.org/legal`).
* Removal of a work:

```bash
python3 scripts/remove_work.py peace
```

* Re-ingest after cleanup:

```bash
python3 scripts/embed_one.py peace
```

---

## 🌐 Deploy via Cloudflare Tunnel

### Ephemeral (quick demo)

```bash
sudo apt-get update && sudo apt-get install -y cloudflared
cloudflared tunnel --url http://127.0.0.1:8000
```

### Named Tunnel (stable domain)

```bash
cloudflared tunnel login
cloudflared tunnel create bahai-assistant

# config
mkdir -p ~/.cloudflared
nano ~/.cloudflared/config.yml
# paste:
tunnel: bahai-assistant
credentials-file: /home/$USER/.cloudflared/xxxx.json
ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404

# DNS route
cloudflared tunnel route dns bahai-assistant api.yourdomain.com

# run as service
sudo cloudflared service install
sudo systemctl restart cloudflared
```

Now open `https://api.yourdomain.com/docs`.

---

## 📖 System Instructions

The assistant is a **study companion**, not an authority.
It always:

* Provides verbatim quotes.
* Gives citations (title, section, link).
* Summarizes in plain language.
* Acknowledges its limits.
* Ends with a disclaimer:

  > *I am an AI study assistant, not a representative of official Bahá’í institutions or clergy. Please continue your own exploration of the texts.*

---

## ✅ Post-Deployment Checklist

* API starts on boot (`tmux`, `screen`, or systemd unit).
* Cloudflare tunnel service active.
* Evaluation metrics synced to R2.
* Logs optionally rotated or synced.

````

---

# 2. API Reference `README_API.md`

```markdown
# Bahá’í Assistant API Reference

This document describes how to call the API endpoints for the Bahá’í Assistant.

---

## 🔍 Endpoints

### 1. Health Check
**GET** `/healthz`

**Response:**
```json
{ "ok": true }
````

---

### 2. Search

**POST** `/search`

**Request Body:**

```json
{
  "query": "What is the Most Great Peace?",
  "k": 5
}
```

**Optional Fields:**

* `work_id`: limit results to one work.

**Response:**

```json
{
  "results": [
    {
      "id": "peace-c00054",
      "parent_id": null,
      "work_id": "peace",
      "work_title": "Peace",
      "paragraph_id": "",
      "text": "The Most Great Peace...",
      "source_url": "https://www.bahai.org/library/...",
      "score": 0.64
    }
  ],
  "used_mode": "dense_only"
}
```

---

### 3. Answer

**POST** `/answer`

Generates a cited answer using GPT-4.1.

**Request Body:**

```json
{
  "query": "Explain Huqúqu’lláh (how it works, when due, exemptions)—quote and cite.",
  "k": 8
}
```

**Response:**

```json
{
  "answer": "\"Huqúqu’lláh is a great law ...\"",
  "citations": [
    {
      "work_title": "Codification Law Huququllah",
      "paragraph_id": null,
      "source_url": "https://www.bahai.org/library/...",
      "work_id": "codification-law-huququllah"
    }
  ],
  "context_preview": [
    "Huqúqu’lláh is a great law and a sacred institution..."
  ],
  "used_mode": "dense_only"
}
```

---

## 📌 Notes for Developers

* All responses are **JSON**.
* **/search** returns raw retrieved passages.
* **/answer** runs retrieval → passes to GPT-4.1 → returns a **verbatim quoted answer with citations**.
* Set headers:

  ```http
  Content-Type: application/json
  ```
* Safe to call from browser or frontend app (CORS enabled).
* If no results are found, `answer` will return a fallback explanation with disclaimer.

---

## Example cURL Calls

### Health Check

```bash
curl -s http://127.0.0.1:8000/healthz
```

### Search

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the Lesser Peace?","k":5}' | jq
```

### Answer

```bash
curl -s -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain the Bahá’í law of fasting.","k":5}' | jq -r '.answer'
```

---

That’s it — your frontend dev just needs to POST to `/answer` with a query and show the `answer` + `citations`.

```
