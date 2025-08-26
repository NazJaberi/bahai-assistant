# Bahá’í Assistant Backend

This project ingests authoritative Bahá’í texts from [bahai.org](https://www.bahai.org/library/authoritative-texts/), chunks and embeds them, stores them in Milvus/Zilliz Cloud, and exposes a FastAPI-based retrieval/QA API.

The pipeline ensures every answer is **grounded in the texts**, with **verbatim quotes** and **clickable citations**.

---

## Table of Contents

* [Overview](#overview)
* [Architecture](#architecture)
* [Setup Phases](#setup-phases)

  * [Phase 0: Environment & Prereqs](#phase-0-environment--prereqs)
  * [Phase 1: Cloudflare R2 Storage](#phase-1-cloudflare-r2-storage)
  * [Phase 2: Data Layout](#phase-2-data-layout)
  * [Phase 3: Collect Text Sources](#phase-3-collect-text-sources)
  * [Phase 4: Export & Children Files](#phase-4-export--children-files)
  * [Phase 5: Chunking](#phase-5-chunking)
  * [Phase 6: Embedding & Upload](#phase-6-embedding--upload)
  * [Phase 7: API Build](#phase-7-api-build)
  * [Phase 8: API Polish](#phase-8-api-polish)
  * [Phase 9: Evaluation Loop](#phase-9-evaluation-loop)
* [Next Steps](#next-steps)
* [License](#license)

---

## Overview

**Goal**: Build a private retrieval + QA assistant over the Bahá’í Writings.

* **Sources**: Official XHTML texts from bahai.org.
* **Storage**: Cloudflare R2 (S3-compatible).
* **Vector DB**: Zilliz Cloud (Milvus).
* **Embeddings**: OpenAI `text-embedding-3-large`.
* **API**: FastAPI, with `/search` (retrieve passages) and `/answer` (compose answers with citations).
* **Eval**: Golden set with hit\@k / MRR\@k metrics.

---

## Architecture

```
bahai.org (XHTML) --> data/originals/
                     |
                     v
              normalization & parsing
                     |
                     v
         data/exports/*_children.jsonl
                     |
                     v
          Embeddings via OpenAI API
                     |
                     v
              Milvus/Zilliz Cloud
                     |
                     v
          FastAPI --> /search, /answer
```

* **Cloudflare R2** holds all pipeline artifacts (`originals/`, `normalized/`, `manifests/`, `exports/`, `logs/`, `eval/`).
* **Zilliz Cloud** stores the embedded chunks (`brl_chunks` collection).
* **FastAPI** exposes endpoints with citations.

---

## Setup Phases

### Phase 0: Environment & Prereqs

* Ubuntu 22.04 (WSL or server).
* Python 3.12 with venv:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
* Tools:

  ```bash
  sudo apt-get update && sudo apt-get install -y rclone jq
  ```

---

### Phase 1: Cloudflare R2 Storage

* Created R2 bucket: **`bahai-texts`**.
* Configured `rclone` with:

  ```bash
  rclone config
  # remote name: r2
  # type: s3
  # provider: Cloudflare
  # access key + secret key (admin token)
  # endpoint: https://<account_id>.r2.cloudflarestorage.com
  ```

Tested:

```bash
rclone lsf r2:bahai-texts/
```

---

### Phase 2: Data Layout

Created prefixes:

```
originals/
normalized/
manifests/
exports/
logs/
```

Verified with:

```bash
rclone lsf "r2:bahai-texts/" --dirs-only
```

---

### Phase 3: Collect Text Sources

* Used **only XHTML links** from [bahai.org](https://www.bahai.org).
* Collected \~70 works across Bahá’u’lláh, the Báb, ‘Abdu’l-Bahá, Shoghi Effendi, UHJ, Compilations, Prayers.
* Stored URLs in a manifest for ingestion.

---

### Phase 4: Export & Children Files

* Scraped XHTML → JSONL exports in `data/exports/`.

* Each `_children.jsonl` has chunk-level records:

  ```json
  {
    "id": "world-order-bahaullah-c00176",
    "parent_id": "world-order-bahaullah-p0123",
    "work_id": "world-order-bahaullah",
    "author": "Shoghi Effendi",
    "work_title": "World Order Bahaullah",
    "paragraph_id": "p0123",
    "text": "The Most Great Peace ...",
    "source_url": "https://www.bahai.org/.../xhtml",
    "lang": "en",
    "hash": "sha256..."
  }
  ```

* Validated with:

  ```bash
  head -n 3 data/exports/*_children.jsonl
  ```

---

### Phase 5: Chunking

* Ran `scripts/chunk_brl.py`.
* Produced \~10–700 children per work depending on length.
* Fixed missing `parent_id` issues.

---

### Phase 6: Embedding & Upload

* Embedded all children with OpenAI `text-embedding-3-large`.
* Inserted into Milvus/Zilliz `brl_chunks` collection.
* Verified with test search:

  ```bash
  python3 scripts/test_dense_only.py
  ```

Output example:

```
=== Dense (COSINE) ===
1. id=world-order-bahaullah-c00176 score=0.6729
   work_id   : world-order-bahaullah
   text      : "The Most Great Peace ..."
```

---

### Phase 7: API Build

* `api/app.py` (FastAPI).
* Endpoints:

  * `/healthz` → `{ "ok": true }`
  * `/search` → top passages (chunks).
  * `/answer` → LLM answer with quotes + citations.
* Start with:

  ```bash
  ./scripts/run_api.sh
  ```
* Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

### Phase 8: API Polish

* Added CORS middleware.
* Wrapper script `scripts/run_api.sh` loads `.env` and starts uvicorn.
* Curl tests:

  ```bash
  curl -s http://127.0.0.1:8000/healthz
  curl -s -X POST http://127.0.0.1:8000/search -d '{"query":"Most Great Peace"}'
  curl -s -X POST http://127.0.0.1:8000/answer -d '{"query":"Most Great Peace"}'
  ```

---

### Phase 9: Evaluation Loop

* Golden set: `eval/golden_set.csv` (\~25 Qs with expected work\_ids).
* Eval script: `scripts/eval_retrieval.py`.
* Run:

  ```bash
  python3 scripts/eval_retrieval.py | tee eval/last_run.txt
  rclone copy eval "r2:bahai-texts/eval" --create-empty-src-dirs
  ```

Example metrics:

```json
{
  "n": 25,
  "hit@5": 0.84,
  "hit@10": 0.92,
  "mrr@5": 0.693,
  "mrr@10": 0.702
}
```

---

## Next Steps

* **Phase 10: Safety & Provenance**

  * Verify every record has `source_url`.
  * Add/remove scripts (`remove_work.py`, `embed_one.py`).
  * Ensure disclaimer appears in `/answer`.

* **Deployment**

  * Run API under systemd.
  * Expose via Cloudflare Tunnel (`cloudflared tunnel ...`).
  * Optionally route to `api.yourdomain.com`.

---

## License

All authoritative texts © Bahá’í International Community. See [https://www.bahai.org/legal](https://www.bahai.org/legal) for license terms.

This project includes scripts for ingestion, retrieval, and QA. Redistribution of texts must respect the source license.


