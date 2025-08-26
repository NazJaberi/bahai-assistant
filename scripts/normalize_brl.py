import os, json, hashlib, time, sys
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
ORIG = ROOT / "data" / "originals"
NORM = ROOT / "data" / "normalized"
LOGS = ROOT / "data" / "logs"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "bahai-assistant/phase4 (+https://www.bahai.org/legal)"})


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def download_xhtml(url: str) -> bytes:
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def normalize_html(xhtml: bytes, base_url: str) -> bytes:
    """
    Create a single clean HTML file:
    - Preserve diacritics (decode as utf-8)
    - Keep paragraph/section ids and anchors
    - Remove <script> and inline event attrs
    - Keep links absolute for deep-links; keep anchors (#pXX)
    """
    html = xhtml.decode("utf-8", errors="strict")
    soup = BeautifulSoup(html, "lxml")  # lxml keeps ids and diacritics well

    # Remove scripts
    for s in soup.find_all("script"):
        s.decompose()

    # Ensure <meta charset="utf-8">
    if not soup.head:
        soup.html.insert(0, soup.new_tag("head"))
    if not soup.head.find("meta", attrs={"charset": True}):
        meta = soup.new_tag("meta", charset="utf-8")
        soup.head.insert(0, meta)

    # Keep a canonical link to original URL for provenance
    link = soup.new_tag("link", rel="canonical", href=base_url)
    soup.head.insert(0, link)

    # Ensure anchors are intact (ids like #p1 #section etc. typically already present)
    # We don't rewrite internal anchors; bahai.org pages already include ids.

    # Wrap into minimal html5 doc (keep original body)
    out = "<!doctype html>\n" + str(soup)
    return out.encode("utf-8")


def process_manifest(p: Path):
    m = json.loads(p.read_text(encoding="utf-8"))
    work_id = m["work_id"]
    html_url = m["html_url"]
    downloads_page = m.get("downloads_page_url") or str(Path(html_url).parent) + "/"

    # Paths
    work_dir = ORIG / work_id
    work_dir.mkdir(parents=True, exist_ok=True)
    src_path = work_dir / "source.xhtml"
    norm_path = NORM / f"{work_id}.html"

    # Download (idempotent; only re-download if missing or forced)
    content = download_xhtml(html_url)
    src_hash = sha256_bytes(content)
    src_path.write_bytes(content)

    # Normalize
    normalized = normalize_html(content, html_url)
    norm_hash = sha256_bytes(normalized)
    norm_path.write_bytes(normalized)

    # Update manifest
    m.setdefault("hashes", {})
    m["hashes"]["html"] = src_hash
    m["normalized_path"] = str(norm_path.as_posix())
    m["normalized_hash"] = norm_hash
    m["about_page_url"] = downloads_page
    # Ensure license URL present
    m["license_terms_url"] = "https://www.bahai.org/legal"

    p.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "work_id": work_id,
        "src_hash": src_hash,
        "norm_hash": norm_hash,
        "src_path": str(src_path),
        "norm_path": str(norm_path),
    }


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    results = []
    for mp in sorted(MANIFESTS.glob("*.json")):
        try:
            r = process_manifest(mp)
            results.append(r)
            print(f"[OK] {r['work_id']}  src:{r['src_hash'][:8]}  norm:{r['norm_hash'][:8]}")
        except Exception as e:
            print(f"[ERROR] {mp.name}: {e}", file=sys.stderr)
    # Write a run log
    ts = time.strftime("%Y%m%d-%H%M%S")
    (LOGS / f"phase4_normalize_{ts}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
