import json, hashlib, time
from pathlib import Path
from bs4 import BeautifulSoup
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
NORM = ROOT / "data" / "normalized"
EXPORTS = ROOT / "data" / "exports"
LOGS = ROOT / "data" / "logs"

ENC = tiktoken.get_encoding("cl100k_base")

CHILD_MIN, CHILD_MAX = 200, 380
PARENT_MIN, PARENT_MAX = 850, 1250

def ntoks(text): return len(ENC.encode(text))
def sha256(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()

def extract_blocks(html_path: Path):
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["header","nav","footer","script","style"]):
        tag.decompose()
    main = soup.find("main") or soup.body
    blocks=[]
    for el in main.find_all(["h1","h2","h3","h4","h5","h6","p","blockquote","li"], recursive=True):
        txt=el.get_text(" ", strip=True)
        if not txt: continue
        el_id=el.get("id") or ""
        base = soup.find("link", rel="canonical")
        base_href = base["href"] if base and base.has_attr("href") else ""
        src=f"{base_href}#{el_id}" if el_id and base_href else base_href or ""
        blocks.append((el.name.upper(), el_id, txt, src))
    return blocks

def group_children(blocks, work_id, author, title):
    children=[]
    buf_text, buf_ids, buf_sources=[],[],[]
    for kind, el_id, txt, src in blocks:
        if kind.startswith("H"):
            if buf_text and ntoks(" ".join(buf_text))>=CHILD_MIN:
                child_text="\n".join(buf_text)
                paragraph_id=next((i for i in buf_ids if i), "")
                source_url=next((s for s in buf_sources if s), "")
                children.append({
                    "id": f"{work_id}-c{len(children)+1:05d}",
                    "parent_id": "",
                    "work_id": work_id,
                    "author": author,
                    "work_title": title,
                    "section_id": "",
                    "paragraph_id": paragraph_id,
                    "text": child_text,
                    "source_url": source_url,
                    "lang": "en",
                    "hash": sha256(child_text)
                })
                buf_text,buf_ids,buf_sources=[],[],[]
            if ntoks(txt)<=40:
                buf_text.append(txt); buf_ids.append(el_id); buf_sources.append(src)
            continue
        buf_text.append(txt); buf_ids.append(el_id); buf_sources.append(src)
        toks=ntoks(" ".join(buf_text))
        if CHILD_MIN<=toks<=CHILD_MAX or toks>CHILD_MAX+80:
            child_text="\n".join(buf_text)
            paragraph_id=next((i for i in buf_ids if i), "")
            source_url=next((s for s in buf_sources if s), "")
            children.append({
                "id": f"{work_id}-c{len(children)+1:05d}",
                "parent_id": "",
                "work_id": work_id,
                "author": author,
                "work_title": title,
                "section_id": "",
                "paragraph_id": paragraph_id,
                "text": child_text,
                "source_url": source_url,
                "lang": "en",
                "hash": sha256(child_text)
            })
            buf_text,buf_ids,buf_sources=[],[],[]
    if buf_text:
        child_text="\n".join(buf_text)
        paragraph_id=next((i for i in buf_ids if i), "")
        source_url=next((s for s in buf_sources if s), "")
        children.append({
            "id": f"{work_id}-c{len(children)+1:05d}",
            "parent_id": "",
            "work_id": work_id,
            "author": author,
            "work_title": title,
            "section_id": "",
            "paragraph_id": paragraph_id,
            "text": child_text,
            "source_url": source_url,
            "lang": "en",
            "hash": sha256(child_text)
        })
    return children

def group_parents(children, work_id):
    parents=[]
    cur=[]
    for ch in children:
        cur.append(ch)
        toks=ntoks(" ".join(x["text"] for x in cur))
        if PARENT_MIN<=toks<=PARENT_MAX or toks>PARENT_MAX+120:
            parents.append(cur); cur=[]
    if cur: parents.append(cur)
    out_parents=[]
    for i, group in enumerate(parents,1):
        pid=f"{work_id}-p{i:04d}"
        for ch in group: ch["parent_id"]=pid
        parent_text="\n\n".join(x["text"] for x in group)
        out_parents.append({
            "id": pid,
            "work_id": work_id,
            "text": parent_text,
            "hash": sha256(parent_text)
        })
    return out_parents, children

def main():
    EXPORTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    summary=[]
    for mpath in sorted(MANIFESTS.glob("*.json")):
        m=json.loads(mpath.read_text(encoding="utf-8"))
        work_id=m["work_id"]; author=m["author"]; title=m["work_title"]
        html_path=NORM/f"{work_id}.html"
        if not html_path.exists():
            print(f"[SKIP] {work_id}: normalized HTML missing"); continue
        blocks=extract_blocks(html_path)
        children=group_children(blocks,work_id,author,title)
        parents,children=group_parents(children,work_id)
        with (EXPORTS/f"{work_id}_children.jsonl").open("w",encoding="utf-8") as fo:
            for r in children: fo.write(json.dumps(r,ensure_ascii=False)+"\n")
        with (EXPORTS/f"{work_id}_parents.jsonl").open("w",encoding="utf-8") as fo:
            for r in parents: fo.write(json.dumps(r,ensure_ascii=False)+"\n")
        summary.append({"work_id":work_id,"parents":len(parents),"children":len(children)})
        print(f"[OK] {work_id}: {len(parents)} parents, {len(children)} children")
    ts=time.strftime("%Y%m%d-%H%M%S")
    (LOGS/f"phase5_chunk_{ts}.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

if __name__=="__main__":
    main()
