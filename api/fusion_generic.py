from typing import List, Dict, Tuple
import re, math, unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def strip_diacritics(s: str) -> str:
    if not s: return s
    nkfd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nkfd if not unicodedata.combining(ch))

def norm_text(s: str) -> str:
    s = s or ""
    s = strip_diacritics(s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tfidf_rerank(query: str, rows: List[Dict], top_k: int = 50) -> List[Tuple[int, float]]:
    """Generic TF-IDF reranker over candidate set (1-2 gram). Returns (idx, cosine) pairs."""
    texts = [norm_text(r.get("text","")) for r in rows]
    q = norm_text(query)
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(texts + [q])
    tv = X[:-1]    # docs
    qv = X[-1]     # query
    sims = linear_kernel(qv, tv).flatten()
    order = sims.argsort()[::-1][:min(top_k, len(texts))]
    return [(idx, float(sims[idx])) for idx in order]

def rrf_fuse(dense_scored: List[Tuple[int,float]], sparse_scored: List[Tuple[int,float]], k: float = 60.0) -> Dict[int, float]:
    """Reciprocal Rank Fusion over indices present in either list."""
    def to_ranks(arr: List[Tuple[int,float]]):
        order = sorted(arr, key=lambda x: x[1], reverse=True)
        return {idx: i+1 for i, (idx, _) in enumerate(order)}
    dr = to_ranks(dense_scored)
    sr = to_ranks(sparse_scored)
    all_idx = set(dr) | set(sr)
    fused = {}
    for i in all_idx:
        rd = dr.get(i, 10**9)
        rs = sr.get(i, 10**9)
        fused[i] = 1.0/(k+rd) + 1.0/(k+rs)
    return fused

def pick_with_fusion(rows: List[Dict], query: str, dense_key: str = "score",
                     take_dense: int = 100, final_k: int = 10) -> List[Dict]:
    """
    rows: [{'text', 'score', ...}] from dense search (higher is better).
    1) take top-N dense
    2) sparse (TF-IDF) rerank on that candidate set
    3) RRF fuse dense + sparse
    """
    if not rows: return []
    # 1) take top-N by dense
    ranked = sorted(list(enumerate(rows)), key=lambda t: t[1].get(dense_key,0.0), reverse=True)[:min(take_dense, len(rows))]
    idxs   = [i for i, _ in ranked]
    cand   = [rows[i] for i in idxs]

    # dense list as (original_idx, score)
    dense_pairs = [(i, r.get(dense_key, 0.0)) for i, r in zip(idxs, cand)]

    # 2) sparse rerank on cand (returns (cand_local_idx, score)); map back to original idx
    sparse_local = tfidf_rerank(query, cand, top_k=len(cand))
    sparse_pairs = [(idxs[i], s) for (i, s) in sparse_local]

    # 3) fuse
    fused = rrf_fuse(dense_pairs, sparse_pairs, k=60.0)
    best = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:min(final_k, len(cand))]
    return [rows[i] for (i, _) in best]
