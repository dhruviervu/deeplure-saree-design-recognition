"""Checks for grouping, Rank-1 / mAP, and k-reciprocal re-ranking."""

import numpy as np

from deeplure_core import (
    cosine_query_scores,
    k_reciprocal_distance,
    query_gallery_split,
    rank1_and_map,
    reranked_query_scores,
    source_group,
)


def test_source_group():
    from pathlib import Path

    assert source_group(Path("a_jpg.rf.abc.jpg")) == "a_jpg"
    assert source_group(Path("a_jpg.rf.def.jpg")) == "a_jpg"
    assert source_group(Path("plain.jpg")) == "plain"


def test_split_and_perfect_retrieval():
    from pathlib import Path

    records = []
    for group in range(12):
        for copy in range(3):
            records.append({"path": Path(f"g{group:02d}_c{copy}.jpg"), "group": f"g{group:02d}"})
    queries, gallery = query_gallery_split(records)
    assert len(gallery) == 12
    assert len(queries) == 24
    assert set(queries).isdisjoint(gallery)

    dim = 32
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(12, dim))
    emb = np.zeros((len(records), dim), dtype=np.float64)
    for i, rec in enumerate(records):
        gid = int(rec["group"][1:])
        emb[i] = centers[gid] + 0.01 * rng.normal(size=dim)

    q_groups = [records[i]["group"] for i in queries]
    g_groups = [records[i]["group"] for i in gallery]
    scores = cosine_query_scores(emb, queries, gallery)
    metrics = rank1_and_map(scores, q_groups, g_groups)
    assert metrics["rank1"] == 1.0
    assert metrics["mAP"] == 1.0
    assert metrics["n_queries"] == 24

    reranked = reranked_query_scores(emb, queries, gallery, k1=4, k2=2, lam=0.3)
    rerank_metrics = rank1_and_map(reranked, q_groups, g_groups)
    assert rerank_metrics["rank1"] == 1.0


def test_single_positive_average_precision():
    # Ranked order is gallery 1 (b), 0 (a), 2 (c), 3 (d). The positive is rank 2, so AP is 1/2.
    scores = np.array([[0.2, 0.9, 0.1, 0.0]])
    metrics = rank1_and_map(scores, ["a"], ["a", "b", "c", "d"])
    assert metrics["rank1"] == 0.0
    assert abs(metrics["mAP"] - 0.5) < 1e-9


def test_rerank_from_similarity_matches_embeddings():
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(18, 8))
    queries = list(range(0, 12))
    gallery = list(range(12, 18))
    from deeplure_core import reranked_from_similarity

    features = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    via_emb = reranked_query_scores(emb, queries, gallery, k1=4, k2=2, lam=0.3)
    via_sim = reranked_from_similarity(features @ features.T, queries, gallery, k1=4, k2=2, lam=0.3)
    assert np.allclose(via_emb, via_sim)


def test_notebook_embeds_the_helper_module():
    import json
    from pathlib import Path

    notebook = json.loads(Path("saree_design_recognition.ipynb").read_text(encoding="utf-8"))
    helper = Path("deeplure_core.py").read_text(encoding="utf-8").strip()
    sources = ["".join(cell["source"]).strip() for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert helper in sources


def test_rerank_diagonal_and_shape():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(25, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    dist = k_reciprocal_distance(emb @ emb.T, k1=5, k2=3, lam=0.3)
    assert dist.shape == (25, 25)
    assert np.allclose(np.diag(dist), 0.0)
    assert np.isfinite(dist).all()


if __name__ == "__main__":
    test_source_group()
    test_split_and_perfect_retrieval()
    test_single_positive_average_precision()
    test_rerank_diagonal_and_shape()
    test_rerank_from_similarity_matches_embeddings()
    test_notebook_embeds_the_helper_module()
    print("all retrieval checks passed")
