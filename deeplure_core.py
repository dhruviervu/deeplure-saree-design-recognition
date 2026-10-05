"""Zero-shot color-invariant saree retrieval.

DINOv2 ViT-B/14 is a frozen feature extractor. Weave-folder names are never
used as training labels. This module holds the pieces that do not need a GPU
so they can be tested on their own: grouping, metrics, and k-reciprocal
re-ranking.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def source_group(path: Path) -> str:
    """Identity of the photo before Roboflow's .rf.<hash> copy suffix.

    `stem.rf.abc123.jpg` and `stem.rf.def456.jpg` are copies of one source
    image, so they share a group. Files without `.rf.` are their own group.
    """
    name = path.name
    marker = ".rf."
    if marker in name:
        return name.split(marker)[0]
    return path.stem


def discover_roboflow(root: Path) -> list[dict]:
    """Index train/valid/test class folders.

    The group id is the weave folder plus the source stem. Short names such as
    `image19.jpeg` were reused inside every class folder, so the stem alone
    would glue unrelated photographs together. The weave folder is only a
    namespace that keeps those files apart. It is not a class the model learns.
    """
    root = Path(root)
    records = []
    if not root.exists():
        return records
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
                    continue
                stem = source_group(path)
                records.append(
                    {
                        "path": path.resolve(),
                        "group": f"{class_dir.name}/{stem}",
                        "weave": class_dir.name,
                        "split": split_dir.name,
                    }
                )
    return records


def discover_flat(root: Path) -> list[dict]:
    """Index an unlabeled folder. Each file is its own group."""
    root = Path(root)
    if not root.exists():
        return []
    records = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        records.append(
            {
                "path": path.resolve(),
                "group": path.stem,
                "weave": "",
                "split": "unlabeled",
            }
        )
    return records


def audit_groups(records: list[dict]) -> dict:
    """Summarize whether .rf. copies stayed in one folder and one weave name."""
    by_group = defaultdict(list)
    for rec in records:
        by_group[rec["group"]].append(rec)
    sizes = [len(v) for v in by_group.values()]
    split_across_folders = 0
    mixed_weave = 0
    for members in by_group.values():
        if len({m["split"] for m in members}) > 1:
            split_across_folders += 1
        weaves = {m["weave"] for m in members if m["weave"]}
        if len(weaves) > 1:
            mixed_weave += 1
    usable = sum(1 for n in sizes if n >= 2)
    return {
        "images": len(records),
        "groups": len(by_group),
        "groups_with_a_positive": usable,
        "max_copies": max(sizes) if sizes else 0,
        "groups_split_across_folders": split_across_folders,
        "groups_with_mixed_weave_names": mixed_weave,
    }


def query_gallery_split(records: list[dict]) -> tuple[list[int], list[int]]:
    """One gallery reference per source photo, remaining copies are queries.

    The first filename in a group is the reference. Groups with a single file
    have no positive and are left out. Query and gallery indices are disjoint.
    """
    by_group = defaultdict(list)
    for index, rec in enumerate(records):
        by_group[rec["group"]].append(index)

    queries: list[int] = []
    gallery: list[int] = []
    for group in sorted(by_group):
        members = sorted(by_group[group], key=lambda i: records[i]["path"].name)
        if len(members) < 2:
            continue
        gallery.append(members[0])
        queries.extend(members[1:])
    return queries, gallery


def rank1_and_map(scores: np.ndarray, query_groups: list[str], gallery_groups: list[str]) -> dict:
    """Rank-1 and mAP from a query-by-gallery score matrix (higher is closer).

    A gallery item is relevant when its group equals the query group. Average
    precision uses the precision at each relevant rank.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be a query-by-gallery matrix")
    n_query, n_gallery = scores.shape
    if n_query != len(query_groups) or n_gallery != len(gallery_groups):
        raise ValueError("group lists must match the score matrix shape")

    order = np.argsort(-scores, axis=1, kind="mergesort")
    gallery_groups = list(gallery_groups)
    hits = []
    average_precisions = []
    reciprocal_ranks = []

    for row, ranked in enumerate(order):
        target = query_groups[row]
        n_rel = sum(group == target for group in gallery_groups)
        if n_rel == 0:
            continue
        hits.append(gallery_groups[int(ranked[0])] == target)
        found = 0
        precision_sum = 0.0
        first_rank = None
        for rank, col in enumerate(ranked, start=1):
            if gallery_groups[int(col)] != target:
                continue
            found += 1
            precision_sum += found / rank
            if first_rank is None:
                first_rank = rank
            if found == n_rel:
                break
        average_precisions.append(precision_sum / n_rel)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

    n = len(average_precisions)
    return {
        "rank1": float(np.mean(hits)) if n else float("nan"),
        "mAP": float(np.mean(average_precisions)) if n else float("nan"),
        "mrr": float(np.mean(reciprocal_ranks)) if n else float("nan"),
        "n_queries": n,
        "n_gallery": n_gallery,
    }


def k_reciprocal_distance(similarity: np.ndarray, k1: int = 20, k2: int = 6, lam: float = 0.3) -> np.ndarray:
    """Pairwise distance after k-reciprocal re-ranking.

    Implements Zhong, Zheng, Cao, and Li, "Re-ranking Person Re-identification
    with k-reciprocal Encoding", CVPR 2017. `similarity` is cosine similarity.
    The returned matrix is a distance: smaller means more similar. Diagonal is 0.

    Final distance = (1 - lam) * cosine_distance + lam * jaccard_distance.
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    n = similarity.shape[0]
    if n == 0:
        return similarity.copy()

    dist = np.clip(1.0 - similarity, 0.0, None)
    np.fill_diagonal(dist, 0.0)
    # Rank neighbors from the raw cosine distance. A copy is kept because the
    # expansion step must not see the later blended distance.
    initial_rank = np.argsort(dist, axis=1, kind="mergesort")
    k1 = int(max(1, min(k1, n - 1)))
    k2 = int(max(1, min(k2, n)))
    half = max(1, int(round(k1 / 2.0)))

    encoding = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        forward = initial_rank[i, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal_slots = np.where(backward == i)[0]
        reciprocal = forward[reciprocal_slots]

        expanded = reciprocal
        for candidate in reciprocal:
            cand_forward = initial_rank[candidate, : half + 1]
            cand_backward = initial_rank[cand_forward, : half + 1]
            cand_slots = np.where(cand_backward == candidate)[0]
            cand_reciprocal = cand_forward[cand_slots]
            overlap = np.intersect1d(cand_reciprocal, reciprocal, assume_unique=False)
            if cand_reciprocal.size > 0 and overlap.size > (2.0 / 3.0) * cand_reciprocal.size:
                expanded = np.append(expanded, cand_reciprocal)
        expanded = np.unique(expanded)
        weights = np.exp(-dist[i, expanded])
        encoding[i, expanded] = weights / np.sum(weights)

    if k2 > 1:
        smoothed = np.empty_like(encoding)
        for i in range(n):
            smoothed[i] = encoding[initial_rank[i, :k2]].mean(axis=0)
        encoding = smoothed

    # Inverted lists keep the Jaccard loop on the non-zero support only.
    inverted = [np.flatnonzero(encoding[:, col]) for col in range(n)]
    jaccard = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        support = np.flatnonzero(encoding[i])
        if support.size == 0:
            jaccard[i] = 1.0
            jaccard[i, i] = 0.0
            continue
        overlap_mass = np.zeros(n, dtype=np.float64)
        for col in support:
            rows = inverted[col]
            overlap_mass[rows] += np.minimum(encoding[i, col], encoding[rows, col])
        # Both rows are L1-normalized, so sum(max) = 2 - sum(min).
        jaccard[i] = 1.0 - overlap_mass / np.clip(2.0 - overlap_mass, 1e-12, None)
        jaccard[i, i] = 0.0

    blended = (1.0 - lam) * dist + lam * jaccard
    np.fill_diagonal(blended, 0.0)
    return blended


def reranked_query_scores(
    embeddings: np.ndarray,
    query_index: list[int],
    gallery_index: list[int],
    k1: int = 20,
    k2: int = 6,
    lam: float = 0.3,
) -> np.ndarray:
    """Cosine k-reciprocal scores for queries against a disjoint gallery.

    Re-ranking is computed on the union of query and gallery images, which is
    the closed-set setting used at evaluation. Higher scores are closer.
    """
    if set(query_index) & set(gallery_index):
        raise ValueError("query and gallery indices overlap")
    union = list(gallery_index) + list(query_index)
    position = {image_index: row for row, image_index in enumerate(union)}
    features = np.asarray(embeddings, dtype=np.float64)[union]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, 1e-12, None)
    similarity = features @ features.T
    distance = k_reciprocal_distance(similarity, k1=k1, k2=k2, lam=lam)
    rows = [position[i] for i in query_index]
    cols = [position[i] for i in gallery_index]
    return -distance[np.ix_(rows, cols)]


def reranked_from_similarity(
    similarity: np.ndarray,
    query_index: list[int],
    gallery_index: list[int],
    k1: int = 20,
    k2: int = 6,
    lam: float = 0.3,
) -> np.ndarray:
    """Re-rank a full pairwise cosine matrix and return query-gallery scores.

    `similarity[i, j]` is the cosine of dataset row i with dataset row j.
    Higher returned scores are closer.
    """
    if set(query_index) & set(gallery_index):
        raise ValueError("query and gallery indices overlap")
    union = list(gallery_index) + list(query_index)
    position = {image_index: row for row, image_index in enumerate(union)}
    block = np.asarray(similarity, dtype=np.float64)[np.ix_(union, union)]
    distance = k_reciprocal_distance(block, k1=k1, k2=k2, lam=lam)
    rows = [position[i] for i in query_index]
    cols = [position[i] for i in gallery_index]
    return -distance[np.ix_(rows, cols)]


def cosine_query_scores(embeddings: np.ndarray, query_index: list[int], gallery_index: list[int]) -> np.ndarray:
    features = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.clip(norms, 1e-12, None)
    return features[query_index] @ features[gallery_index].T


def topk_indices(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return gallery indices and scores of the top-k matches per query."""
    k = int(min(k, scores.shape[1]))
    if k <= 0:
        empty = np.zeros((scores.shape[0], 0), dtype=np.int64)
        return empty, empty.astype(np.float64)
    chosen = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, chosen], axis=1, kind="mergesort")
    cols = chosen[row, order]
    return cols, scores[row, cols]
