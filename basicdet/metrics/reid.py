"""ReID retrieval metrics — mAP and CMC under the Market-1501 protocol.

Standard cross-camera retrieval evaluation (Zheng et al., "Scalable Person
Re-identification: A Benchmark", ICCV 2015): for each query, gallery entries
with the SAME identity AND SAME camera are excluded (they are near-duplicates
of the query, not re-identifications), then average precision and the
cumulative match curve are computed over the ranked gallery.

Model-agnostic: operates on precomputed feature matrices, so ft_net, CLIP-ReID,
or any future embedder evaluates through the same code.
"""

from __future__ import annotations

import logging
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Annotated as TypeAlias so mypy treats them as types even when the pre-commit
# hook's isolated env lacks numpy (npt.NDArray then degrades to Any, but the
# alias stays valid as a type).
FloatArray: TypeAlias = npt.NDArray[np.float32]
IntArray: TypeAlias = npt.NDArray[np.int64]


def cosine_distance(query: FloatArray, gallery: FloatArray) -> FloatArray:
    """Pairwise cosine distance between L2-normalised feature rows.

    Args:
        query: ``[n_query, dim]`` L2-normalised features.
        gallery: ``[n_gallery, dim]`` L2-normalised features.

    Returns:
        ``[n_query, n_gallery]`` distances in ``[0, 2]`` (``1 - cosine``).
    """
    return (1.0 - query @ gallery.T).astype(np.float32)


def evaluate_retrieval(
    query_feats: FloatArray,
    query_pids: IntArray,
    query_camids: IntArray,
    gallery_feats: FloatArray,
    gallery_pids: IntArray,
    gallery_camids: IntArray,
    ranks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute mAP and CMC rank-k under the Market-1501 protocol.

    Args:
        query_feats: ``[n_query, dim]`` L2-normalised query features.
        query_pids: Query identity labels.
        query_camids: Query camera ids.
        gallery_feats: ``[n_gallery, dim]`` L2-normalised gallery features.
        gallery_pids: Gallery identity labels.
        gallery_camids: Gallery camera ids.
        ranks: CMC ranks to report.

    Returns:
        ``{"mAP": ..., "rank1": ..., "rank5": ..., ...}`` — all in ``[0, 1]``.

    Raises:
        ValueError: If no query has a valid cross-camera match in the gallery.
    """
    dist = cosine_distance(query_feats, gallery_feats)
    n_query = dist.shape[0]
    max_rank = max(ranks)

    all_ap: list[float] = []
    all_cmc: list[np.ndarray] = []
    for i in range(n_query):
        order = np.argsort(dist[i])
        # Market protocol: drop same-identity/same-camera gallery entries.
        keep = ~(
            (gallery_pids[order] == query_pids[i]) & (gallery_camids[order] == query_camids[i])
        )
        matches = (gallery_pids[order][keep] == query_pids[i]).astype(np.int32)
        if not matches.any():
            continue  # identity absent from other cameras — excluded per protocol

        # CMC: 1 from the first correct match onward (0 everywhere if the first
        # hit falls outside max_rank).
        first_hit = int(np.argmax(matches))
        cmc = np.zeros(max_rank, dtype=np.float64)
        if first_hit < max_rank:
            cmc[first_hit:] = 1.0
        all_cmc.append(cmc)

        # Average precision over the ranked list.
        n_rel = int(matches.sum())
        hit_ranks = np.flatnonzero(matches) + 1  # 1-based ranks of correct matches
        precision_at_hits = np.arange(1, n_rel + 1) / hit_ranks
        all_ap.append(float(precision_at_hits.mean()))

    if not all_ap:
        raise ValueError("No query has a cross-camera match in the gallery — check the splits.")

    cmc_mean = np.stack(all_cmc).mean(axis=0)
    results = {"mAP": float(np.mean(all_ap))}
    for r in ranks:
        results[f"rank{r}"] = float(cmc_mean[r - 1])
    logger.info(
        "retrieval eval: %d/%d valid queries — %s",
        len(all_ap),
        n_query,
        ", ".join(f"{k}={v:.4f}" for k, v in results.items()),
    )
    return results
