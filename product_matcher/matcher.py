from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import PlatformDataset
from .text_utils import clean_text, color_score, jaccard, price_score, token_set


CandidateColumns = [
    "source_platform",
    "target_platform",
    "source_product_id",
    "target_product_id",
    "source_title",
    "target_title",
    "source_color",
    "target_color",
    "source_price",
    "target_price",
    "source_url",
    "target_url",
    "source_image_urls",
    "target_image_urls",
    "group_key",
    "target_group_key",
    "blocking_key",
    "candidate_rank",
    "retrieval_score",
    "text_score",
    "color_score",
    "price_score",
    "attribute_overlap",
]

DetailScoreColumns = CandidateColumns + [
    "title_overlap",
    "visual_proxy_score",
    "hybrid_rag_score",
    "graph_rag_score",
    "multimodal_rag_score",
    "corrective_rag_score",
    "verification_score",
    "stage2_rank",
]


@dataclass(frozen=True)
class MatchConfig:
    min_candidates: int = 3
    max_candidates: int = 10
    match_threshold: float = 0.35


def retrieve_candidates(source: PlatformDataset, target: PlatformDataset, config: MatchConfig) -> pd.DataFrame:
    source_df = source.canonical.reset_index(drop=True)
    target_df = target.canonical.reset_index(drop=True)
    rows = []
    active_groups = sorted(set(source_df["blocking_key"]).intersection(set(target_df["blocking_key"])))

    for blocking_key in active_groups:
        left = source_df[source_df["blocking_key"].eq(blocking_key)].reset_index(drop=True)
        right = target_df[target_df["blocking_key"].eq(blocking_key)].reset_index(drop=True)
        if left.empty or right.empty:
            continue

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
        matrix = vectorizer.fit_transform(pd.concat([left["match_text"], right["match_text"]], ignore_index=True))
        similarities = cosine_similarity(matrix[: len(left)], matrix[len(left) :])

        for source_idx, source_row in left.iterrows():
            scored = []
            for target_idx, target_row in right.iterrows():
                text_score = float(similarities[source_idx, target_idx])
                c_score = color_score(source_row["color"], target_row["color"])
                p_score = price_score(source_row["price"], target_row["price"])
                attribute_overlap = jaccard(
                    token_set(source_row["pattern_norm"], source_row["fabric"], source_row["type_norm"], source_row["neck_norm"]),
                    token_set(target_row["pattern_norm"], target_row["fabric"], target_row["type_norm"], target_row["neck_norm"]),
                )
                retrieval_score = (0.58 * text_score) + (0.18 * c_score) + (0.14 * p_score) + (0.10 * attribute_overlap)
                scored.append((retrieval_score, text_score, c_score, p_score, attribute_overlap, target_idx, target_row))

            shortlist = min(config.max_candidates, max(config.min_candidates, len(scored)))
            for rank, item in enumerate(sorted(scored, key=lambda x: x[0], reverse=True)[:shortlist], start=1):
                retrieval_score, text_score, c_score, p_score, attribute_overlap, _, target_row = item
                rows.append(
                    {
                        "source_platform": source.platform,
                        "target_platform": target.platform,
                        "source_product_id": source_row["product_id"],
                        "target_product_id": target_row["product_id"],
                        "source_title": source_row["title"],
                        "target_title": target_row["title"],
                        "source_color": source_row["color"],
                        "target_color": target_row["color"],
                        "source_price": source_row["price"],
                        "target_price": target_row["price"],
                        "source_url": source_row["url"],
                        "target_url": target_row["url"],
                        "source_image_urls": source_row["image_urls"],
                        "target_image_urls": target_row["image_urls"],
                        "group_key": source_row["group_key"],
                        "target_group_key": target_row["group_key"],
                        "blocking_key": blocking_key,
                        "candidate_rank": rank,
                        "retrieval_score": round(retrieval_score, 6),
                        "text_score": round(text_score, 6),
                        "color_score": round(c_score, 6),
                        "price_score": round(p_score, 6),
                        "attribute_overlap": round(attribute_overlap, 6),
                    }
                )
    return pd.DataFrame(rows, columns=CandidateColumns)


def rag_verify(candidates: pd.DataFrame, config: MatchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(columns=DetailScoreColumns)

    df = candidates.copy()
    df["title_overlap"] = df.apply(lambda r: jaccard(token_set(r["source_title"]), token_set(r["target_title"])), axis=1)
    df["visual_proxy_score"] = df.apply(
        lambda r: jaccard(token_set(r["source_image_urls"]), token_set(r["target_image_urls"])),
        axis=1,
    )
    df["hybrid_rag_score"] = (
        0.42 * df["retrieval_score"]
        + 0.22 * df["text_score"]
        + 0.16 * df["color_score"]
        + 0.12 * df["price_score"]
        + 0.08 * df["attribute_overlap"]
    )
    df["graph_rag_score"] = (
        0.30
        + 0.24 * df["title_overlap"]
        + 0.20 * df["color_score"]
        + 0.16 * df["price_score"]
        + 0.10 * df["attribute_overlap"]
    )
    df["multimodal_rag_score"] = (
        0.48 * df["title_overlap"]
        + 0.24 * df["color_score"]
        + 0.16 * df["price_score"]
        + 0.12 * df["visual_proxy_score"]
    )
    df["corrective_rag_score"] = df["hybrid_rag_score"]
    df.loc[df["color_score"].eq(0), "corrective_rag_score"] -= 0.12
    df.loc[df["text_score"].lt(0.05), "corrective_rag_score"] -= 0.08
    df["corrective_rag_score"] = df["corrective_rag_score"].clip(lower=0)
    df["verification_score"] = (
        0.30 * df["hybrid_rag_score"]
        + 0.24 * df["graph_rag_score"]
        + 0.22 * df["multimodal_rag_score"]
        + 0.24 * df["corrective_rag_score"]
    )

    df = df.sort_values(["source_product_id", "verification_score"], ascending=[True, False])
    df["stage2_rank"] = df.groupby("source_product_id").cumcount() + 1
    best = df[df["stage2_rank"].eq(1)].copy()
    second = (
        df[df["stage2_rank"].eq(2)][["source_product_id", "verification_score"]]
        .rename(columns={"verification_score": "second_best_score"})
    )
    best = best.merge(second, on="source_product_id", how="left")
    best["score_margin"] = best["verification_score"] - best["second_best_score"].fillna(0)
    best["status"] = np.where(best["verification_score"].ge(config.match_threshold), "matched", "unmatched")
    best["confidence"] = np.where(
        best["status"].eq("matched"),
        (0.55 + best["verification_score"].clip(0, 1) * 0.44 + best["score_margin"].clip(0, 0.20) * 0.05).clip(0.55, 0.99),
        np.nan,
    )
    best["reasoning"] = best.apply(reasoning, axis=1)
    return best, df


def reasoning(row: pd.Series) -> str:
    if row["status"] != "matched":
        return "Best candidate did not pass the verification threshold."
    signals = []
    if row["color_score"] >= 1:
        signals.append("color match")
    if row["price_score"] >= 0.75:
        signals.append("price close")
    if row["text_score"] >= 0.25:
        signals.append("strong text similarity")
    if row["attribute_overlap"] >= 0.25:
        signals.append("attribute overlap")
    return "; ".join(signals) or "same group with highest retrieval and verification score"


def build_outputs(
    source: PlatformDataset,
    target: PlatformDataset,
    best: pd.DataFrame,
    detail_scores: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    matched = best[best["status"].eq("matched")].copy() if not best.empty else pd.DataFrame()
    unmatched = best[best["status"].ne("matched")].copy() if not best.empty else pd.DataFrame()

    source_ids_with_candidates = set(best["source_product_id"]) if not best.empty else set()
    source_without_candidates = source.canonical[~source.canonical["product_id"].isin(source_ids_with_candidates)].copy()
    if not source_without_candidates.empty:
        no_candidate = pd.DataFrame(
            {
                "source_platform": source.platform,
                "target_platform": target.platform,
                "source_product_id": source_without_candidates["product_id"],
                "source_title": source_without_candidates["title"],
                "source_color": source_without_candidates["color"],
                "source_price": source_without_candidates["price"],
                "source_url": source_without_candidates["url"],
                "group_key": source_without_candidates["group_key"],
                "status": "unmatched",
                "reasoning": "No candidate found in the target platform for this group.",
            }
        )
        unmatched = pd.concat([unmatched, no_candidate], ignore_index=True)

    mapped_columns = [
        "source_platform",
        "target_platform",
        "source_product_id",
        "target_product_id",
        "source_title",
        "target_title",
        "source_color",
        "target_color",
        "source_price",
        "target_price",
        "source_url",
        "target_url",
        "group_key",
        "confidence",
        "verification_score",
        "retrieval_score",
        "text_score",
        "color_score",
        "price_score",
        "attribute_overlap",
        "reasoning",
    ]
    unmatched_columns = [
        "source_platform",
        "target_platform",
        "source_product_id",
        "source_title",
        "source_color",
        "source_price",
        "source_url",
        "group_key",
        "status",
        "reasoning",
    ]

    mapped_path = output_dir / "mapped_products.csv"
    unmatched_path = output_dir / "unmatched_products.csv"
    detail_path = output_dir / "rag_candidate_scores.csv"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    matched.reindex(columns=mapped_columns).to_csv(mapped_path, index=False)
    unmatched.reindex(columns=unmatched_columns).to_csv(unmatched_path, index=False)
    detail_scores.to_csv(detail_path, index=False)

    summary = {
        "source_platform": source.platform,
        "source_seller": source.seller,
        "target_platform": target.platform,
        "target_seller": target.seller,
        "source_products": int(len(source.canonical)),
        "target_products": int(len(target.canonical)),
        "matched_products": int(len(matched)),
        "unmatched_products": int(len(unmatched)),
        "candidate_pairs_scored": int(len(detail_scores)),
        "match_rate": round(float(len(matched) / max(len(source.canonical), 1)), 4),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(
        "\n".join(
            [
                "# Product Matching Summary",
                "",
                f"- Source: {source.platform} / {source.seller}",
                f"- Target: {target.platform} / {target.seller}",
                f"- Source products: {summary['source_products']}",
                f"- Target products: {summary['target_products']}",
                f"- Candidate pairs scored: {summary['candidate_pairs_scored']}",
                f"- Matched products: {summary['matched_products']}",
                f"- Unmatched products: {summary['unmatched_products']}",
                f"- Match rate: {summary['match_rate']:.2%}",
                "",
                "Outputs:",
                f"- mapped_products.csv",
                f"- unmatched_products.csv",
                f"- rag_candidate_scores.csv",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "mapped": mapped_path,
        "unmatched": unmatched_path,
        "detail_scores": detail_path,
        "summary_json": summary_json_path,
        "summary_md": summary_md_path,
    }
