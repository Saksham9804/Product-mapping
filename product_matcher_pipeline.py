from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from product_matcher.matcher import MatchConfig, build_outputs, rag_verify, retrieve_candidates
from product_matcher.schema import canonicalize
from product_matcher.scrapers import scrape_seller_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end cross-platform product matching pipeline.")
    parser.add_argument("--source-platform", required=True, help="Source platform, e.g. snapdeal.")
    parser.add_argument("--source-seller", required=True, help="Source seller username/name.")
    parser.add_argument("--target-platform", required=True, help="Target platform, e.g. flipkart.")
    parser.add_argument("--target-seller", required=True, help="Target seller username/name.")
    parser.add_argument("--source-input", type=Path, help="Optional source CSV. If omitted, scraper is used.")
    parser.add_argument("--target-input", type=Path, help="Optional target CSV. If omitted, scraper is used.")
    parser.add_argument("--source-seller-url", help="Optional source seller/catalog URL for scraping.")
    parser.add_argument("--target-seller-url", help="Optional target seller/catalog URL for scraping.")
    parser.add_argument("--output-dir", type=Path, default=Path("product_matcher_runs/latest"))
    parser.add_argument("--raw-dir", type=Path, default=Path("product_matcher_runs/raw"))
    parser.add_argument("--scrape-limit", type=int, default=100)
    parser.add_argument("--min-candidates", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--match-threshold", type=float, default=0.35)
    return parser.parse_args()


def load_or_scrape(platform: str, seller: str, input_csv: Path | None, seller_url: str | None, raw_dir: Path, limit: int) -> pd.DataFrame:
    if input_csv:
        return pd.read_csv(input_csv)
    raw_path = raw_dir / f"{platform.lower()}_{seller.lower().replace(' ', '_')}_catalog.csv"
    scrape_seller_catalog(platform=platform, seller=seller, output_csv=raw_path, seller_url=seller_url, limit=limit)
    return pd.read_csv(raw_path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    source_raw = load_or_scrape(
        args.source_platform,
        args.source_seller,
        args.source_input,
        args.source_seller_url,
        args.raw_dir,
        args.scrape_limit,
    )
    target_raw = load_or_scrape(
        args.target_platform,
        args.target_seller,
        args.target_input,
        args.target_seller_url,
        args.raw_dir,
        args.scrape_limit,
    )

    source = canonicalize(source_raw, args.source_platform.lower(), args.source_seller)
    target = canonicalize(target_raw, args.target_platform.lower(), args.target_seller)

    source.canonical.to_csv(args.output_dir / f"{source.platform}_canonical.csv", index=False)
    target.canonical.to_csv(args.output_dir / f"{target.platform}_canonical.csv", index=False)

    config = MatchConfig(
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        match_threshold=args.match_threshold,
    )
    candidates = retrieve_candidates(source, target, config)
    best, detail_scores = rag_verify(candidates, config)
    paths = build_outputs(source, target, best, detail_scores, args.output_dir)

    print("Pipeline complete.")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

