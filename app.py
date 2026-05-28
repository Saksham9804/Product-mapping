from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from product_matcher.matcher import MatchConfig, build_outputs, rag_verify, retrieve_candidates
from product_matcher.schema import canonicalize
from product_matcher.scrapers import scrape_seller_catalog


st.set_page_config(
    page_title="Cross-Platform Product Matcher",
    layout="wide",
)


PLATFORMS = ["snapdeal", "flipkart", "amazon", "myntra", "meesho", "other"]


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    return pd.read_csv(uploaded_file)


def make_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def make_file_download(path: Path) -> bytes:
    return path.read_bytes()


def run_pipeline(
    source_raw: pd.DataFrame,
    target_raw: pd.DataFrame,
    source_platform: str,
    source_seller: str,
    target_platform: str,
    target_seller: str,
    output_dir: Path,
    min_candidates: int,
    max_candidates: int,
    match_threshold: float,
) -> dict:
    source = canonicalize(source_raw, source_platform, source_seller)
    target = canonicalize(target_raw, target_platform, target_seller)

    source.canonical.to_csv(output_dir / f"{source.platform}_canonical.csv", index=False)
    target.canonical.to_csv(output_dir / f"{target.platform}_canonical.csv", index=False)

    config = MatchConfig(
        min_candidates=min_candidates,
        max_candidates=max_candidates,
        match_threshold=match_threshold,
    )
    candidates = retrieve_candidates(source, target, config)
    best, detail_scores = rag_verify(candidates, config)
    paths = build_outputs(source, target, best, detail_scores, output_dir)
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    return {
        "source": source,
        "target": target,
        "candidates": candidates,
        "best": best,
        "detail_scores": detail_scores,
        "paths": paths,
        "summary": summary,
    }


def scrape_to_dataframe(platform: str, seller: str, seller_url: str, limit: int, output_dir: Path) -> pd.DataFrame:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{platform}_{seller.lower().replace(' ', '_')}_catalog.csv"
    scrape_seller_catalog(
        platform=platform,
        seller=seller,
        output_csv=csv_path,
        seller_url=seller_url or None,
        limit=limit,
    )
    return pd.read_csv(csv_path)


st.title("Cross-Platform Product Matcher")
st.caption("Scrape or upload two seller catalogs, clean and group products, retrieve vector candidates, then verify matches with RAG-style scoring.")

with st.sidebar:
    st.header("Input Mode")
    input_mode = st.radio(
        "Catalog source",
        ["Upload CSV files", "Best-effort scrape"],
        help="CSV upload is recommended because marketplace scraping can be blocked or inconsistent.",
    )

    st.header("Matching Settings")
    min_candidates = st.number_input("Minimum candidates", min_value=1, max_value=20, value=3)
    max_candidates = st.number_input("Maximum candidates", min_value=1, max_value=50, value=10)
    match_threshold = st.slider("Match threshold", min_value=0.05, max_value=0.95, value=0.35, step=0.01)
    scrape_limit = st.number_input("Scrape limit", min_value=1, max_value=1000, value=100)

if max_candidates < min_candidates:
    st.warning("Maximum candidates should be greater than or equal to minimum candidates.")

source_col, target_col = st.columns(2)

with source_col:
    st.subheader("Source Catalog")
    source_platform = st.selectbox("Source platform", PLATFORMS, index=0)
    source_seller = st.text_input("Source seller username/name", value="AUSK")
    source_seller_url = ""
    source_upload = None
    if input_mode == "Upload CSV files":
        source_upload = st.file_uploader("Upload source CSV", type=["csv"], key="source_csv")
    else:
        source_seller_url = st.text_input("Source seller/catalog URL", placeholder="Optional but recommended")

with target_col:
    st.subheader("Target Catalog")
    target_platform = st.selectbox("Target platform", PLATFORMS, index=1)
    target_seller = st.text_input("Target seller username/name", value="AUSK")
    target_seller_url = ""
    target_upload = None
    if input_mode == "Upload CSV files":
        target_upload = st.file_uploader("Upload target CSV", type=["csv"], key="target_csv")
    else:
        target_seller_url = st.text_input("Target seller/catalog URL", placeholder="Optional but recommended")

run_disabled = (
    not source_seller.strip()
    or not target_seller.strip()
    or max_candidates < min_candidates
    or (input_mode == "Upload CSV files" and (source_upload is None or target_upload is None))
)

run_clicked = st.button("Run Matching Pipeline", type="primary", disabled=run_disabled, use_container_width=True)

if run_clicked:
    with tempfile.TemporaryDirectory(prefix="product_matcher_streamlit_") as temp_dir:
        output_dir = Path(temp_dir) / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with st.status("Preparing catalogs...", expanded=True) as status:
                if input_mode == "Upload CSV files":
                    source_raw = read_uploaded_csv(source_upload)
                    target_raw = read_uploaded_csv(target_upload)
                    st.write(f"Loaded source rows: {len(source_raw):,}")
                    st.write(f"Loaded target rows: {len(target_raw):,}")
                else:
                    st.write("Scraping source catalog...")
                    source_raw = scrape_to_dataframe(source_platform, source_seller, source_seller_url, scrape_limit, output_dir)
                    st.write(f"Scraped source rows: {len(source_raw):,}")
                    st.write("Scraping target catalog...")
                    target_raw = scrape_to_dataframe(target_platform, target_seller, target_seller_url, scrape_limit, output_dir)
                    st.write(f"Scraped target rows: {len(target_raw):,}")

                st.write("Cleaning, grouping, vectorizing, and verifying matches...")
                result = run_pipeline(
                    source_raw=source_raw,
                    target_raw=target_raw,
                    source_platform=source_platform,
                    source_seller=source_seller,
                    target_platform=target_platform,
                    target_seller=target_seller,
                    output_dir=output_dir,
                    min_candidates=int(min_candidates),
                    max_candidates=int(max_candidates),
                    match_threshold=float(match_threshold),
                )
                status.update(label="Pipeline complete", state="complete", expanded=False)

            st.session_state["latest_result"] = {
                "summary": result["summary"],
                "mapped": pd.read_csv(result["paths"]["mapped"]),
                "unmatched": pd.read_csv(result["paths"]["unmatched"]),
                "detail_scores": pd.read_csv(result["paths"]["detail_scores"]),
                "summary_md": result["paths"]["summary_md"].read_text(encoding="utf-8"),
                "summary_json": result["paths"]["summary_json"].read_text(encoding="utf-8"),
                "source_canonical": result["source"].canonical,
                "target_canonical": result["target"].canonical,
            }
        except Exception as exc:  # noqa: BLE001 - Streamlit should show actionable error text.
            st.error(f"Pipeline failed: {exc}")

latest = st.session_state.get("latest_result")

if latest:
    summary = latest["summary"]
    st.divider()
    st.subheader("Run Summary")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Source Products", f"{summary['source_products']:,}")
    metric_cols[1].metric("Target Products", f"{summary['target_products']:,}")
    metric_cols[2].metric("Candidate Pairs", f"{summary['candidate_pairs_scored']:,}")
    metric_cols[3].metric("Mapped", f"{summary['matched_products']:,}")
    metric_cols[4].metric("Unmatched", f"{summary['unmatched_products']:,}")

    tab_mapped, tab_unmatched, tab_scores, tab_summary, tab_clean = st.tabs(
        ["Mapped Products", "Unmatched Products", "RAG Scores", "Summary", "Canonical Data"]
    )

    with tab_mapped:
        st.dataframe(latest["mapped"], use_container_width=True, height=420)
        st.download_button(
            "Download mapped_products.csv",
            data=make_csv_download(latest["mapped"]),
            file_name="mapped_products.csv",
            mime="text/csv",
        )

    with tab_unmatched:
        st.dataframe(latest["unmatched"], use_container_width=True, height=420)
        st.download_button(
            "Download unmatched_products.csv",
            data=make_csv_download(latest["unmatched"]),
            file_name="unmatched_products.csv",
            mime="text/csv",
        )

    with tab_scores:
        st.dataframe(latest["detail_scores"], use_container_width=True, height=420)
        st.download_button(
            "Download rag_candidate_scores.csv",
            data=make_csv_download(latest["detail_scores"]),
            file_name="rag_candidate_scores.csv",
            mime="text/csv",
        )

    with tab_summary:
        st.markdown(latest["summary_md"])
        st.download_button(
            "Download summary.md",
            data=latest["summary_md"].encode("utf-8"),
            file_name="summary.md",
            mime="text/markdown",
        )
        st.download_button(
            "Download summary.json",
            data=latest["summary_json"].encode("utf-8"),
            file_name="summary.json",
            mime="application/json",
        )

    with tab_clean:
        clean_left, clean_right = st.columns(2)
        with clean_left:
            st.write("Source canonical data")
            st.dataframe(latest["source_canonical"], use_container_width=True, height=360)
            st.download_button(
                "Download source_canonical.csv",
                data=make_csv_download(latest["source_canonical"]),
                file_name="source_canonical.csv",
                mime="text/csv",
            )
        with clean_right:
            st.write("Target canonical data")
            st.dataframe(latest["target_canonical"], use_container_width=True, height=360)
            st.download_button(
                "Download target_canonical.csv",
                data=make_csv_download(latest["target_canonical"]),
                file_name="target_canonical.csv",
                mime="text/csv",
            )
else:
    st.info("Upload two catalog CSVs, or choose best-effort scrape mode, then run the matcher.")
