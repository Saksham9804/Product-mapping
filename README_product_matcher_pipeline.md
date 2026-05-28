# Cross-Platform Product Matcher Pipeline

This turns the notebook workflow into a reusable end-to-end pipeline:

1. Load or scrape seller catalog data for two platforms.
2. Canonicalize platform-specific columns into a common product schema.
3. Clean attributes and build deterministic group keys.
4. Vectorize product evidence with TF-IDF inside each group.
5. Retrieve candidate pairs.
6. Verify candidates with local RAG-style scorers: hybrid, graph, multimodal proxy, and corrective scoring.
7. Export a detailed summary, mapped products, unmatched products, and all candidate scores.

## Run With Your Existing Data

From `C:\Users\jaisw\Desktop\Projects\Codes`:

```powershell
python product_matcher_pipeline.py `
  --source-platform snapdeal `
  --source-seller AUSK `
  --source-input "C:\Users\jaisw\Desktop\Projects\Data_Extraction\Snapdeal123_enriched.csv" `
  --target-platform flipkart `
  --target-seller AUSK `
  --target-input "C:\Users\jaisw\Desktop\Projects\Data_Extraction\Flipkart123_enriched.csv" `
  --output-dir "C:\Users\jaisw\Desktop\Projects\Data_Extraction\pipeline_run"
```

## Run With Scraping

If CSV inputs are omitted, the pipeline runs a best-effort lightweight scraper:

```powershell
python product_matcher_pipeline.py `
  --source-platform snapdeal `
  --source-seller AUSK `
  --source-seller-url "https://www.snapdeal.com/search?keyword=AUSK" `
  --target-platform flipkart `
  --target-seller AUSK `
  --target-seller-url "https://www.flipkart.com/search?q=AUSK" `
  --output-dir "C:\Users\jaisw\Desktop\Projects\Data_Extraction\pipeline_run_scraped"
```

Marketplace pages often block basic HTTP scraping or change markup. For production use, replace only `product_matcher/scrapers.py` with Playwright/API/platform-specific scraping while keeping the cleaning, grouping, vectorization, and mapping stages unchanged.

## Outputs

- `summary.md`: human-readable run summary.
- `summary.json`: machine-readable run summary.
- `mapped_products.csv`: one best target-platform match per matched source product.
- `unmatched_products.csv`: source products without a confident target match.
- `rag_candidate_scores.csv`: all retrieved candidates and verifier scores for audit/debugging.
- `<platform>_canonical.csv`: cleaned intermediate tables.

