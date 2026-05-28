from __future__ import annotations

import csv
import datetime as dt
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def fetch(url: str, timeout: int = 25) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="ignore")


def clean_html(value: Any) -> str:
    if value is None:
        return ""
    value = html.unescape(str(value))
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_json_ld(page: str) -> list[Any]:
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    parsed = []
    for block in blocks:
        try:
            parsed.append(json.loads(html.unescape(block).strip()))
        except json.JSONDecodeError:
            continue
    return parsed


def walk_json_ld(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        rows.append(value)
        if isinstance(value.get("@graph"), list):
            for child in value["@graph"]:
                rows.extend(walk_json_ld(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(walk_json_ld(child))
    return rows


def product_from_page(url: str, page: str, platform: str, seller: str) -> dict[str, str]:
    product = {}
    for item in walk_json_ld(extract_json_ld(page)):
        item_type = item.get("@type", "")
        types = {str(x).lower() for x in item_type} if isinstance(item_type, list) else {str(item_type).lower()}
        if "product" in types:
            product = item
            break

    offers = product.get("offers", {}) if isinstance(product.get("offers"), dict) else {}
    images = product.get("image", [])
    if isinstance(images, str):
        images = [images]
    if not isinstance(images, list):
        images = []

    title = clean_html(product.get("name")) or clean_html(re.search(r"<title[^>]*>(.*?)</title>", page, re.I | re.S).group(1) if re.search(r"<title[^>]*>(.*?)</title>", page, re.I | re.S) else "")
    return {
        "platform": platform,
        "seller": seller,
        "product_id": product.get("sku") or product.get("productID") or url.rstrip("/").split("/")[-1],
        "title": title,
        "brand": clean_html(product.get("brand", {}).get("name") if isinstance(product.get("brand"), dict) else product.get("brand")),
        "description": clean_html(product.get("description")),
        "price": clean_html(offers.get("price")),
        "url": url,
        "image_urls": " | ".join(str(x) for x in images),
        "scraped_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    }


def seller_url_from_username(platform: str, seller: str) -> str:
    encoded = urllib.parse.quote_plus(seller)
    platform = platform.lower()
    if platform == "flipkart":
        return f"https://www.flipkart.com/search?q={encoded}&marketplace=FLIPKART"
    if platform == "snapdeal":
        return f"https://www.snapdeal.com/search?keyword={encoded}"
    if platform == "amazon":
        return f"https://www.amazon.in/s?k={encoded}"
    return f"https://www.google.com/search?q={urllib.parse.quote_plus(platform + ' ' + seller)}"


def extract_product_links(platform: str, base_url: str, page: str) -> list[str]:
    links = []
    for href in re.findall(r'href=["\']([^"\']+)["\']', page, flags=re.IGNORECASE):
        absolute = urllib.parse.urljoin(base_url, html.unescape(href))
        platform_l = platform.lower()
        keep = (
            (platform_l == "snapdeal" and "/product/" in absolute)
            or (platform_l == "flipkart" and "/p/" in absolute)
            or (platform_l == "amazon" and "/dp/" in absolute)
        )
        if keep and absolute not in links:
            links.append(absolute.split("?")[0])
    return links


def scrape_seller_catalog(
    platform: str,
    seller: str,
    output_csv: Path,
    seller_url: str | None = None,
    limit: int = 100,
    sleep: float = 1.0,
) -> Path:
    """Best-effort lightweight scraper.

    Most marketplaces change markup and may block automated requests. For reliable
    production scraping, replace this adapter with Playwright/proxy/API code while
    keeping the downstream matcher unchanged.
    """
    seller_url = seller_url or seller_url_from_username(platform, seller)
    listing_page = fetch(seller_url)
    links = extract_product_links(platform, seller_url, listing_page)[:limit]

    rows = []
    for index, link in enumerate(links, start=1):
        try:
            rows.append(product_from_page(link, fetch(link), platform, seller))
        except Exception as exc:  # noqa: BLE001 - keep failed scrape evidence in CSV.
            rows.append({"platform": platform, "seller": seller, "product_id": link, "url": link, "scrape_error": str(exc)})
        if index < len(links):
            time.sleep(sleep)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) or ["platform", "seller", "product_id", "url"]
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_csv

