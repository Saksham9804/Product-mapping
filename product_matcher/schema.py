from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .text_utils import (
    clean_text,
    color_family,
    first_present,
    normalize_category,
    normalize_fit,
    normalize_gender,
    normalize_neck,
    normalize_pattern,
    normalize_type,
    parse_price,
)


SELLER_ALIASES = ["seller_code", "seller_name", "seller_data_name", "seller", "username"]

ALIASES = {
    "product_id": ["product_id", "product_sku_id", "pog_id", "asin", "sku", "id"],
    "title": ["title", "product_sku_name", "product_name", "product_name_scraped", "pt_name", "name"],
    "brand": ["brand", "brand_name", "brand_scraped"],
    "category": ["category", "category_norm", "category_name", "category_group", "category_v2"],
    "subcategory": ["subcategory", "sub_category_name", "subcategory_name", "pt_name"],
    "gender": ["gender", "Gender", "gender_norm"],
    "color": ["color", "color_primary", "color_norm"],
    "pattern": ["pattern", "pattern_norm"],
    "fabric": ["fabric"],
    "fit": ["fit"],
    "type": ["type", "type_field", "pt_name", "tshirt_type", "track_type"],
    "neck": ["neck", "neck_v2"],
    "size": ["size", "product_size"],
    "price": [
        "price",
        "product_variation_price",
        "price_scraped",
        "sale_price",
        "current_price",
        "current_selling_price",
        "product_discounted_price",
        "product_price",
    ],
    "url": ["url", "product_sku_url", "source_url", "final_url", "product_url"],
    "image_urls": ["image_urls", "product_images_urls", "image_urls_scraped"],
    "description": ["description", "description_scraped"],
}


@dataclass(frozen=True)
class PlatformDataset:
    platform: str
    seller: str
    raw: pd.DataFrame
    canonical: pd.DataFrame


def filter_by_seller(df: pd.DataFrame, seller: str) -> pd.DataFrame:
    seller_norm = clean_text(seller).lower()
    if not seller_norm:
        return df
    for column in SELLER_ALIASES:
        if column in df.columns:
            mask = df[column].astype(str).str.strip().str.lower().eq(seller_norm)
            if mask.any():
                return df.loc[mask].copy()
    return df


def canonicalize(df: pd.DataFrame, platform: str, seller: str) -> PlatformDataset:
    df = filter_by_seller(df, seller)
    records = []
    for row_number, row in df.iterrows():
        title = first_present(row, ALIASES["title"])
        brand = first_present(row, ALIASES["brand"])
        category = first_present(row, ALIASES["category"])
        subcategory = first_present(row, ALIASES["subcategory"])
        product_type = first_present(row, ALIASES["type"])
        pattern = first_present(row, ALIASES["pattern"])
        fabric = first_present(row, ALIASES["fabric"])
        fit = first_present(row, ALIASES["fit"])
        color = first_present(row, ALIASES["color"])
        neck = first_present(row, ALIASES["neck"])
        gender = first_present(row, ALIASES["gender"])
        description = first_present(row, ALIASES["description"])

        if not description:
            description = " ".join(
                clean_text(value)
                for value in [brand, product_type, subcategory, category, color, pattern, fabric, fit, neck]
                if clean_text(value)
            )
        if not title or clean_text(title).lower() == clean_text(product_type).lower():
            title = description

        normalized_category = normalize_category(
            f"{subcategory} {product_type} {title}",
            category,
        )
        normalized_gender = normalize_gender(gender, f"{title} {category} {subcategory}")
        normalized_pattern = normalize_pattern(pattern, title)
        normalized_fit = normalize_fit(fit, title)
        normalized_type = normalize_type(product_type, title)
        normalized_neck = normalize_neck(neck, title)

        if normalized_category == "tshirts":
            group_key = "|".join([normalized_gender, normalized_category, normalized_type, normalized_pattern, normalized_fit])
            blocking_key = "|".join([normalized_gender, normalized_category, normalized_type])
        elif normalized_category == "sweatshirts":
            group_key = "|".join([normalized_gender, normalized_category, normalized_pattern, normalized_fit, normalized_neck])
            blocking_key = "|".join([normalized_gender, normalized_category, normalized_neck])
        elif normalized_category == "trackpants and tracksuits":
            group_key = "|".join([normalized_gender, normalized_category, normalized_type, normalized_fit])
            blocking_key = "|".join([normalized_gender, normalized_category, normalized_type])
        else:
            group_key = "|".join([normalized_gender, normalized_category, normalized_pattern, normalized_fit])
            blocking_key = "|".join([normalized_gender, normalized_category])

        product_id = clean_text(first_present(row, ALIASES["product_id"])) or f"{platform}-{row_number}"
        record = {
            "platform": platform,
            "seller": seller,
            "row_number": row_number,
            "product_id": product_id,
            "title": clean_text(title),
            "brand": clean_text(brand),
            "category": clean_text(category),
            "subcategory": clean_text(subcategory),
            "gender": clean_text(gender),
            "color": clean_text(color),
            "pattern": clean_text(pattern),
            "fabric": clean_text(first_present(row, ALIASES["fabric"])),
            "fit": clean_text(fit),
            "type": clean_text(product_type),
            "neck": clean_text(neck),
            "size": clean_text(first_present(row, ALIASES["size"])),
            "price": parse_price(first_present(row, ALIASES["price"])),
            "url": clean_text(first_present(row, ALIASES["url"])),
            "image_urls": clean_text(first_present(row, ALIASES["image_urls"])),
            "description": clean_text(description),
            "category_norm": normalized_category,
            "gender_norm": normalized_gender,
            "pattern_norm": normalized_pattern,
            "fit_norm": normalized_fit,
            "type_norm": normalized_type,
            "neck_norm": normalized_neck,
            "color_norm": color_family(color),
            "group_key": group_key,
            "blocking_key": blocking_key,
        }
        record["match_text"] = " ".join(
            clean_text(record[name])
            for name in [
                "title",
                "brand",
                "category_norm",
                "subcategory",
                "pattern_norm",
                "fabric",
                "fit_norm",
                "type_norm",
                "neck_norm",
                "color",
                "description",
            ]
        )
        records.append(record)

    canonical = pd.DataFrame(records).drop_duplicates(["platform", "product_id"], keep="first")
    return PlatformDataset(platform=platform, seller=seller, raw=df, canonical=canonical)
