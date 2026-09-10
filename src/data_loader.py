"""
Dataset loader and conversation parser for Customer Support on Twitter.
Streams AppleSupport conversation threads, parses customer and support turns,
and constructs resolution knowledge bases and evaluation candidates.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.config import DEFAULT_BRAND, HISTORICAL_RESOLUTIONS_PATH

logger = logging.getLogger(__name__)

HF_DATASET_URL = "https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation/resolve/main/data/train-00000-of-00001.parquet"


def parse_conversation_turns(conversation_text: str) -> List[Dict[str, str]]:
    """
    Parse a conversation text formatted as 'Customer: ... Support: ...' into turns.
    Returns a list of dictionaries with 'role' ('Customer' or 'Support') and 'text'.
    """
    turns = []
    lines = conversation_text.strip().split("\n")
    current_role = None
    current_text = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        if line_clean.startswith("Customer:"):
            if current_role and current_text:
                turns.append({"role": current_role, "text": " ".join(current_text).strip()})
            current_role = "Customer"
            current_text = [line_clean[len("Customer:"):].strip()]
        elif line_clean.startswith("Support:"):
            if current_role and current_text:
                turns.append({"role": current_role, "text": " ".join(current_text).strip()})
            current_role = "Support"
            current_text = [line_clean[len("Support:"):].strip()]
        else:
            if current_role:
                current_text.append(line_clean)

    if current_role and current_text:
        turns.append({"role": current_role, "text": " ".join(current_text).strip()})

    return turns


def extract_primary_pair(conversation_text: str) -> Optional[Tuple[str, str]]:
    """
    Extracts the initial customer query and the brand's primary resolution/reply.
    Returns (customer_query, brand_reply) or None if invalid.
    """
    turns = parse_conversation_turns(conversation_text)
    if len(turns) < 2:
        return None

    # Find first customer turn and subsequent support turn
    cust_idx = None
    for i, t in enumerate(turns):
        if t["role"] == "Customer" and len(t["text"]) > 10:
            cust_idx = i
            break

    if cust_idx is None:
        return None

    support_idx = None
    for i in range(cust_idx + 1, len(turns)):
        if turns[i]["role"] == "Support" and len(turns[i]["text"]) > 10:
            support_idx = i
            break

    if support_idx is None:
        return None

    cust_msg = clean_tweet_text(turns[cust_idx]["text"])
    support_msg = clean_tweet_text(turns[support_idx]["text"])

    if not cust_msg or not support_msg:
        return None

    return cust_msg, support_msg


def clean_tweet_text(text: str) -> str:
    """
    Cleans raw tweet text while preserving URLs, technical symbols, and punctuation.
    """
    # Remove leading mentions like @AppleSupport or @115858
    cleaned = re.sub(r"^(@\w+\s*)+", "", text)
    # Remove surrogate/garbled characters from encoding issues
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)
    # Fix repeated whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def stream_brand_conversations(
    company: str = DEFAULT_BRAND,
    max_conversations: int = 500,
    max_row_groups: int = 20,
    dataset_url: str = HF_DATASET_URL
) -> List[Dict]:
    """
    Streams row groups from the remote parquet file and extracts conversations for a given company.
    """
    import pyarrow.parquet as pq
    import fsspec

    results = []
    logger.info(f"Connecting to dataset stream for company '{company}'...")

    fs, path = fsspec.core.url_to_fs(dataset_url)
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        total_rg = min(pf.num_row_groups, max_row_groups)

        for rg_idx in range(total_rg):
            batch = pf.read_row_group(rg_idx, columns=["company", "conversation", "conversation_id"])
            df = batch.to_pandas()
            brand_df = df[df["company"] == company]

            for _, row in brand_df.iterrows():
                cid = row["conversation_id"]
                conv = row["conversation"]
                pair = extract_primary_pair(conv)
                if pair:
                    cust_msg, supp_msg = pair
                    results.append({
                        "id": cid,
                        "company": company,
                        "customer_message": cust_msg,
                        "support_reply": supp_msg,
                        "raw_conversation": conv
                    })
                    if len(results) >= max_conversations:
                        break

            if len(results) >= max_conversations:
                break

    logger.info(f"Streamed {len(results)} valid conversation pairs for {company}.")
    return results


def load_or_create_historical_resolutions(force_refresh: bool = False) -> List[Dict]:
    """
    Loads historical brand resolutions from local disk, or creates it by streaming.
    """
    if HISTORICAL_RESOLUTIONS_PATH.exists() and not force_refresh:
        with open(HISTORICAL_RESOLUTIONS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data)} historical resolutions from {HISTORICAL_RESOLUTIONS_PATH}")
            return data

    resolutions = stream_brand_conversations(max_conversations=300)
    with open(HISTORICAL_RESOLUTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(resolutions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(resolutions)} resolutions to {HISTORICAL_RESOLUTIONS_PATH}")
    return resolutions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_or_create_historical_resolutions()
    print(f"Sample resolution count: {len(data)}")
    if data:
        print("Sample 1:")
        print("Customer:", data[0]["customer_message"])
        print("Support:", data[0]["support_reply"])
