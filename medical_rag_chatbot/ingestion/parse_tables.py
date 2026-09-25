import os
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import pdfplumber

from ingestion.parse_pdf import extract_claim_id

def row_to_string(headers: List[str], values: List[Any]) -> str:
    """Format row as: Headers: col1 | col2. Values: val1 | val2"""
    clean_headers = [str(h).strip() for h in headers if h is not None]
    clean_values = [str(v).strip() if v is not None else "" for v in values]
    header_str = " | ".join(clean_headers)
    value_str = " | ".join(clean_values)
    return f"Headers: {header_str}. Values: {value_str}"

def parse_csv(filepath: str | Path) -> List[Dict[str, Any]]:
    """
    Parse a CSV or Excel table file into standardized row chunks.
    Normalizes column headers to snake_case and builds metadata per row.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Table file not found: {filepath}")

    filename = filepath.name
    claim_id = extract_claim_id(filename)

    if filepath.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    # Normalize column names to snake_case
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    headers = list(df.columns)
    results = []

    for idx, row in df.iterrows():
        # Check if row has an explicit claim_id column
        row_claim_id = claim_id
        if "claim_id" in row and pd.notna(row["claim_id"]):
            row_claim_id = str(row["claim_id"]).strip().upper()

        values = [row[c] for c in headers]
        row_str = row_to_string(headers, values)

        metadata = {
            "source_file": filename,
            "row_index": int(idx),
            "claim_id": row_claim_id,
            "doc_type": "billing",
            "modality": "table",
            "page_number": 1,
            "file_path": str(filepath)
        }
        results.append({
            "text": row_str,
            "metadata": metadata
        })

    return results

def parse_tables_from_pdf(filepath: str | Path) -> List[Dict[str, Any]]:
    """
    Extract tables embedded inside a PDF using pdfplumber (with camelot support if present).
    Each table row becomes one chunk prepended with headers.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    filename = filepath.name
    claim_id = extract_claim_id(filename)
    results = []

    # Use pdfplumber table extraction
    with pdfplumber.open(str(filepath)) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for tbl_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue
                # First row as header
                raw_headers = table[0]
                headers = [str(h).strip() if h else f"col_{j}" for j, h in enumerate(raw_headers)]
                
                for row_idx, row in enumerate(table[1:], start=1):
                    if not any(row):
                        continue
                    row_str = row_to_string(headers, row)
                    metadata = {
                        "source_file": filename,
                        "page_number": page_idx,
                        "table_index": tbl_idx,
                        "row_index": row_idx,
                        "claim_id": claim_id,
                        "doc_type": "billing",
                        "modality": "table",
                        "file_path": str(filepath)
                    }
                    results.append({
                        "text": row_str,
                        "metadata": metadata
                    })

    return results
