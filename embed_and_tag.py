#!/usr/bin/env python
"""
upload_to_supabase.py  –  final robust version
• Works with wide vectors even if column names have extra spaces
• Still falls back to packed 'embedding' column when needed
"""

import os, json, argparse, pandas as pd, ast, re
from dotenv import load_dotenv
from supabase import create_client, Client

# ─── ENV / CLIENT ────────────────────────────────────────────────── #
load_dotenv()

def get_client() -> Client:
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
    if not (url and key):
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_KEY")
    return create_client(url, key)

def upsert_bulk(sp: Client, table: str, rows: list[dict], chunk=100):
    for i in range(0, len(rows), chunk):
        sp.table(table).upsert(rows[i : i + chunk]).execute()

# ─── HELPERS ─────────────────────────────────────────────────────── #
def safe_tags(cell):
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str) and cell.strip():
        try:
            return json.loads(cell)
        except json.JSONDecodeError:
            return [cell]
    return []

def parse_embedding(row, embed_cols):
    # --- wide layout ---
    if embed_cols:
        try:
            return [float(row[c]) for c in embed_cols]
        except Exception:
            return None
    # --- packed layout ---
    s = row.get("embedding", "")
    if isinstance(s, str) and "[" in s and "]" in s:
        for parser in (json.loads, ast.literal_eval):
            try:
                vec = parser(s)
                if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                    return vec
            except Exception:
                continue
    return None

# ─── MAIN ────────────────────────────────────────────────────────── #
def main(city: str):
    sp = get_client()

    df_places = pd.read_csv(f"places_{city}.csv")

    df_vec = pd.read_csv(
        f"vectors_{city}.csv",
        dtype={"vibe_tags": "string"},
        keep_default_na=False,
    ).fillna({"vibe_tags": ""})

    # -------- strip spaces from column names --------
    df_vec.columns = [c.strip() for c in df_vec.columns]

    embed_cols = [c for c in df_vec.columns if re.fullmatch(r"v\d+", c)]

    # ---------- build place rows ----------
    places_rows = [
        {"id": str(r.osm_id), "name": r.name, "lat": float(r.lat),
         "lon": float(r.lon), "category": r.category, "raw_tags": r.raw_tags}
        for r in df_places.itertuples(index=False)
    ]

    images_rows, skipped = [], 0
    for _, row in df_vec.iterrows():
        vec = parse_embedding(row, embed_cols)
        if vec is None:
            skipped += 1
            continue
        images_rows.append({
            "place_id": str(row.osm_id),
            "embedding": vec,
            "vibe_tags": safe_tags(row.vibe_tags),
        })

    if skipped:
        print(f"⚠️  Skipped {skipped} rows with missing/bad embeddings")

    upsert_bulk(sp, "places", places_rows)
    upsert_bulk(sp, "images", images_rows)
    print(f"✔ Uploaded {len(places_rows)} places and {len(images_rows)} embeddings")

# ─── CLI ─────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--city", default="delhi")
    args = p.parse_args()
    main(args.city)
