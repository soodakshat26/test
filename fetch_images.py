#!/usr/bin/env python
# fetch_images_fast.py  — concurrent CC-image crawler (Commons ▸ Openverse ▸ Unsplash)

import csv, os, pathlib, argparse, asyncio, random
from urllib.parse import urlencode

import aiohttp
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm   # tqdm ≥ 4.66

# ─────────────────────────────── CONFIG ────────────────────────────── #

OPENVERSE = "https://api.openverse.engineering/v1/images"
UNSPLASH  = "https://api.unsplash.com/search/photos"
UNSPLASH_KEY = os.getenv("Asy7CcHAbnOIX6LavU4_oES47oGa7NSNutC9EjjdwyM", "")          # set this env var if you have a key

# free-tier quotas → limiters (tokens, period-seconds)
lim_commons   = AsyncLimiter(60,   60)       # 60 req / min
lim_openverse = AsyncLimiter(60,   60)
lim_unsplash  = AsyncLimiter(50, 3600)       # 50 req / hour

# ─────────────────────────────── HELPERS ───────────────────────────── #

async def commons_geo(session, lat, lon, radius=300):
    qs  = urlencode({
        "action": "query", "format": "json",
        "prop": "imageinfo", "iiprop": "url",
        "generator": "geosearch",
        "ggscoord": f"{lat}|{lon}", "ggsradius": radius, "ggslimit": 5
    })
    url = f"https://commons.wikimedia.org/w/api.php?{qs}"
    try:
        async with lim_commons:
            async with session.get(url, timeout=20) as r:
                j = await r.json(content_type=None)
        for page in j.get("query", {}).get("pages", {}).values():
            imgs = page.get("imageinfo")
            if imgs:
                return imgs[0]["url"]
    except (asyncio.TimeoutError, aiohttp.ClientError, ValueError):
        pass
    return None


async def openverse_search(session, query):
    params = {"q": query, "page_size": 5}
    try:
        async with lim_openverse:
            async with session.get(OPENVERSE, params=params, timeout=20) as r:
                j = await r.json(content_type=None)
        if j.get("results"):
            return random.choice(j["results"])["url"]
    except (asyncio.TimeoutError, aiohttp.ClientError, ValueError):
        pass
    return None


async def unsplash_search(session, query):
    if not UNSPLASH_KEY:
        return None
    hdrs  = {"Authorization": f"Client-ID {UNSPLASH_KEY}"}
    params = {"query": query, "per_page": 5}
    try:
        async with lim_unsplash:
            async with session.get(UNSPLASH, params=params,
                                   headers=hdrs, timeout=20) as r:
                j = await r.json(content_type=None)
        if j.get("results"):
            return random.choice(j["results"])["urls"]["regular"]
    except (asyncio.TimeoutError, aiohttp.ClientError, ValueError):
        pass
    return None


async def choose_image(session, name, city, lat, lon):
    return (
        await commons_geo(session, lat, lon)
        or await openverse_search(session, f"{name} {city}")
        or await unsplash_search(session, f"{name.split()[0]} {city} travel")
    )

# ─────────────────────────────── CRAWLER ───────────────────────────── #

async def crawl_city(city: str, workers: int = 20):
    places = list(csv.DictReader(open(f"places_{city}.csv", encoding="utf-8")))
    out_dir = pathlib.Path("images"); out_dir.mkdir(exist_ok=True)
    out_csv = pathlib.Path(f"place_images_{city}.csv")

    # resume-aware
    done = set()
    if out_csv.exists():
        for row in csv.DictReader(open(out_csv, encoding="utf-8")):
            done.add(row["osm_id"])

    csv_fp = open(out_csv, "a", newline='', encoding="utf-8")
    writer = csv.writer(csv_fp)
    if out_csv.stat().st_size == 0:
        writer.writerow(["osm_id", "img_url", "source"])

    sem = asyncio.Semaphore(workers)

    async with aiohttp.ClientSession() as session:

        async def handle(p):
            osm_id, name, lat, lon = p["osm_id"], p["name"], p["lat"], p["lon"]
            if osm_id in done:
                return
            img_path = out_dir / f"{osm_id}.jpg"
            if img_path.exists():
                return

            try:
                async with sem:
                    url = await choose_image(session, name, city.capitalize(), lat, lon)
                    if not url:
                        return
                    async with session.get(url, timeout=30) as resp:
                        data = await resp.read()
                    img_path.write_bytes(data)
                    source = (
                        "commons"   if "wikimedia"      in url else
                        "openverse" if "creativecommons" in url else
                        "unsplash"
                    )
                    writer.writerow([osm_id, url, source])
                    csv_fp.flush()
            except Exception:            # any timeout / 5xx / write error
                img_path.unlink(missing_ok=True)

        await tqdm.gather(*(handle(p) for p in places),
                           total=len(places), desc="images")

    csv_fp.close()

# ─────────────────────────────── CLI ──────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city",    default="delhi")
    ap.add_argument("--workers", type=int, default=20,
                    help="parallel download slots (default 20)")
    args = ap.parse_args()
    asyncio.run(crawl_city(args.city, args.workers))
    print("✔ Image crawl finished")

if __name__ == "__main__":
    main()
