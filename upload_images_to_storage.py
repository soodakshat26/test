# scripts/upload_images_to_storage.py
import os, glob
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY")
if not (url and key):
    raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")

sp: Client = create_client(url, key)
bucket = "place-images"

# create bucket if not exists
try:
    sp.storage.create_bucket(bucket)
except Exception as e:
    if "already exists" not in str(e):
        raise

store = sp.storage.from_(bucket)

for fp in glob.glob("images/*.jpg"):
    name = os.path.basename(fp)  # e.g. 10172949302.jpg
    with open(fp, "rb") as f:
        data = f.read()

    # first try upload; if it exists, fall back to update()
    try:
        store.upload(name, data, file_options={"content-type":"image/jpeg"})
    except Exception as e:
        if "The resource already exists" in str(e) or "409" in str(e):
            store.update(name, data, file_options={"content-type":"image/jpeg"})
        else:
            raise

print("✔ all images uploaded / updated in Supabase storage")
