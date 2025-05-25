import os, base64, io, json, numpy as np, math
from PIL import Image
import onnxruntime as rt
from supabase import create_client, Client

# ---- load model once (cold start) ----
sess = rt.InferenceSession("clip_b32_image.onnx",
                           providers=["CPUExecutionProvider"])
INP = sess.get_inputs()[0].name
OUT = sess.get_outputs()[0].name

SB: Client = create_client(os.environ["SUPABASE_URL"],
                           os.environ["SUPABASE_SERVICE_KEY"])

def cosine(a, b):
    a=np.asarray(a); b=np.asarray(b)
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-7))

def diversity(items, min_cos=0.95, k=6):
    out=[]
    for it in items:
        if all(cosine(it["embedding"],x["embedding"])<min_cos for x in out):
            out.append(it)
        if len(out)==k: break
    return out

async def handler(req, res):
    body = await req.json()
    img  = Image.open(io.BytesIO(base64.b64decode(body["image_base64"])))\
               .convert("RGB").resize((224,224))
    arr  = np.expand_dims(np.transpose(np.array(img)/255.0,(2,0,1)),0)\
             .astype("float32")
    vec  = sess.run([OUT], {INP: arr})[0][0].tolist()

    top30 = SB.rpc("top30", {"query_vec": vec}).execute().data

    # exact cosine rerank
    for r in top30:
        r["score"] = cosine(r["embedding"], vec)
    top30.sort(key=lambda x:-x["score"])

    final = diversity(top30)

    # prepare response
    bucket = "place-images"
    for r in final:
        r.pop("embedding")
        r["image_url"] = f"{os.environ['SUPABASE_URL']}/storage/v1/object/public/"\
                         f"{bucket}/{r['place_id']}.jpg"
    return res.json(final)
