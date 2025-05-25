#!/usr/bin/env python
"""
LoRA-tune the OpenCLIP **text** tower on five “vibe” classes, then export the
(pre-trained) **vision** tower to ONNX.  Works on Torch 1.13.1+cpu → ONNX opset 12
without `_native_multi_head_attention`.
"""

import json, math, torch, torch.nn as nn, pandas as pd
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import open_clip

# ─────────────────────────── Device ────────────────────────────── #
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", DEV)

# ──────────────── Vibe prompts & label mapping ─────────────────── #
PROMPTS = [
    "a peaceful historical monument",
    "a lively crowded marketplace",
    "a romantic candle-lit restaurant",
    "a vibrant nightlife bar",
    "a lush green park",
]
LBL = {p: i for i, p in enumerate(PROMPTS)}

# ─────────── Map images → label using vectors_delhi.csv ────────── #
vec_df = pd.read_csv("vectors_delhi.csv", usecols=["osm_id", "vibe_tags"])
file2lab = {}
for _, row in vec_df.iterrows():
    prompt = json.loads(row.vibe_tags)[0]
    if prompt in LBL:
        file2lab[f"images/{row.osm_id}.jpg"] = LBL[prompt]
print("usable images:", len(file2lab))

# ──────────────────────── Dataset / loader ─────────────────────── #
class ImgDS(Dataset):
    tfm = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.4815, 0.4578, 0.4082],
                  [0.2686, 0.2613, 0.2758]),
    ])
    def __init__(self, mp): self.items = list(mp.items())
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, lab = self.items[idx]
        return self.tfm(Image.open(path).convert("RGB")), lab

loader = DataLoader(ImgDS(file2lab), batch_size=32, shuffle=True)

# ──────────────── OpenCLIP + LoRA on *text* tower ──────────────── #
model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", device="cpu")
tok = open_clip.get_tokenizer("ViT-B-32")

# LoRA targets = every Linear layer inside text transformer
targets = sorted({
    n.split(".")[-1]
    for n, m in model.named_modules()
    if isinstance(m, nn.Linear) and n.startswith("transformer")
})
cfg = LoraConfig(
    task_type="FEATURE_EXTRACTION",
    r=4, lora_alpha=8,
    target_modules=targets,
    lora_dropout=0.1
)
model = get_peft_model(model, cfg).to(DEV)
for n, p in model.named_parameters():
    p.requires_grad = "lora_" in n

class_tok = tok(PROMPTS).to(DEV)
optim = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
ce = nn.CrossEntropyLoss()

for epoch in range(5):
    tot = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEV), labels.to(DEV)
        img_f = model.encode_image(imgs)      # B × 512
        txt_f = model.encode_text(class_tok)  # 5 × 512
        loss  = ce(img_f @ txt_f.T, labels)
        optim.zero_grad(); loss.backward(); optim.step()
        tot += loss.item()
    print(f"epoch {epoch+1}  loss {tot/len(loader):.4f}")

# ──────────── Patch vision-tower MHA for ONNX export ───────────── #
class SimpleMHA(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention that uses only
    matmul / softmax – safe for ONNX opset 11/12.
    """
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self,
                query,
                key=None,
                value=None,
                need_weights: bool = False,
                attn_mask=None,
                average_attn_weights: bool = False):
        # mirror nn.MultiheadAttention signature
        K = key   if key   is not None else query
        V = value if value is not None else K

        B, N, C = query.shape
        # project & reshape for multi-head
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(K).view(B, N, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(V).view(B, N, self.num_heads, self.head_dim).transpose(1,2)

        # scaled dot-prod attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = scores.softmax(-1)

        out = (weights @ v).transpose(1,2).reshape(B, N, C)
        y   = self.out_proj(out)

        # we never return weights for CLIP vision
        return (y, weights) if need_weights else y

# swap every nn.MultiheadAttention in the vision tower
for blk in model.visual.transformer.resblocks:
    orig: nn.MultiheadAttention = blk.attn
    smha = SimpleMHA(orig.embed_dim, orig.num_heads, bias=True)

    # copy the weights & biases
    E = orig.embed_dim
    with torch.no_grad():
        smha.q_proj.weight.copy_(orig.in_proj_weight[:E])
        smha.k_proj.weight.copy_(orig.in_proj_weight[E:2*E])
        smha.v_proj.weight.copy_(orig.in_proj_weight[2*E:3*E])
        smha.out_proj.weight.copy_(orig.out_proj.weight)
        if orig.in_proj_bias is not None:
            smha.q_proj.bias.copy_(orig.in_proj_bias[:E])
            smha.k_proj.bias.copy_(orig.in_proj_bias[E:2*E])
            smha.v_proj.bias.copy_(orig.in_proj_bias[2*E:3*E])
            smha.out_proj.bias.copy_(orig.out_proj.bias)

    blk.attn = smha

print("Patched Vision MHA → SimpleMHA (weights copied)")

# ───────────────────────── Export to ONNX ───────────────────────── #
model.visual.eval().cpu()
torch.onnx.export(
    model.visual,
    torch.randn(1, 3, 224, 224),
    "clip_b32_lora_image.onnx",
    input_names=["pixel_values"],
    output_names=["emb"],
    opset_version=12,
    do_constant_folding=True,
)
print("✔  ONNX export finished (opset 12, no fused op)")
