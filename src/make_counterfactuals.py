"""Lesion-masked / same-size control block / background-masked variants.
Subset: up to 50 images per class (= 306 total on this split), stratified, seed 42."""
import pandas as pd, numpy as np, os
from PIL import Image
m = pd.read_csv("data/ham_test_manifest.csv")
m = m[m.mask_path.astype(str).str.len() > 3]
m = pd.concat([g.sample(min(50, len(g)), random_state=42) for _, g in m.groupby("label")])
os.makedirs("data/cf", exist_ok=True); rows = []; rng = np.random.default_rng(42); dropped = 0
for _, r in m.iterrows():
    im = np.array(Image.open(r.image_path).convert("RGB"))
    mk = np.array(Image.open(r.mask_path).convert("L").resize((im.shape[1], im.shape[0]))) > 127
    if mk.sum() == 0 or (~mk).sum() == 0: dropped += 1; continue
    fill = np.median(im[~mk], axis=0).astype(np.uint8)
    les = im.copy(); les[mk] = fill
    bg = im.copy(); bg[~mk] = fill
    ys, xs = np.where(mk); h, w = int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1
    ctrl = None
    if h < im.shape[0] and w < im.shape[1]:
        for _ in range(300):
            y0 = int(rng.integers(0, im.shape[0] - h)); x0 = int(rng.integers(0, im.shape[1] - w))
            if not mk[y0:y0 + h, x0:x0 + w].any():
                ctrl = im.copy(); ctrl[y0:y0 + h, x0:x0 + w] = fill; break
    if ctrl is None: dropped += 1; continue
    for name, arr in [("lesion", les), ("control", ctrl), ("background", bg)]:
        p = f"data/cf/{r.image_id}_{name}.jpg"; Image.fromarray(arr).save(p, quality=95)
        rows.append(dict(image_id=f"{r.image_id}__{name}", image_path=p, label=r.label, mask_path="", variant=name, orig_id=r.image_id))
df = pd.DataFrame(rows); df.to_csv("data/cf_manifest.csv", index=False)
print("variants:", len(df), "images:", len(df) // 3, "dropped:", dropped)
print(df[df.variant == "lesion"].label.value_counts().to_dict())
