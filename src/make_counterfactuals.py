"""Lesion / area-matched control / background variants. Keeps sampling per class until 50 kept (seed 42).
Controls matched on occluded area (>=0.5x lesion area). Excluded images (no background room) are logged."""
import pandas as pd, numpy as np, os
from PIL import Image
m = pd.read_csv("data/ham_test_manifest.csv"); m = m[m.mask_path.astype(str).str.len() > 3]
os.makedirs("data/cf", exist_ok=True); rows = []; rng = np.random.default_rng(42); dropped = []
def make(r):
    im = np.array(Image.open(r.image_path).convert("RGB"))
    mk = np.array(Image.open(r.mask_path).convert("L").resize((im.shape[1], im.shape[0]))) > 127
    if mk.sum() == 0 or (~mk).sum() == 0: return None
    fill = np.median(im[~mk], axis=0).astype(np.uint8)
    les = im.copy(); les[mk] = fill
    bg = im.copy(); bg[~mk] = fill
    ys, xs = np.where(mk); aspect = (int(np.ptp(ys)) + 1) / (int(np.ptp(xs)) + 1)
    area = int(mk.sum())
    for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        a = area * scale
        h = min(int(round((a * aspect) ** 0.5)), im.shape[0] - 1)
        w = min(int(round(a / max(h, 1))), im.shape[1] - 1)
        if h < 8 or w < 8: continue
        for _ in range(400):
            y0 = int(rng.integers(0, im.shape[0] - h)); x0 = int(rng.integers(0, im.shape[1] - w))
            if not mk[y0:y0 + h, x0:x0 + w].any():
                ctrl = im.copy(); ctrl[y0:y0 + h, x0:x0 + w] = fill
                return les, ctrl, bg, round(h * w / area, 3)
    return None
for lbl, g in m.groupby("label"):
    kept = 0
    for _, r in g.sample(frac=1, random_state=42).iterrows():
        if kept >= 50: break
        out = make(r)
        if out is None: dropped.append(r.image_id); continue
        les, ctrl, bg, ratio = out; kept += 1
        for name, arr in [("lesion", les), ("control", ctrl), ("background", bg)]:
            p = f"data/cf/{r.image_id}_{name}.jpg"; Image.fromarray(arr).save(p, quality=95)
            rows.append(dict(image_id=f"{r.image_id}__{name}", image_path=p, label=r.label, mask_path="",
                             variant=name, orig_id=r.image_id, control_area_ratio=ratio))
df = pd.DataFrame(rows); df.to_csv("data/cf_manifest.csv", index=False)
pd.Series(dropped, name="image_id").to_csv("data/cf_dropped.csv", index=False)
print("variants:", len(df), "images:", len(df) // 3, "dropped:", len(dropped))
print(df[df.variant == "lesion"].label.value_counts().to_dict())
print("control/lesion area ratio: median", df.control_area_ratio.median(), "min", df.control_area_ratio.min())
