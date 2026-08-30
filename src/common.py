import json, re, os
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def parse_json(txt):
    """Extract the first JSON object from model text. Returns dict or None."""
    if not txt:
        return None
    t = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)          # drop thinking blocks if any
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # tolerate trailing commas / single quotes
        s = re.sub(r",\s*}", "}", m.group(0)).replace("'", '"')
        try:
            return json.loads(s)
        except Exception:
            return None

def norm_dx(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in CLASSES:
        return s
    if "uncertain" in s or "unknown" in s:
        return "uncertain"
    for c in CLASSES:                       # e.g. "melanoma (mel)"
        if re.search(rf"\b{c}\b", s):
            return c
    alias = {"melanoma": "mel", "nevus": "nv", "naevus": "nv", "basal": "bcc", "keratosis": "bkl",
             "dermatofibroma": "df", "vascular": "vasc", "actinic": "akiec"}
    for k, c in alias.items():
        if k in s:
            return c
    return None

def norm_conf(v):
    try:
        x = float(str(v).strip().rstrip("%"))
    except Exception:
        return None
    if 0 <= x <= 1:
        x *= 100
    return max(0.0, min(100.0, x))

def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["image_id"] for l in open(path) if l.strip()}
