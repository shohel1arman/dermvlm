"""Re-derive parsed/diagnosis/confidence from stored raw text in all output JSONLs (idempotent)."""
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import parse_json, norm_dx, norm_conf
for f in glob.glob("outputs/*.jsonl"):
    rows, changed = [], 0
    for l in open(f):
        if not l.strip(): continue
        r = json.loads(l); p = parse_json(r.get("raw", ""))
        new = dict(parsed=p, parse_fail=int(p is None),
                   diagnosis=norm_dx(p.get("diagnosis")) if p else None,
                   confidence=norm_conf(p.get("confidence")) if p else None)
        if any(r.get(k) != v for k, v in new.items()): changed += 1
        r.update(new); rows.append(r)
    if changed:
        with open(f, "w") as out:
            for r in rows: out.write(json.dumps(r, default=str) + "\n")
    print(f"{os.path.basename(f)}: {changed} records repaired")
