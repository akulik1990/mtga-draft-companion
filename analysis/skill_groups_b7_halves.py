import sys, zlib, io, contextlib
sys.path.insert(0, "/home/morris/Projects/mtga-draft-companion/analysis")
import skill_groups_dropped as sg, pandas as pd
A = sg.A
src = open(f"{A}/podcast_lands18.py").read(); i0 = src.index("# ---------- (a)"); i1 = src.index("by = {c["); i2 = src.index("for smin in")
ns = {"__name__": "x", "__file__": f"{A}/podcast_lands18.py"}
with sg.quiet(): exec(compile(src[:i0] + src[i1:i2], "lands18", "exec"), ns)
LD = ns["build"](0.0); LD["h"] = LD.draft.map(lambda x: zlib.crc32(x.encode()) % 2)
for g in ("all", "good", "avg", "weak"):
    for h in (0, 1):
        s = sg.sel(LD, g); s = s[(s.h == h) & (s.ct >= 2) & (s.lands != 18)].copy(); s["x"] = (s.lands == 16).astype(int); s["y"] = s.w / s.g
        m = sg.glm_prop(s, "y ~ x + quality + skill + C(lane)"); e, se = sg.contrast(m, {"x": 1})
        print(g, "half", h, f"{e:+.2f} [{e-1.96*se:+.2f}, {e+1.96*se:+.2f}]", int(s.x.sum()), int((1-s.x).sum()))
