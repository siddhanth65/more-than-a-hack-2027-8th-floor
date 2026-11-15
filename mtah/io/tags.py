"""Metrica Smart Tagging (SportsCode XML) as frame intervals."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


def _team(text):
    if not text:
        return None
    t = text.strip().lower()
    if t.startswith("barc"):
        return "H"
    if t.startswith("at"):
        return "A"
    return None


def _half(text):
    if not text:
        return None
    t = text.strip().lower()
    return 1 if t.startswith("1") else 2 if t.startswith("2") else None


def parse_sportscode(xml_path, fps=25):
    rows = []
    for inst in ET.parse(xml_path).getroot().iter("instance"):
        labels = {}
        for lbl in inst.findall("label"):
            labels[(lbl.findtext("group") or "").strip()] = (lbl.findtext("text") or "").strip()
        s = float(inst.findtext("start")); e = float(inst.findtext("end"))
        rows.append({
            "start": int(round(s * fps)), "end": int(round(e * fps)),
            "code": (inst.findtext("code") or "").strip().upper(),
            "team": _team(labels.get("Team")), "type": labels.get("Type") or None,
            "half": _half(labels.get("Half")),
        })
    cols = ["start", "end", "code", "team", "type", "half"]
    data = {c: [r[c] for r in rows] for c in cols}
    return pd.DataFrame({c: pd.Series(data[c], dtype=(object if c in ("team", "type", "half") else None))
                         for c in cols}, columns=cols)


def shift_set_pieces(tags, shifts, fps=25):
    """Amendment 5: move SET PIECES rows (start and end together) by shifts[type prefix] seconds (key "*" = every other type;
    prefixes are lower-case: "throw", "goal", "corner", "free", "kick"). Positive = later. Returns a copy."""
    out = tags.copy()
    for i, r in enumerate(out.itertuples(index=False)):
        if r.code != "SET PIECES":
            continue
        t = (r.type or "").lower()
        sec = shifts.get("*", 0.0)
        for k, v in shifts.items():
            if k != "*" and t.startswith(k):
                sec = v; break
        d = int(round(sec * fps))
        out.iat[i, out.columns.get_loc("start")] = int(r.start) + d
        out.iat[i, out.columns.get_loc("end")] = int(r.end) + d
    return out


def env_shift(var):
    """Opt-in JSON shift from the environment variable `var` (None when unset)."""
    import json, os
    v = os.environ.get(var)
    return json.loads(v) if v else None


def load_tags(data_dir):
    tags = parse_sportscode(Path(data_dir) / "tactical" / "BAR-ATH_tactical_smart_tagging.xml")
    sh = env_shift("MTAH_SP_SHIFT_REAL")   # Amendment 5 corrected candidate; off by default
    return shift_set_pieces(tags, sh) if sh else tags


def coverage(df, n_frames, team_only=False):
    cov = np.zeros(n_frames, dtype=bool)
    for r in df.itertuples(index=False):
        if team_only and (r.team is None or pd.isna(r.team)):
            continue
        a = max(0, r.start); b = min(n_frames - 1, r.end)
        if b >= a:
            cov[a:b + 1] = True
    return cov
