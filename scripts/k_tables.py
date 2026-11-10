"""Markdown tables from runs/kh/agg.json -> runs/kh/K_tables.md"""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "runs" / "kh"
a = json.loads((R / "agg.json").read_text())
L = []
for b, t in (("nat", "K-nat (SkillCorner natural gaps, X = no detected player)"), ("sim", "K-sim (same halves, blackout simulator, runs/sim_extra2)")):
    L.append(f"\n#### {t}, {a[b]['n_halves']} halves\n\n| candidate | spellF1 ctrl mean | sd | spellF1 raw mean | macroF1 ctrl |\n|---|---|---|---|---|")
    for c, v in a[b]["cand"].items():
        L.append(f"| {c} | {v['ctrl_Dfull']['spellF1_mean']:.3f} | {v['ctrl_Dfull']['sd']:.3f} | {v['raw_Dfull']['spellF1_mean']:.3f} | {v['ctrl_Dfull']['macroF1_mean']:.3f} |")
    L.append("\nPaired (x minus y, ctrl_Dfull):\n\n| pair | mean diff | x wins | y wins |\n|---|---|---|---|")
    for p, v in a[b]["pairs"].items():
        z = v["ctrl_Dfull"]; L.append(f"| {p.replace(':', ' - ')} | {z['mean_diff_x_minus_y']:+.4f} | {z['x_wins']:.2f} | {z['y_wins']:.2f} |")
(R / "K_tables.md").write_text("\n".join(L), encoding="utf-8")
print("ok")
