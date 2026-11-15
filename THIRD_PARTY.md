# Third-party components

| component | how it is used | licence | where |
|---|---|---|---|
| PathCRF (Kim et al., arXiv 2602.12080) | run unmodified as an external tool in its own Python environment; its code and its pretrained checkpoint `saved/140` are **not** copied into this repository (`third_party/pathcrf/run_infer.py` is our own wrapper that imports the upstream checkout) | Mozilla Public License 2.0 (the upstream `LICENSE` file is the MPL-2.0 text, as found at the pinned commit) | https://github.com/hyunsungkim-ds/pathcrf , pinned commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997` |
| organisers' scorer `possession_metrics.py` | vendored unmodified in `mtah/eval/official/` (md5-pinned in the code) | MIT, copyright 2026-2027 Metrica Sports; notice in `mtah/eval/official/LICENSE-organisers` | the organisers' repository |
| DFL / Sportec open data (Bassek et al., Sci. Data 12, 195, 2025) | training and selection proxy; no data in this repository | CC BY 4.0 | https://doi.org/10.6084/m9.figshare.28196177 |
| SkillCorner open data (A-League 2024/25) | other-game hold-out; no data in this repository | MIT | https://github.com/SkillCorner/opendata |
| Metrica Sports sample data | other-game validation; no data in this repository | see that repository | https://github.com/metrica-sports/sample-data |
| YOLOv8x (COCO weights) and PANNs Cnn14 (AudioSet) | used only in two rejected experiments described in the technical document, section 2; no weights in this repository | their upstream licences | upstream repositories |
