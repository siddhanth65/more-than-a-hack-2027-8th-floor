"""Evidence -> decoder -> post-processing: one function from inputs to labels."""
from dataclasses import asdict, dataclass, field
from typing import Optional

from mtah.decode.viterbi import DecoderParams, decode
from mtah.evidence.ball_ctrl import BallCtrlParams, ball_ctrl_evidence
from mtah.evidence.dead import dead_evidence, dead_features
from mtah.evidence.pathcrf import PathCRFParams, pathcrf_evidence, pathcrf_posterior
from mtah.evidence.tag_prior import TagPriorParams, load_tag_codes, tag_prior_evidence
from mtah.evidence.touches import TouchParams, detect_touches, strong_touch_frames
from mtah.spells.postprocess import PostParams, postprocess


@dataclass
class PipelineConfig:
    ball: BallCtrlParams = field(default_factory=BallCtrlParams)
    tag: TagPriorParams = field(default_factory=lambda: TagPriorParams(code_table=load_tag_codes()))
    touch: TouchParams = field(default_factory=TouchParams)
    decoder: DecoderParams = field(default_factory=lambda: DecoderParams(weights={"ball": 1.0, "tag": 1.0, "dead": 1.0, "pathcrf": 1.0},
                                                                         lengths={"ball": 1.0, "tag": 1.0, "dead": 1.0, "pathcrf": 1.0}))
    post: PostParams = field(default_factory=PostParams)
    dead_model_path: Optional[str] = None
    dead_policy: str = "full"     # training-time feature policy of the dead model (recorded for reproducibility)
    use_pathcrf: bool = False     # external PathCRF team-posterior stream (needs a cache key per half/match)
    pathcrf: PathCRFParams = field(default_factory=PathCRFParams)

    def to_dict(self):
        d = asdict(self)
        d["pathcrf"] = {k: d["pathcrf"][k] for k in ("ckpt", "eps", "conf_alpha")}   # runtime paths stay out of configs
        return d

    @classmethod
    def from_dict(cls, d):
        tag = dict(d["tag"])
        if tag.get("code_table") is None:          # configs saved before the per-code table existed
            from mtah.evidence.tag_prior import load_tag_codes
            tag["code_table"] = load_tag_codes()
        return cls(ball=BallCtrlParams(**d["ball"]), tag=TagPriorParams(**tag), touch=TouchParams(**d["touch"]),
                   decoder=DecoderParams(**d["decoder"]), post=PostParams(**d["post"]), dead_model_path=d.get("dead_model_path"),
                   dead_policy=d.get("dead_policy", "full"), use_pathcrf=d.get("use_pathcrf", False),
                   pathcrf=PathCRFParams(**{k: v for k, v in d.get("pathcrf", {}).items() if k in ("ckpt", "eps", "conf_alpha")}))


def compute_evidence(match, tags, cfg, dead_model, cache_key=None):
    """Evidence streams by name. `cache_key` names the PathCRF cache entry (required when cfg.use_pathcrf)."""
    ev = {"ball": ball_ctrl_evidence(match, cfg.ball), "tag": tag_prior_evidence(tags, match.n_frames, cfg.tag)}
    if dead_model is not None:
        if dead_model.policy != cfg.dead_policy:
            raise ValueError(f"dead model policy {dead_model.policy!r} != cfg.dead_policy {cfg.dead_policy!r}")
        ev["dead"] = dead_evidence(dead_model.predict_proba(dead_features(match, tags)))
    if cfg.use_pathcrf:
        if cache_key is None:
            raise ValueError("cfg.use_pathcrf needs a cache_key")
        r = pathcrf_posterior(match, cache_key, cfg.pathcrf)
        ev["pathcrf"] = pathcrf_evidence(r["p_home"], r["imputed_frac"], cfg.pathcrf.eps, cfg.pathcrf.conf_alpha)
    return ev


def compute_strong(match, cfg):
    t = detect_touches(match, cfg.touch)
    return (strong_touch_frames(t, 0, cfg.touch, match.n_frames),
            strong_touch_frames(t, 1, cfg.touch, match.n_frames), t)


def predict(match, tags, cfg, dead_model, evidence=None, strong=None, cache_key=None):
    ev = evidence if evidence is not None else compute_evidence(match, tags, cfg, dead_model, cache_key)
    lab = decode(ev, match.half, cfg.decoder)
    st = strong if strong is not None else compute_strong(match, cfg)
    sH, sA = st[0], st[1]
    touches = st[2] if len(st) > 2 else detect_touches(match, cfg.touch)   # tolerate the old 2-tuple
    return postprocess(lab, match.half, match.fps, cfg.post, strong_H=sH, strong_A=sA, touches=touches)
