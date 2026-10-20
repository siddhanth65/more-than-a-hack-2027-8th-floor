"""Evidence -> decoder -> post-processing: one function from inputs to labels."""
from dataclasses import asdict, dataclass, field
from typing import Optional

from mtah.decode.viterbi import DecoderParams, decode
from mtah.evidence.ball_ctrl import BallCtrlParams, ball_ctrl_evidence
from mtah.evidence.dead import dead_evidence, dead_features
from mtah.evidence.tag_prior import TagPriorParams, load_tag_codes, tag_prior_evidence
from mtah.evidence.touches import TouchParams, detect_touches, strong_touch_frames
from mtah.spells.postprocess import PostParams, postprocess


@dataclass
class PipelineConfig:
    ball: BallCtrlParams = field(default_factory=BallCtrlParams)
    tag: TagPriorParams = field(default_factory=lambda: TagPriorParams(code_table=load_tag_codes()))
    touch: TouchParams = field(default_factory=TouchParams)
    decoder: DecoderParams = field(default_factory=lambda: DecoderParams(weights={"ball": 1.0, "tag": 1.0, "dead": 1.0},
                                                                         lengths={"ball": 1.0, "tag": 1.0, "dead": 1.0}))
    post: PostParams = field(default_factory=PostParams)
    dead_model_path: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(ball=BallCtrlParams(**d["ball"]), tag=TagPriorParams(**d["tag"]), touch=TouchParams(**d["touch"]),
                   decoder=DecoderParams(**d["decoder"]), post=PostParams(**d["post"]), dead_model_path=d.get("dead_model_path"))


def compute_evidence(match, tags, cfg, dead_model):
    ev = {"ball": ball_ctrl_evidence(match, cfg.ball), "tag": tag_prior_evidence(tags, match.n_frames, cfg.tag)}
    if dead_model is not None:
        ev["dead"] = dead_evidence(dead_model.predict_proba(dead_features(match, tags)))
    return ev


def compute_strong(match, cfg):
    t = detect_touches(match, cfg.touch)
    return (strong_touch_frames(t, 0, cfg.touch, match.n_frames),
            strong_touch_frames(t, 1, cfg.touch, match.n_frames))


def predict(match, tags, cfg, dead_model, evidence=None, strong=None):
    ev = evidence if evidence is not None else compute_evidence(match, tags, cfg, dead_model)
    lab = decode(ev, match.half, cfg.decoder)
    sH, sA = strong if strong is not None else compute_strong(match, cfg)
    return postprocess(lab, match.half, match.fps, cfg.post, strong_H=sH, strong_A=sA)
