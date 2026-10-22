"""Evidence -> decoder -> post-processing: one function from inputs to labels."""
from dataclasses import asdict, dataclass, field
from typing import Optional

from mtah.decode.viterbi import DecoderParams, decode
from mtah.evidence.ball_ctrl import BallCtrlParams, ball_ctrl_evidence
from mtah.evidence.dead import dead_evidence, dead_table
from mtah.evidence.pathcrf import PathCRFParams, pathcrf_evidence, pathcrf_posterior
from mtah.evidence.tag_prior import TagPriorParams, load_tag_codes, tag_prior_evidence
from mtah.evidence.switch import hazard_to_tau, switch_evidence, switch_features
from mtah.evidence.team_ctrl import team_ctrl_evidence, team_features
from mtah.evidence.touches import TouchParams, detect_touches, strong_touch_frames
from mtah.spells.merge import apply_merge_model
from mtah.spells.postprocess import PostParams, postprocess


@dataclass
class PipelineConfig:
    ball: BallCtrlParams = field(default_factory=BallCtrlParams)
    tag: TagPriorParams = field(default_factory=lambda: TagPriorParams(code_table=load_tag_codes()))
    touch: TouchParams = field(default_factory=TouchParams)
    decoder: DecoderParams = field(default_factory=lambda: DecoderParams(weights={"ball": 1.0, "tag": 1.0, "dead": 1.0, "pathcrf": 1.0, "teamctrl": 1.0},
                                                                         lengths={"ball": 1.0, "tag": 1.0, "dead": 1.0, "pathcrf": 1.0, "teamctrl": 1.0}))
    post: PostParams = field(default_factory=PostParams)
    dead_model_path: Optional[str] = None
    dead_offset: float = 0.0      # log-odds shift of the dead probability before dead_evidence (prior-shift correction)
    dead_policy: str = "full"     # training-time feature policy of the dead model (recorded for reproducibility)
    use_pathcrf: bool = False     # external PathCRF team-posterior stream (needs a cache key per half/match)
    pathcrf: PathCRFParams = field(default_factory=PathCRFParams)
    use_teamctrl: bool = False    # learned team-in-control stream (needs a TeamCtrlModel passed to compute_evidence)
    team_policy: str = "full"     # training-time feature policy of the team-control model
    use_switch: bool = False      # switch hazard (needs a SwitchModel passed to compute_evidence); not a cost stream
    switch_policy: str = "full"   # training-time feature policy of the switch model

    def to_dict(self):
        d = asdict(self)
        d["pathcrf"] = {k: d["pathcrf"][k] for k in ("ckpt", "eps", "conf_alpha", "none_k")}   # runtime paths stay out of configs
        return d

    @classmethod
    def from_dict(cls, d):
        tag = dict(d["tag"])
        if tag.get("code_table") is None:          # configs saved before the per-code table existed
            from mtah.evidence.tag_prior import load_tag_codes
            tag["code_table"] = load_tag_codes()
        return cls(ball=BallCtrlParams(**d["ball"]), tag=TagPriorParams(**tag), touch=TouchParams(**d["touch"]),
                   decoder=DecoderParams(**d["decoder"]), post=PostParams(**d["post"]), dead_model_path=d.get("dead_model_path"),
                   dead_policy=d.get("dead_policy", "full"), dead_offset=float(d.get("dead_offset", 0.0)), use_pathcrf=d.get("use_pathcrf", False),
                   pathcrf=PathCRFParams(**{k: v for k, v in d.get("pathcrf", {}).items() if k in ("ckpt", "eps", "conf_alpha", "none_k")}),
                   use_teamctrl=d.get("use_teamctrl", False), team_policy=d.get("team_policy", "full"),
                   use_switch=d.get("use_switch", False), switch_policy=d.get("switch_policy", "full"))


def compute_evidence(match, tags, cfg, dead_model, cache_key=None, team_model=None, switch_model=None):
    """Evidence streams by name. `cache_key` names the PathCRF cache entry (required when cfg.use_pathcrf).
    "teamctrl" is added iff cfg.use_teamctrl and a `team_model` is given (its policy must equal cfg.team_policy);
    cfg.use_teamctrl with a positive teamctrl weight and no `team_model` raises (the stream would silently vanish).
    "switch" (the hazard, NOT centred and NOT a cost stream: column 0 of an (F, 3) array, NaN elsewhere) is added
    iff cfg.use_switch and a `switch_model` is given (policy must equal cfg.switch_policy); its teamctrl-margin
    features use the "teamctrl" stream computed here (required if the model was trained with it)."""
    ev = {"ball": ball_ctrl_evidence(match, cfg.ball), "tag": tag_prior_evidence(tags, match.n_frames, cfg.tag, fps=match.fps)}
    if dead_model is not None:
        if dead_model.policy != cfg.dead_policy:
            raise ValueError(f"dead model policy {dead_model.policy!r} != cfg.dead_policy {cfg.dead_policy!r}")
        ev["dead"] = dead_evidence(dead_model.predict_proba(dead_table(match, tags, dead_model.policy)),
                              logit_offset=cfg.dead_offset)  # base or base+long
    if cfg.use_pathcrf:
        if cache_key is None:
            raise ValueError("cfg.use_pathcrf needs a cache_key")
        r = pathcrf_posterior(match, cache_key, cfg.pathcrf)
        ev["pathcrf"] = pathcrf_evidence(r["p_home"], r["imputed_frac"], cfg.pathcrf.eps, cfg.pathcrf.conf_alpha,
                                         p_none=r.get("p_none"), none_k=cfg.pathcrf.none_k)
    if cfg.use_teamctrl and team_model is None and cfg.decoder.weights.get("teamctrl", 1.0) > 0:
        raise ValueError(f"cfg.use_teamctrl with teamctrl weight {cfg.decoder.weights.get('teamctrl', 1.0)} > 0 "
                         "needs a team_model")
    if cfg.use_teamctrl and team_model is not None:
        if team_model.policy != cfg.team_policy:
            raise ValueError(f"team model policy {team_model.policy!r} != cfg.team_policy {cfg.team_policy!r}")
        ev["teamctrl"] = team_ctrl_evidence(team_model.predict_proba(team_features(match, cfg.touch)))
    if cfg.use_switch and switch_model is not None:
        if switch_model.policy != cfg.switch_policy:
            raise ValueError(f"switch model policy {switch_model.policy!r} != cfg.switch_policy {cfg.switch_policy!r}")
        if switch_model.with_teamctrl and "teamctrl" not in ev:
            raise ValueError("switch model was trained with teamctrl features but no teamctrl stream is available")
        ts = ev.get("teamctrl") if switch_model.with_teamctrl else None
        ev["switch"] = switch_evidence(switch_model.predict_hazard(switch_features(match, ts, cfg.touch)))
    return ev


def compute_strong(match, cfg):
    t = detect_touches(match, cfg.touch)
    return (strong_touch_frames(t, 0, cfg.touch, match.n_frames),
            strong_touch_frames(t, 1, cfg.touch, match.n_frames), t)


def predict(match, tags, cfg, dead_model, evidence=None, strong=None, cache_key=None, team_model=None, merge_model=None,
            switch_model=None):
    """Labels. With cfg.post.merge_theta < 1 the per-spell merge model runs after post-processing (a `merge_model`
    is then required); merge_theta = 1.0 (default) leaves the post-processed labels untouched.
    With cfg.decoder.w_switch > 0 the "switch" hazard sets a per-frame tau_HA (required then); w_switch = 0 decodes
    exactly as before. cfg.post.snap_mode = "hazard" snaps H<->A boundaries to the hazard peak."""
    ev = evidence if evidence is not None else compute_evidence(match, tags, cfg, dead_model, cache_key, team_model=team_model,
                                                                switch_model=switch_model)
    hazard = ev["switch"][:, 0] if "switch" in ev else None
    tau_t = None
    if cfg.decoder.w_switch > 0:
        if hazard is None:
            raise ValueError(f"cfg.decoder.w_switch = {cfg.decoder.w_switch} > 0 needs a 'switch' hazard in the evidence")
        tau_t = hazard_to_tau(hazard, cfg.decoder.tau_HA, cfg.decoder.w_switch, cfg.decoder.tau_floor)
    lab = decode(ev, match.half, cfg.decoder, tau_HA_t=tau_t)
    st = strong if strong is not None else compute_strong(match, cfg)
    sH, sA = st[0], st[1]
    touches = st[2] if len(st) > 2 else detect_touches(match, cfg.touch)   # tolerate the old 2-tuple
    out = postprocess(lab, match.half, match.fps, cfg.post, strong_H=sH, strong_A=sA, touches=touches, hazard=hazard)
    theta = float(getattr(cfg.post, "merge_theta", 1.0))
    if theta < 1.0:
        if merge_model is None:
            raise ValueError(f"cfg.post.merge_theta = {theta} < 1 needs a merge_model")
        out = apply_merge_model(out, match, ev, (sH, sA, touches), merge_model, theta)
    return out
