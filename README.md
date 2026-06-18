# cd-sarl

Arena wrapper for **CD-SARL** — plain **SARL** (Socially Attentive RL, crowd-robot
attention DRL) trained with **c**urriculum learning and **d**iverse pedestrian
dynamics. Checkpoint from [RAISE-Lab/soc-nav-training](https://github.com/RAISE-Lab/soc-nav-training)
(`CrowdNav/crowd_nav/data/sarl_curr_div/rl_model.pth`).

The value-network architecture is identical to the `crowdnav` planner; this
adapter vendors the same SARL policy and only differs in the shipped checkpoint
and its training config (`configs/policy.config`, copied from `sarl_curr_div`).

## Run

```sh
arena launch mobile:=drl mobile.planner:=cd-sarl
```

Requires a global plan. Defaults to `nav2/navfn`.

## Config / checkpoint

- `kinematics = holonomic` -> holonomic actions, `action_type: omnidirectional`.
- `with_om = false`, `with_global_state = true` -> value-net input width 13
  (joint state, no occupancy-map channels). Verified: `rl_model.pth` loads
  `strict=True` into the SARL net (0 missing / 0 unexpected keys);
  `mlp1.0.weight` is `(150, 13)`.
- `rl_model.pth` is the trained RL policy (`il_model.pth` is the imitation
  pretrain and is not shipped).

## Files

- `planner.py`: entry point. Builds SARL `JointState`, runs `policy.predict`.
- `policy.py`, `state.py`: vendored SARL (shared verbatim with crowdnav).
- `configs/policy.config`: sarl_curr_div training config.
- `planner.yaml`: observation manifest.
- `model/rl_model.pth`: trained RL checkpoint.
- See [ATTRIBUTION.md](ATTRIBUTION.md) for code/checkpoint provenance.

## License

MIT (matches upstream).
