"""CD-SARL (curriculum + diverse-dynamics SARL) wrapper for the arena_planners bridge.

Plain SARL (Crowd-robot attention DRL) trained with curriculum learning and
diverse pedestrian dynamics in RAISE-Lab/soc-nav-training. The value-network
architecture is identical to the crowdnav planner; this adapter only differs in
the checkpoint shipped (sarl_curr_div/rl_model.pth) and its config.
"""

from __future__ import annotations

import configparser
import pathlib

import numpy as np
from arena_planners.geometry import lookahead_on_path
from arena_planners.sdk import load_manifest, main_loop
from policy import SARLPolicy
from state import FullState, JointState, ObservableState

_V_PREF: float = 1.0
_RADIUS: float = 0.3
_TIME_STEP: float = 0.25
_LOOKAHEAD: float = 2.0

_policy: SARLPolicy | None = None


def _build_policy() -> SARLPolicy:
    config_path = pathlib.Path(__file__).parent / "configs" / "policy.config"
    config = configparser.RawConfigParser()
    config.read(str(config_path))
    p = SARLPolicy()
    p.configure(config)
    weights = pathlib.Path(__file__).parent / "model" / "rl_model.pth"
    if weights.exists():
        import torch

        p.model.load_state_dict(torch.load(str(weights), map_location="cpu"), strict=True)
    p.set_device("cpu")
    p.set_phase("test")
    p.time_step = _TIME_STEP
    p.query_env = False
    return p


def step(features: dict) -> list[float]:
    """Map features dict to CD-SARL action, return [vx, vy] (holonomic)."""
    global _policy
    if _policy is None:
        _policy = _build_policy()

    robot_pose = features.get("robot_pose")
    robot_state = features.get("robot_state")
    if robot_pose is None or robot_state is None:
        return [0.0, 0.0]
    px, py, theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    vx, vy = float(robot_state[2]), float(robot_state[3])

    global_plan = features.get("global_plan")
    goal_pose = features.get("goal_pose")
    target: tuple[float, float] | None = None
    if global_plan is not None and len(global_plan) > 0:
        target = lookahead_on_path(global_plan, robot_pose, lookahead=_LOOKAHEAD)
    if target is None and goal_pose is not None:
        target = (float(goal_pose[0]), float(goal_pose[1]))
    if target is None:
        return [0.0, 0.0]
    gx, gy = target

    self_state = FullState(px, py, vx, vy, _RADIUS, gx, gy, _V_PREF, theta)

    human_states = []
    peds = features.get("pedestrians")
    if peds is not None:
        for ped in peds:
            human_states.append(
                ObservableState(
                    float(ped[1]),
                    float(ped[2]),
                    float(ped[3]),
                    float(ped[4]),
                    _RADIUS,
                )
            )

    if not human_states:
        dx, dy = gx - px, gy - py
        dist_to_goal = float(np.hypot(dx, dy))
        if dist_to_goal < 1e-6:
            return [0.0, 0.0]
        scale = min(_V_PREF, dist_to_goal) / dist_to_goal
        return [dx * scale, dy * scale]

    joint_state = JointState(self_state, human_states)
    action = _policy.predict(joint_state)
    return [float(action.vx), float(action.vy)]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _policy
    if _policy is not None:
        _policy.action_space = None


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
