# Modifications by Tommoro: GC-VLA/GCRF runtime integration.
# Public inference redistribution removes inactive research capture and state override utilities.

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.envs.utils import freeze_after_episode_end
from lerobot.processor import RobotObservation



def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _get_suite(name: str) -> benchmark.Benchmark:
    """Instantiate a LIBERO suite by name with clear validation."""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """Validate/normalize task ids. If None → all tasks."""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task_suite.tasks[i].problem_folder
        / task_suite.tasks[i].init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
    return init_states


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        episode_length: int | None = None,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        control_mode: str = "relative",
    ):
        super().__init__()
        self.task_id = task_id
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = _parse_camera_names(
            camera_name
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)

        # Map raw camera names to "image1" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index
        self.episode_length = episode_length
        self._trace_path = os.environ.get("MOLMOACT2_ENV_TRACE_PATH", "").strip()
        self._trace_episode = 0
        self._trace_step = 0
        self._active_init_state_id: int | None = None
        # Load once and keep
        self._init_states = get_task_init_states(task_suite, self.task_id) if self.init_states else None
        self._reset_stride = n_envs  # when performing a reset, append `_reset_stride` to `init_state_id`.

        self.init_state_id = self.episode_index

        self._env = self._make_envs_task(task_suite, self.task_id)
        default_steps = 500
        self._max_episode_steps = (
            TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)
            if self.episode_length is None
            else self.episode_length
        )
        self.control_mode = control_mode
        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "robot_state": spaces.Dict(
                        {
                            "eef": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                    "quat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                    ),
                                    "mat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                    ),
                                }
                            ),
                            "gripper": spaces.Dict(
                                {
                                    "qpos": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                    "qvel": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                }
                            ),
                            "joints": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                    "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                }
                            ),
                        }
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def render(self):
        raw_obs = self._env.env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["image"]
        image = image[::-1, ::-1]  # flip both H and W for visualization
        return image

    def _make_envs_task(self, task_suite: Any, task_id: int = 0):
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
            "camera_depths": False,
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env

    def _format_raw_obs(self, raw_obs: RobotObservation) -> RobotObservation:
        images = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            images[self.camera_name_mapping[camera_name]] = image

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")

        # rotation matrix from controller
        eef_mat = self._env.robots[0].controller.ee_ori_mat if eef_pos is not None else None
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        joint_pos = raw_obs.get("robot0_joint_pos")
        joint_vel = raw_obs.get("robot0_joint_vel")
        obs = {
            "pixels": images,
            "robot_state": {
                "eef": {
                    "pos": eef_pos,  # (3,)
                    "quat": eef_quat,  # (4,)
                    "mat": eef_mat,  # (3, 3)
                },
                "gripper": {
                    "qpos": gripper_qpos,  # (2,)
                    "qvel": gripper_qvel,  # (2,)
                },
                "joints": {
                    "pos": joint_pos,  # (7,)
                    "vel": joint_vel,  # (7,)
                },
            },
        }
        if self.obs_type == "pixels":
            return {"pixels": images.copy()}

        if self.obs_type == "pixels_agent_pos":
            # Validate required fields are present
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                raise ValueError(
                    f"Missing required robot state fields in raw observation. "
                    f"Got eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                    f"gripper_qpos={gripper_qpos is not None}"
                )
            return obs

        raise NotImplementedError(
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
            "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
        )

    @staticmethod
    def _trace_array(value: Any) -> list[float] | None:
        if value is None:
            return None
        try:
            return np.asarray(value, dtype=np.float32).reshape(-1).tolist()
        except (TypeError, ValueError):
            return None

    def _compact_trace_state(self, raw_obs: Mapping[str, Any] | None) -> dict[str, Any]:
        """Return a compact, JSON-safe state snapshot for failure diagnostics."""
        raw_obs = raw_obs or {}
        robot = {
            key: self._trace_array(raw_obs.get(key))
            for key in (
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "robot0_gripper_qvel",
                "robot0_joint_pos",
                "robot0_joint_vel",
            )
        }
        result: dict[str, Any] = {"robot": robot}
        sim = getattr(self._env, "sim", None)
        if sim is None:
            inner = getattr(self._env, "env", None)
            sim = getattr(inner, "sim", None)
        if sim is None:
            return result
        data = getattr(sim, "data", None)
        if data is None:
            return result
        for key in ("qpos", "qvel", "body_xpos", "body_xquat"):
            array = self._trace_array(getattr(data, key, None))
            if array is not None:
                result[key] = array
        ncon = int(getattr(data, "ncon", 0))
        contacts = []
        for index in range(min(ncon, 64)):
            contact = data.contact[index]
            contacts.append([
                int(getattr(contact, "geom1", -1)),
                int(getattr(contact, "geom2", -1)),
                float(getattr(contact, "dist", 0.0)),
            ])
        result["ncon"] = ncon
        result["contacts"] = contacts
        return result

    def _write_trace_metadata(self) -> None:
        if not self._trace_path:
            return
        path = Path(self._trace_path)
        meta = path.with_suffix(".meta.json")
        if meta.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(json.dumps({
            "schema_version": 2,
            "suite": self.task,
            "task_id": int(self.task_id),
            "episode_index": int(self.episode_index),
            "initial_state_id": self._active_init_state_id,
            "control_mode": self.control_mode,
            "action_dim": ACTION_DIM,
            "max_episode_steps": int(self._max_episode_steps),
            "fields": [
                "action_physical", "state_before.robot", "state_after.robot",
                "state_after.qpos", "state_after.qvel", "state_after.body_xpos",
                "state_after.body_xquat", "state_after.contacts", "reward",
                "done", "is_success",
            ],
        }, indent=2) + "\n")

    def _write_trace_row(self, action, raw_before, raw_after, reward, done, is_success, sim_state_before=None) -> None:
        if not self._trace_path:
            return
        path = Path(self._trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_trace_metadata()
        row = {
            "schema_version": 2,
            "episode_index": int(self.episode_index),
            "initial_state_id": self._active_init_state_id,
            "trace_episode": int(self._trace_episode),
            "env_step": int(self._trace_step),
            "task_id": int(self.task_id),
            "task": self.task,
            "action_physical": np.asarray(action, dtype=np.float32).reshape(-1).tolist(),
            "state_before": self._compact_trace_state(raw_before),
            "state_after": self._compact_trace_state(raw_after),
            "sim_state_before": None if sim_state_before is None else np.asarray(sim_state_before, dtype=np.float64).reshape(-1).tolist(),
            "reward": float(reward),
            "done": bool(done),
            "is_success": bool(is_success),
        }
        with path.open("a") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._trace_step += 1

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._env.seed(seed)
        raw_obs = self._env.reset()
        if self.init_states and self._init_states is not None:
            seed_base_raw = os.environ.get("LIBERO_INIT_STATE_FROM_SEED_BASE")
            if seed_base_raw is not None and seed is not None:
                seed_base = int(seed_base_raw)
                init_state_from_seed = int(seed) - seed_base
                if init_state_from_seed < 0:
                    raise ValueError(
                        "LIBERO init-state seed mapping produced a negative index: "
                        f"seed={seed}, base={seed_base}."
                    )
                # eval_policy seeds each explicit rollout batch consecutively.
                # Re-anchoring here prevents auto-resets from changing which
                # official fixed init state is used by the next batch.
                self.init_state_id = init_state_from_seed
            selected_init_state_id = self.init_state_id % len(self._init_states)
            self._active_init_state_id = int(selected_init_state_id)
            raw_obs = self._env.set_init_state(self._init_states[selected_init_state_id])
            self.init_state_id += self._reset_stride  # Change init_state_id when reset

        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        # Increasing this value can improve determinism and reproducibility across resets.
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        observation = self._format_raw_obs(raw_obs)
        self._trace_episode += 1
        self._trace_step = 0
        self._write_trace_metadata()
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_before = self._env.env._get_observations()
        sim_state_before = np.asarray(self._env.sim.get_state().flatten(), dtype=np.float64).copy()
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        self._write_trace_row(action, raw_before, raw_obs, reward, done, is_success, sim_state_before=sim_state_before)
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        # Return the terminal observation unchanged. SyncVectorEnv owns the
        # reset on the following step through NEXT_STEP autoreset.
        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
) -> list[Callable[[], LiberoEnv]]:
    """Build n_envs factory callables for a single (suite, task_id)."""

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            n_envs=n_envs,
            control_mode=control_mode,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized LIBERO environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_ids` (list[int]) inside `gym_kwargs` to restrict tasks per suite.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks

    camera_names = _parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    print(
        f"Creating LIBERO envs | suites={suite_names} | n_envs(per task)={n_envs} | init_states={init_states}"
    )
    if task_ids_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter)
        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                episode_length=episode_length,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
            )
            if env_cls is gym.vector.SyncVectorEnv:
                out[suite_name][tid] = gym.vector.SyncVectorEnv(
                    [freeze_after_episode_end(fn) for fn in fns],
                    autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
                )
            else:
                out[suite_name][tid] = env_cls(fns)
            print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    # return plain dicts for predictability
    return {suite: dict(task_map) for suite, task_map in out.items()}
