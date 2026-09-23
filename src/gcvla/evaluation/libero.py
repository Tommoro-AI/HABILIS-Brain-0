from __future__ import annotations

from collections.abc import Callable, Sequence

import gymnasium as gym

NEW_ROLLOUT_OPTION = "lerobot_new_rollout"


class FreezeAfterEpisodeEnd(gym.Wrapper):
    """Replay a terminal transition until an explicit new rollout starts."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._frozen: tuple | None = None

    def reset(self, *, seed=None, options=None):
        if self._frozen is not None and not (options or {}).get(NEW_ROLLOUT_OPTION):
            observation, _, _, _, info = self._frozen
            return observation, info
        self._frozen = None
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self._frozen is not None:
            return self._frozen
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            self._frozen = (observation, 0.0, terminated, truncated, info)
        return observation, reward, terminated, truncated, info


def _freeze_factory(env_fn: Callable[[], gym.Env]) -> Callable[[], gym.Env]:
    def make_env() -> gym.Env:
        return FreezeAfterEpisodeEnd(env_fn())

    return make_env


def make_libero_vector_env(
    env_fns: Sequence[Callable[[], gym.Env]],
) -> gym.vector.SyncVectorEnv:
    """Create the LIBERO vector environment with the upstream LeRobot reset contract."""
    return gym.vector.SyncVectorEnv(
        [_freeze_factory(env_fn) for env_fn in env_fns],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )


def reset_libero_rollout(vector_env: gym.vector.VectorEnv, *, seed=None):
    """Start a real rollout and thaw sub-environments frozen by the prior rollout."""
    return vector_env.reset(seed=seed, options={NEW_ROLLOUT_OPTION: True})
