"""Recording extension for the StackCubeObstacle task.

The official :class:`~mani_skill.utils.wrappers.record.RecordEpisode` only
persists ``success``/``fail`` for each episode. Evaluation of this task also
produces the separate component flags ``stack_success``, ``obstacle_collision``
and ``obstacle_crossing``. This subclass keeps those components in the original
JSON episode metadata (including failed episodes) and writes the deterministic
obstacle geometry configuration into the original ``env_info``. It reuses the
official recorder end to end and adds no separate report or sidecar file.
"""

from typing import Optional

from mani_skill.utils import common
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.wrappers.record import RecordEpisode

VALIDITY_KEYS = ("stack_success", "obstacle_collision", "obstacle_crossing")


class ObstacleRecordEpisode(RecordEpisode):
    """RecordEpisode that preserves StackCubeObstacle validity components."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_info = None
        self._episode_obstacle_config = None
        self._refresh_obstacle_config()

    def _refresh_obstacle_config(self):
        """Write the current (possibly reconfigured) obstacle geometry to JSON.

        The collector samples the real radius per episode and rebuilds the scene
        via ``reset(options={"reconfigure": True, ...})``; this keeps the single
        ``env_info["obstacle_config"]`` entry consistent with the episode that is
        eventually flushed.
        """
        if not self.save_trajectory:
            return
        env_info = self._json_data.get("env_info")
        obstacle_config = self._current_obstacle_config()
        if env_info is None or obstacle_config is None:
            return
        env_info["obstacle_config"] = obstacle_config
        dump_json(self._json_path, self._json_data, indent=2)

    def reset(self, *args, **kwargs):
        # ``super().reset`` flushes the previous trajectory before the new
        # episode is configured, so the config captured for that episode below
        # still belongs to it when ``flush_trajectory`` runs.
        obs, info = super().reset(*args, **kwargs)
        # Refresh on every reset, not only on reconfigure: a reset that only
        # moves the obstacle centre (same radius) must not leave the env-level
        # config pointing at the previous episode's location.
        self._refresh_obstacle_config()
        self._last_info = None
        self._episode_obstacle_config = self._current_obstacle_config()
        return obs, info

    def _current_obstacle_config(self) -> Optional[dict]:
        obstacle_config = getattr(self.base_env, "obstacle_config", None)
        if not callable(obstacle_config):
            return None
        return obstacle_config()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # The flush triggered by the next reset happens before that reset's info
        # is available, so keep the latest step info to summarise the episode.
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def _episode_validity(self) -> dict:
        if self._last_info is None:
            return {}
        validity = {}
        for key in VALIDITY_KEYS:
            if key in self._last_info:
                validity[key] = bool(
                    common.to_numpy(self._last_info[key]).reshape(-1)[0]
                )
        return validity

    def flush_trajectory(self, *args, **kwargs):
        if not self.save_trajectory:
            return super().flush_trajectory(*args, **kwargs)
        num_episodes_before = len(self._json_data["episodes"])
        super().flush_trajectory(*args, **kwargs)
        appended = self._json_data["episodes"][num_episodes_before:]
        if appended:
            validity = self._episode_validity()
            for episode in appended:
                episode.update(validity)
                if self._episode_obstacle_config is not None:
                    episode["obstacle_config"] = self._episode_obstacle_config
            dump_json(self._json_path, self._json_data, indent=2)
