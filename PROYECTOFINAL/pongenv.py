from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from controllers import HeuristicBotController
from pong_core import PongCore, Theme, mirror_perspective_action


class PongEnv(gym.Env):
    """
    Wrapper de Gymnasium para entrenar UN agente a la vez.

    El agente siempre se entrena como jugador izquierdo.
    El oponente suele ser un bot heurístico congelado.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        theme: Theme | None = None,
        opponent_controller=None,
        **core_kwargs,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.core = PongCore(theme=theme, **core_kwargs)
        self.opponent_controller = opponent_controller or HeuristicBotController()

        # Misma representación vectorial que venías usando,
        # pero ahora unificada con PongCore.
        self.action_space = spaces.Discrete(9)
        low = np.array([0, 0, -1, -1, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _obs(self):
        return np.array(self.core.get_obs_left(), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Empata el RNG del motor con el RNG de Gymnasium.
        # Así el saque inicial y la velocidad vertical sí dependen del seed del entorno.
        self.core.set_rng(self.np_random)
        self.core.reset_match()

        if self.render_mode == "human":
            self.core.render_to_window()
        return self._obs(), self.core.info()

    def step(self, action):
        opponent_obs = self.core.get_obs_right()
        opponent_action_perspective = self.opponent_controller.act(opponent_obs)
        opponent_action_screen = mirror_perspective_action(opponent_action_perspective)

        reward_left, reward_right, terminated, info = self.core.step(int(action), opponent_action_screen)
        if self.render_mode == "human":
            self.core.render_to_window()
        truncated = False
        return self._obs(), float(reward_left), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.core.render_to_window()

    def close(self):
        self.core.close()
