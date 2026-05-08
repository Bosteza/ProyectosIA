from __future__ import annotations

from pathlib import Path
from typing import Type

import numpy as np
import pygame
from stable_baselines3 import DQN, PPO

from pong_core import (
    ACTION_STAY,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP_LEFT,
    ACTION_UP_RIGHT,
    ACTION_DOWN_LEFT,
    ACTION_DOWN_RIGHT,
)


def _combine(dx: int, dy: int) -> int:
    if dx == 0 and dy == 0:
        return ACTION_STAY
    if dx == 0 and dy < 0:
        return ACTION_UP
    if dx == 0 and dy > 0:
        return ACTION_DOWN
    if dx < 0 and dy == 0:
        return ACTION_LEFT
    if dx > 0 and dy == 0:
        return ACTION_RIGHT
    if dx < 0 and dy < 0:
        return ACTION_UP_LEFT
    if dx > 0 and dy < 0:
        return ACTION_UP_RIGHT
    if dx < 0 and dy > 0:
        return ACTION_DOWN_LEFT
    return ACTION_DOWN_RIGHT


class BaseController:
    """
    action_mode:
    - "screen": la acción ya viene en coordenadas de pantalla
    - "perspective": la acción viene como si el jugador estuviera a la izquierda
    """
    action_mode = "perspective"

    def act(self, obs):
        raise NotImplementedError


class HumanController(BaseController):
    action_mode = "screen"

    def __init__(self, up_key=pygame.K_UP, down_key=pygame.K_DOWN, left_key=pygame.K_LEFT, right_key=pygame.K_RIGHT):
        self.up_key = up_key
        self.down_key = down_key
        self.left_key = left_key
        self.right_key = right_key

    def act(self, obs=None):
        keys = pygame.key.get_pressed()
        dx = 0
        dy = 0
        if keys[self.up_key]:
            dy -= 1
        if keys[self.down_key]:
            dy += 1
        if keys[self.left_key]:
            dx -= 1
        if keys[self.right_key]:
            dx += 1
        return _combine(dx, dy)


class HeuristicBotController(BaseController):
    action_mode = "perspective"

    def __init__(
        self,
        deadzone_y=0.03,
        deadzone_x=0.03,
        return_x=0.03,
        paddle_half_height_norm=0.085,
    ):
        self.deadzone_y = deadzone_y
        self.deadzone_x = deadzone_x
        self.return_x = return_x
        self.paddle_half_height_norm = paddle_half_height_norm

    def act(self, obs):
        ball_x, ball_y, ball_vx, ball_vy, self_x, self_y, opp_y = obs

        # El bot trabaja en observaciones normalizadas.
        # paddle_half_height_norm evita el hardcode 0.17/2 y lo alinea con el entorno.
        paddle_center_y = self_y + self.paddle_half_height_norm
        dx = 0
        dy = 0

        if ball_y < paddle_center_y - self.deadzone_y:
            dy = -1
        elif ball_y > paddle_center_y + self.deadzone_y:
            dy = 1

        # Si la pelota viene hacia mí o ya está en mi mitad, la persigo en X.
        if ball_vx < 0 or ball_x < 0.55:
            if ball_x < self_x - self.deadzone_x:
                dx = -1
            elif ball_x > self_x + self.deadzone_x:
                dx = 1
        else:
            # Si la pelota se fue al otro lado, regreso a mi posición base.
            if self_x > self.return_x + self.deadzone_x:
                dx = -1
            elif self_x < self.return_x - self.deadzone_x:
                dx = 1

        return _combine(dx, dy)


class SB3Controller(BaseController):
    action_mode = "perspective"

    def __init__(self, algo_cls: Type, model_path: str):
        self.algo_cls = algo_cls
        self.model_path = model_path
        if not Path(model_path).with_suffix(".zip").exists():
            raise FileNotFoundError(f"No existe el modelo: {model_path}.zip")
        self.model = algo_cls.load(model_path)

    def act(self, obs):
        action, _ = self.model.predict(np.array(obs, dtype=np.float32), deterministic=True)
        return int(action)


def make_dqn_controller(model_path: str) -> SB3Controller:
    return SB3Controller(DQN, model_path)


def make_ppo_controller(model_path: str) -> SB3Controller:
    return SB3Controller(PPO, model_path)
