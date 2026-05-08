from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np
import pygame


# ============================================================
# ACCIONES
# ============================================================

# Mismo espacio discreto de 9 acciones

ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_UP_LEFT = 5
ACTION_UP_RIGHT = 6
ACTION_DOWN_LEFT = 7
ACTION_DOWN_RIGHT = 8

ACTION_DELTAS = {
    ACTION_STAY: (0, 0),
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
    ACTION_UP_LEFT: (-1, -1),
    ACTION_UP_RIGHT: (1, -1),
    ACTION_DOWN_LEFT: (-1, 1),
    ACTION_DOWN_RIGHT: (1, 1),
}


def mirror_perspective_action(action: int) -> int:
    """
    Convierte una acción pensada desde la perspectiva de la izquierda
    a su equivalente visual en la derecha.

    Esto es CLAVE para poder reutilizar el mismo tipo de política
    entrenada como "jugador izquierdo" y ponerla del lado derecho.
    """
    mapping = {
        ACTION_STAY: ACTION_STAY,
        ACTION_UP: ACTION_UP,
        ACTION_DOWN: ACTION_DOWN,
        ACTION_LEFT: ACTION_RIGHT,
        ACTION_RIGHT: ACTION_LEFT,
        ACTION_UP_LEFT: ACTION_UP_RIGHT,
        ACTION_UP_RIGHT: ACTION_UP_LEFT,
        ACTION_DOWN_LEFT: ACTION_DOWN_RIGHT,
        ACTION_DOWN_RIGHT: ACTION_DOWN_LEFT,
    }
    return mapping[action]


@dataclass
class Theme:
    """
    Todo lo visual se concentra aquí.
    Si quieres cambiar colores, nombres o hacer el juego "más bonito",
    empieza por esta clase.
    """
    background: Tuple[int, int, int] = (14, 18, 32)
    center_line: Tuple[int, int, int] = (52, 64, 92)
    limit_line: Tuple[int, int, int] = (70, 110, 86)
    ball: Tuple[int, int, int] = (245, 245, 245)
    left_paddle: Tuple[int, int, int] = (89, 173, 255)
    right_paddle: Tuple[int, int, int] = (255, 120, 120)
    text: Tuple[int, int, int] = (240, 243, 255)
    subtext: Tuple[int, int, int] = (180, 190, 220)
    button: Tuple[int, int, int] = (40, 52, 82)
    button_hover: Tuple[int, int, int] = (62, 78, 122)
    overlay: Tuple[int, int, int] = (8, 10, 18)
    score_panel: Tuple[int, int, int] = (22, 28, 46)
    pause_button: Tuple[int, int, int] = (70, 84, 125)
    pause_button_hover: Tuple[int, int, int] = (92, 110, 160)
    left_name: str = "DQN"
    right_name: str = "PPO"


class PongCore:
    """
    Motor base del juego.

    Aquí vive la física, las reglas, el marcador y el render.
    No depende de SB3, así que se puede usar tanto para:
    - entrenamiento (vía Gymnasium wrapper)
    - juego real con menú y botones
    - modo espectador IA vs IA
    """

    def __init__(
        self,
        width: int = 900,
        height: int = 640,
        max_score: int = 5,
        paddle_w: int = 14,
        paddle_h: int = 110,
        paddle_speed_y: float = 8.0,
        paddle_speed_x: float = 6.0,
        ball_size: int = 14,
        ball_speed_x: float = 5.0,
        track_ball_reward: float = 0.01,
        track_ball_penalty: float = 0.005,
        hit_reward: float = 0.2,
        score_reward: float = 1.0,
        lose_reward: float = -1.0,
        match_win_bonus: float = 2.0,
        match_lose_penalty: float = -2.0,
        center_margin: int = 24,
        left_x_initial: int = 30,
        right_x_margin: int = 30,
        theme: Theme | None = None,
    ):
        self.width = width
        self.height = height
        self.max_score = max_score

        self.paddle_w = paddle_w
        self.paddle_h = paddle_h
        self.paddle_speed_y = paddle_speed_y
        self.paddle_speed_x = paddle_speed_x

        self.ball_size = ball_size
        self.ball_speed_x = ball_speed_x

        self.track_ball_reward = track_ball_reward
        self.track_ball_penalty = track_ball_penalty
        self.hit_reward = hit_reward
        self.score_reward = score_reward
        self.lose_reward = lose_reward
        self.match_win_bonus = match_win_bonus
        self.match_lose_penalty = match_lose_penalty

        self.center_margin = center_margin
        self.left_x_initial = left_x_initial
        self.right_x_initial = self.width - right_x_margin - self.paddle_w

        # Límites de movimiento horizontal de cada pala.
        self.left_x_min = self.left_x_initial
        self.left_x_max = self.width // 2 - self.center_margin - self.paddle_w

        self.right_x_min = self.width // 2 + self.center_margin
        self.right_x_max = self.right_x_initial

        self.theme = theme or Theme()

        self.window = None
        self.clock = None
        self.running = True

        # RNG propio del motor. Se reemplaza desde PongEnv para que
        # la aleatoriedad dependa del seed del entorno y no del módulo random global.
        self.rng = np.random.default_rng(0)

        self.reset_match()

    # ============================================================
    # ESTADO
    # ============================================================

    def set_rng(self, rng) -> None:
        """
        Permite que el wrapper Gymnasium inyecte el generador aleatorio
        del entorno. Así el saque y la velocidad vertical sí quedan seedados.
        """
        self.rng = rng

    def reset_match(self) -> None:
        self.left_score = 0
        self.right_score = 0

        self.left_x = float(self.left_x_initial)
        self.left_y = float(self.height / 2 - self.paddle_h / 2)

        self.right_x = float(self.right_x_initial)
        self.right_y = float(self.height / 2 - self.paddle_h / 2)

        self.winner = None
        self.reset_ball()

    def reset_ball(self) -> None:
        self.ball_x = float(self.width / 2 - self.ball_size / 2)
        self.ball_y = float(self.height / 2 - self.ball_size / 2)

        self.ball_vx = float(self.rng.choice([-self.ball_speed_x, self.ball_speed_x]))
        self.ball_vy = float(self.rng.uniform(-3.0, 3.0))

    # ============================================================
    # OBSERVACIONES DESDE CADA PERSPECTIVA
    # ============================================================

    def _max_paddle_y(self) -> float:
        return max(1.0, self.height - self.paddle_h)

    def _max_vx(self) -> float:
        return max(1.0, self.ball_speed_x)

    def _max_vy(self) -> float:
        return 6.0

    def get_obs_left(self):
        """
        Observación normalizada desde la perspectiva del jugador izquierdo.
        """
        return [
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_vx / self._max_vx(),
            self.ball_vy / self._max_vy(),
            self.left_x / self.width,
            self.left_y / self._max_paddle_y(),
            self.right_y / self._max_paddle_y(),
        ]

    def get_obs_right(self):
        """
        Observación normalizada pero ESPEJADA, como si el jugador derecho
        estuviera del lado izquierdo.

        Esto permite que un PPO o DQN entrenado "como jugador izquierdo"
        se pueda usar del lado derecho sin reentrenarlo en otro espacio.
        """
        mirrored_ball_x = self.width - self.ball_x - self.ball_size
        mirrored_self_x = self.width - self.right_x - self.paddle_w

        return [
            mirrored_ball_x / self.width,
            self.ball_y / self.height,
            (-self.ball_vx) / self._max_vx(),
            self.ball_vy / self._max_vy(),
            mirrored_self_x / self.width,
            self.right_y / self._max_paddle_y(),
            self.left_y / self._max_paddle_y(),
        ]

    def info(self) -> Dict[str, float]:
        return {
            "left_score": self.left_score,
            "right_score": self.right_score,
            "left_x": self.left_x,
            "left_y": self.left_y,
            "right_x": self.right_x,
            "right_y": self.right_y,
            "winner": self.winner,
        }

    # ============================================================
    # FÍSICA
    # ============================================================

    def _center_dist_to_ball(self, side: str) -> float:
        if side == "left":
            px = self.left_x + self.paddle_w / 2
            py = self.left_y + self.paddle_h / 2
        else:
            px = self.right_x + self.paddle_w / 2
            py = self.right_y + self.paddle_h / 2

        bx = self.ball_x + self.ball_size / 2
        by = self.ball_y + self.ball_size / 2
        return math.sqrt((px - bx) ** 2 + (py - by) ** 2)

    def _apply_action_to_left(self, action: int) -> None:
        dx_sign, dy_sign = ACTION_DELTAS.get(action, (0, 0))
        self.left_x += dx_sign * self.paddle_speed_x
        self.left_y += dy_sign * self.paddle_speed_y
        self.left_x = float(max(self.left_x_min, min(self.left_x, self.left_x_max)))
        self.left_y = float(max(0, min(self.left_y, self.height - self.paddle_h)))

    def _apply_action_to_right(self, action: int) -> None:
        dx_sign, dy_sign = ACTION_DELTAS.get(action, (0, 0))
        self.right_x += dx_sign * self.paddle_speed_x
        self.right_y += dy_sign * self.paddle_speed_y
        self.right_x = float(max(self.right_x_min, min(self.right_x, self.right_x_max)))
        self.right_y = float(max(0, min(self.right_y, self.height - self.paddle_h)))

    def step(self, left_action: int, right_action: int):
        """
        Avanza un frame lógico.

        Devuelve recompensas para ambos jugadores para que el mismo motor
        sirva tanto al DQN como al PPO.
        """
        reward_left = 0.0
        reward_right = 0.0
        terminated = False

        # Distancias antes de mover, para reward shaping.
        left_dist_before = self._center_dist_to_ball("left")
        right_dist_before = self._center_dist_to_ball("right")

        self._apply_action_to_left(left_action)
        self._apply_action_to_right(right_action)

        left_dist_after = self._center_dist_to_ball("left")
        right_dist_after = self._center_dist_to_ball("right")

        # Reward shaping pequeño por acercarse a la pelota sólo
        # cuando la pelota viene hacia ese lado.
        if self.ball_vx < 0:
            if left_dist_after < left_dist_before:
                reward_left += self.track_ball_reward
            elif left_dist_after > left_dist_before:
                reward_left -= self.track_ball_penalty

        if self.ball_vx > 0:
            if right_dist_after < right_dist_before:
                reward_right += self.track_ball_reward
            elif right_dist_after > right_dist_before:
                reward_right -= self.track_ball_penalty

        # Mover pelota
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Rebote arriba/abajo
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy *= -1

        if self.ball_y >= self.height - self.ball_size:
            self.ball_y = self.height - self.ball_size
            self.ball_vy *= -1

        # Colisión pala izquierda
        if (
            self.ball_x <= self.left_x + self.paddle_w
            and self.ball_x + self.ball_size >= self.left_x
            and self.left_y <= self.ball_y + self.ball_size
            and self.ball_y <= self.left_y + self.paddle_h
            and self.ball_vx < 0
        ):
            self.ball_x = self.left_x + self.paddle_w
            self.ball_vx *= -1
            reward_left += self.hit_reward

        # Colisión pala derecha
        if (
            self.ball_x + self.ball_size >= self.right_x
            and self.ball_x <= self.right_x + self.paddle_w
            and self.right_y <= self.ball_y + self.ball_size
            and self.ball_y <= self.right_y + self.paddle_h
            and self.ball_vx > 0
        ):
            self.ball_x = self.right_x - self.ball_size
            self.ball_vx *= -1
            reward_right += self.hit_reward

        # Puntos
        if self.ball_x < 0:
            self.right_score += 1
            reward_left = self.lose_reward
            reward_right = self.score_reward
            self.reset_ball()

        elif self.ball_x > self.width:
            self.left_score += 1
            reward_left = self.score_reward
            reward_right = self.lose_reward
            self.reset_ball()

        # Fin de partido
        if self.left_score >= self.max_score:
            reward_left += self.match_win_bonus
            reward_right += self.match_lose_penalty
            terminated = True
            self.winner = "left"

        elif self.right_score >= self.max_score:
            reward_left += self.match_lose_penalty
            reward_right += self.match_win_bonus
            terminated = True
            self.winner = "right"

        return reward_left, reward_right, terminated, self.info()

    # ============================================================
    # RENDER
    # ============================================================

    def ensure_window(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Pong RL Arena")
            self.clock = pygame.time.Clock()

    def draw(self, surface, paused: bool = False, show_pause_button: bool = True):
        t = self.theme
        surface.fill(t.background)

        # Panel superior
        pygame.draw.rect(surface, t.score_panel, (0, 0, self.width, 78), border_radius=0)

        # Línea central
        pygame.draw.line(surface, t.center_line, (self.width // 2, 90), (self.width // 2, self.height), 2)

        # Límites horizontales
        pygame.draw.line(surface, t.limit_line, (int(self.left_x_max + self.paddle_w), 90), (int(self.left_x_max + self.paddle_w), self.height), 1)
        pygame.draw.line(surface, t.limit_line, (int(self.right_x_min), 90), (int(self.right_x_min), self.height), 1)

        # Palas y pelota
        pygame.draw.rect(surface, t.left_paddle, (self.left_x, self.left_y, self.paddle_w, self.paddle_h), border_radius=8)
        pygame.draw.rect(surface, t.right_paddle, (self.right_x, self.right_y, self.paddle_w, self.paddle_h), border_radius=8)
        pygame.draw.rect(surface, t.ball, (self.ball_x, self.ball_y, self.ball_size, self.ball_size), border_radius=8)

        title_font = pygame.font.SysFont("arial", 24, bold=True)
        score_font = pygame.font.SysFont("arial", 34, bold=True)
        small_font = pygame.font.SysFont("arial", 18)

        left_name_surf = title_font.render(t.left_name, True, t.text)
        right_name_surf = title_font.render(t.right_name, True, t.text)
        score_surf = score_font.render(f"{self.left_score}   -   {self.right_score}", True, t.text)

        surface.blit(left_name_surf, (40, 18))
        surface.blit(right_name_surf, (self.width - right_name_surf.get_width() - 40, 18))
        surface.blit(score_surf, (self.width // 2 - score_surf.get_width() // 2, 14))

        hint_surf = small_font.render("ESC o botón para pausar", True, t.subtext)
        surface.blit(hint_surf, (self.width // 2 - hint_surf.get_width() // 2, 50))

        pause_rect = pygame.Rect(self.width - 140, 18, 100, 34)
        if show_pause_button:
            pygame.draw.rect(surface, t.pause_button, pause_rect, border_radius=10)
            pause_text = small_font.render("Pausa", True, t.text)
            surface.blit(
                pause_text,
                (
                    pause_rect.centerx - pause_text.get_width() // 2,
                    pause_rect.centery - pause_text.get_height() // 2,
                ),
            )

        if paused:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((*t.overlay, 190))
            surface.blit(overlay, (0, 0))

        return pause_rect

    def render_to_window(self, paused: bool = False, show_pause_button: bool = True):
        self.ensure_window()
        pause_rect = self.draw(self.window, paused=paused, show_pause_button=show_pause_button)
        pygame.display.update()
        self.clock.tick(60)
        return pause_rect

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None