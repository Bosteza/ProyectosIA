from __future__ import annotations

import sys
from dataclasses import dataclass

import pygame

from common_config import DEFAULT_THEME, DQN_MODEL_PATH, ENV_KWARGS, PPO_MODEL_PATH
from controllers import HumanController, make_dqn_controller, make_ppo_controller
from pong_core import PongCore, Theme, mirror_perspective_action


@dataclass
class Button:
    text: str
    rect: pygame.Rect

    def draw(self, surface, font, theme: Theme, mouse_pos):
        hovered = self.rect.collidepoint(mouse_pos)
        color = theme.button_hover if hovered else theme.button
        pygame.draw.rect(surface, color, self.rect, border_radius=14)
        txt = font.render(self.text, True, theme.text)
        surface.blit(txt, (self.rect.centerx - txt.get_width() // 2, self.rect.centery - txt.get_height() // 2))


class PongArenaApp:
    """
    Launcher con estados:
    - MENU
    - GAME
    - PAUSE
    - END

    Aquí puedes cambiar:
    - colores del juego en self.theme
    - nombres visuales de las IAs
    - tamaño de botones y textos
    """

    def __init__(self):
        # TEMA VISUAL
        # Cambia aquí fondo, colores y nombres si quieres personalizar el display.
        self.theme = Theme(**DEFAULT_THEME.__dict__)

        self.core = PongCore(theme=self.theme, **ENV_KWARGS)
        self.core.ensure_window()

        self.title_font = pygame.font.SysFont("arial", 40, bold=True)
        self.button_font = pygame.font.SysFont("arial", 28, bold=True)
        self.small_font = pygame.font.SysFont("arial", 22)

        self.state = "MENU"
        self.mode = None
        self.error_text = None

        self.left_controller = None
        self.right_controller = None

        # Modelos
        try:
            self.dqn = make_dqn_controller(DQN_MODEL_PATH)
        except FileNotFoundError as e:
            self.dqn = None
            self.error_text = str(e)

        try:
            self.ppo = make_ppo_controller(PPO_MODEL_PATH)
        except FileNotFoundError as e:
            self.ppo = None
            self.error_text = str(e) if self.error_text is None else self.error_text

        self.human = HumanController()

        self.menu_buttons = [
            Button("Jugar contra DQN", pygame.Rect(300, 220, 300, 58)),
            Button("Jugar contra PPO", pygame.Rect(300, 300, 300, 58)),
            Button("Ver DQN vs PPO", pygame.Rect(300, 380, 300, 58)),
        ]

        self.pause_resume_button = Button("Reanudar", pygame.Rect(340, 260, 220, 56))
        self.pause_exit_button = Button("Salir al menú", pygame.Rect(340, 334, 220, 56))

        self.end_menu_button = Button("Volver al menú", pygame.Rect(340, 360, 220, 56))

    def configure_mode(self, mode: str):
        self.mode = mode
        self.core.reset_match()

        if mode == "HUMAN_VS_DQN":
            self.left_controller = self.dqn
            self.right_controller = self.human
            self.theme.left_name = "DQN"
            self.theme.right_name = "TÚ"

        elif mode == "HUMAN_VS_PPO":
            self.left_controller = self.ppo
            self.right_controller = self.human
            self.theme.left_name = "PPO"
            self.theme.right_name = "TÚ"

        elif mode == "AI_VS_AI":
            self.left_controller = self.dqn
            self.right_controller = self.ppo
            self.theme.left_name = "DQN"
            self.theme.right_name = "PPO"

        self.state = "GAME"

    def _menu_draw(self):
        surface = self.core.window
        surface.fill(self.theme.background)

        title = self.title_font.render("Pong RL Arena", True, self.theme.text)
        subtitle = self.small_font.render("Elige un modo", True, self.theme.subtext)

        surface.blit(title, (self.core.width // 2 - title.get_width() // 2, 92))
        surface.blit(subtitle, (self.core.width // 2 - subtitle.get_width() // 2, 140))

        mouse_pos = pygame.mouse.get_pos()
        for btn in self.menu_buttons:
            btn.draw(surface, self.button_font, self.theme, mouse_pos)

        tips = self.small_font.render("Durante la partida: flechas para moverte, ESC para pausar.", True, self.theme.subtext)
        surface.blit(tips, (self.core.width // 2 - tips.get_width() // 2, 510))

        if self.error_text:
            err = self.small_font.render("Faltan modelos. Corre train_dqn.py y train_ppo.py.", True, (255, 170, 170))
            surface.blit(err, (self.core.width // 2 - err.get_width() // 2, 560))

        pygame.display.update()
        self.core.clock.tick(60)

    def _pause_draw(self):
        surface = self.core.window
        self.core.draw(surface, paused=True, show_pause_button=True)

        title = self.title_font.render("Pausa", True, self.theme.text)
        surface.blit(title, (self.core.width // 2 - title.get_width() // 2, 170))

        mouse_pos = pygame.mouse.get_pos()
        self.pause_resume_button.draw(surface, self.button_font, self.theme, mouse_pos)
        self.pause_exit_button.draw(surface, self.button_font, self.theme, mouse_pos)

        pygame.display.update()
        self.core.clock.tick(60)

    def _end_draw(self):
        surface = self.core.window
        self.core.draw(surface, paused=True, show_pause_button=False)

        winner_name = self.theme.left_name if self.core.winner == "left" else self.theme.right_name
        title = self.title_font.render(f"Ganó {winner_name}", True, self.theme.text)
        surface.blit(title, (self.core.width // 2 - title.get_width() // 2, 190))

        score = self.button_font.render(f"{self.core.left_score} - {self.core.right_score}", True, self.theme.subtext)
        surface.blit(score, (self.core.width // 2 - score.get_width() // 2, 240))

        mouse_pos = pygame.mouse.get_pos()
        self.end_menu_button.draw(surface, self.button_font, self.theme, mouse_pos)

        pygame.display.update()
        self.core.clock.tick(60)

    def _get_controller_action(self, controller, side: str):
        if controller.action_mode == "screen":
            return controller.act()

        obs = self.core.get_obs_left() if side == "left" else self.core.get_obs_right()
        perspective_action = controller.act(obs)
        return perspective_action if side == "left" else mirror_perspective_action(perspective_action)

    def _game_tick(self):
        pause_rect = self.core.render_to_window(paused=False, show_pause_button=False)

        left_action = self._get_controller_action(self.left_controller, "left")
        right_action = self._get_controller_action(self.right_controller, "right")

        _, _, terminated, _ = self.core.step(left_action, right_action)
        if terminated:
            self.state = "END"
        return pause_rect

    def run(self):
        while True:
            if self.state == "MENU":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.core.close()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.menu_buttons[0].rect.collidepoint(event.pos) and self.dqn is not None:
                            self.configure_mode("HUMAN_VS_DQN")
                        elif self.menu_buttons[1].rect.collidepoint(event.pos) and self.ppo is not None:
                            self.configure_mode("HUMAN_VS_PPO")
                        elif self.menu_buttons[2].rect.collidepoint(event.pos) and self.dqn is not None and self.ppo is not None:
                            self.configure_mode("AI_VS_AI")

                self._menu_draw()

            elif self.state == "GAME":
                pause_rect = None
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.core.close()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.state = "PAUSE"
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and pause_rect is not None:
                        if pause_rect.collidepoint(event.pos):
                            self.state = "PAUSE"

                pause_rect = self._game_tick()

            elif self.state == "PAUSE":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.core.close()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self.state = "GAME"
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.pause_resume_button.rect.collidepoint(event.pos):
                            self.state = "GAME"
                        elif self.pause_exit_button.rect.collidepoint(event.pos):
                            self.state = "MENU"

                self._pause_draw()

            elif self.state == "END":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.core.close()
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                        self.state = "MENU"
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.end_menu_button.rect.collidepoint(event.pos):
                            self.state = "MENU"

                self._end_draw()


if __name__ == "__main__":
    app = PongArenaApp()
    app.run()
