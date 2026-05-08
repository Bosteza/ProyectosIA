from __future__ import annotations

from common_config import DEFAULT_THEME, DQN_MODEL_PATH, EVAL_MATCHES_PER_SIDE, ENV_KWARGS, PPO_MODEL_PATH, SEED
from controllers import make_dqn_controller, make_ppo_controller
from pong_core import PongCore, Theme, mirror_perspective_action


def get_action(core: PongCore, controller, side: str) -> int:
    obs = core.get_obs_left() if side == "left" else core.get_obs_right()
    perspective_action = controller.act(obs)
    return perspective_action if side == "left" else mirror_perspective_action(perspective_action)


def play_match(left_controller, right_controller, match_seed: int, render: bool = False):
    theme = Theme(**DEFAULT_THEME.__dict__)
    core = PongCore(theme=theme, **ENV_KWARGS)
    core.set_rng(__import__("numpy").random.default_rng(match_seed))
    core.reset_match()

    while True:
        left_action = get_action(core, left_controller, "left")
        right_action = get_action(core, right_controller, "right")
        _, _, terminated, _ = core.step(left_action, right_action)
        if render:
            core.render_to_window(paused=False, show_pause_button=False)
        if terminated:
            winner = core.winner
            final_score = (core.left_score, core.right_score)
            core.close()
            return winner, final_score


def main():
    dqn = make_dqn_controller(DQN_MODEL_PATH)
    ppo = make_ppo_controller(PPO_MODEL_PATH)

    dqn_wins = 0
    ppo_wins = 0
    draws = 0

    print(f"Evaluando {EVAL_MATCHES_PER_SIDE * 2} partidos en total...")
    print("Mitad con DQN izquierda / PPO derecha y mitad al revés.\n")

    for i in range(EVAL_MATCHES_PER_SIDE):
        seed = SEED + i

        # DQN izquierda, PPO derecha
        winner, score = play_match(dqn, ppo, seed, render=False)
        if winner == "left":
            dqn_wins += 1
        elif winner == "right":
            ppo_wins += 1
        else:
            draws += 1

        # PPO izquierda, DQN derecha, mismo seed para misma distribución inicial
        winner, score = play_match(ppo, dqn, seed, render=False)
        if winner == "left":
            ppo_wins += 1
        elif winner == "right":
            dqn_wins += 1
        else:
            draws += 1

    total = dqn_wins + ppo_wins + draws
    print("=== RESULTADOS ===")
    print(f"Total partidos: {total}")
    print(f"Victorias DQN: {dqn_wins}")
    print(f"Victorias PPO: {ppo_wins}")
    print(f"Empates: {draws}")
    if total > 0:
        print(f"Winrate DQN: {100 * dqn_wins / total:.1f}%")
        print(f"Winrate PPO: {100 * ppo_wins / total:.1f}%")


if __name__ == "__main__":
    main()
