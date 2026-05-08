from __future__ import annotations

import os
from copy import deepcopy

from stable_baselines3 import DQN

from common_config import (
    DEFAULT_THEME,
    DQN_BUFFER_PATH,
    DQN_KWARGS,
    DQN_MODEL_PATH,
    ENV_KWARGS,
    MODELS_DIR,
    POLICY_KWARGS,
    SEED,
    TOTAL_TIMESTEPS,
    make_bot_controller,
)
from pongenv import PongEnv

# ============================================================
# ENTRENAMIENTO JUSTO DE DQN
# ============================================================
# Criterios de justicia frente a PPO:
# - mismo entorno
# - mismo bot de entrenamiento
# - mismo seed
# - mismo budget de timesteps
# - misma arquitectura base de red (ver POLICY_KWARGS)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

theme = deepcopy(DEFAULT_THEME)
theme.left_name = "DQN"
theme.right_name = "BOT"

env = PongEnv(
    render_mode=None,
    theme=theme,
    opponent_controller=make_bot_controller(),
    **ENV_KWARGS,
)
env.reset(seed=SEED)
env.action_space.seed(SEED)

loaded_existing = False

if os.path.exists(DQN_MODEL_PATH + ".zip"):
    print("Cargando modelo DQN anterior...")
    model = DQN.load(DQN_MODEL_PATH, env=env)
    loaded_existing = True
    if os.path.exists(DQN_BUFFER_PATH):
        print("Cargando replay buffer DQN...")
        model.load_replay_buffer(DQN_BUFFER_PATH)
else:
    print("Creando DQN nuevo...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        seed=SEED,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        **DQN_KWARGS,
    )

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=not loaded_existing,
    progress_bar=True,
)

model.save(DQN_MODEL_PATH)
model.save_replay_buffer(DQN_BUFFER_PATH)
env.close()
print("DQN guardado en", DQN_MODEL_PATH + ".zip")
