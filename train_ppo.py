from __future__ import annotations

import os
from copy import deepcopy

from stable_baselines3 import PPO

from common_config import (
    DEFAULT_THEME,
    ENV_KWARGS,
    MODELS_DIR,
    POLICY_KWARGS,
    PPO_KWARGS,
    PPO_MODEL_PATH,
    SEED,
    TOTAL_TIMESTEPS,
    make_bot_controller,
)
from pongenv import PongEnv

# ============================================================
# ENTRENAMIENTO JUSTO DE PPO
# ============================================================
# Criterios de justicia frente a DQN:
# - mismo entorno
# - mismo bot de entrenamiento
# - mismo seed
# - mismo budget de timesteps
# - misma arquitectura base de red (ver POLICY_KWARGS)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

theme = deepcopy(DEFAULT_THEME)
theme.left_name = "PPO"
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

if os.path.exists(PPO_MODEL_PATH + ".zip"):
    print("Cargando PPO anterior...")
    model = PPO.load(PPO_MODEL_PATH, env=env)
    loaded_existing = True
else:
    print("Creando PPO nuevo...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=SEED,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        **PPO_KWARGS,
    )

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=not loaded_existing,
    progress_bar=True,
)

model.save(PPO_MODEL_PATH)
env.close()
print("PPO guardado en", PPO_MODEL_PATH + ".zip")