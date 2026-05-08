from __future__ import annotations

from pathlib import Path

from controllers import HeuristicBotController
from pong_core import Theme

# ============================================================
# RUTAS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DQN_MODEL_PATH = str(MODELS_DIR / "dqn_pong")
DQN_BUFFER_PATH = str(MODELS_DIR / "dqn_pong_replay.pkl")
PPO_MODEL_PATH = str(MODELS_DIR / "ppo_pong")

# ============================================================
# JUSTICIA DE LA COMPARACIÓN
# ============================================================
# Esta versión intenta ser más limpia y reproducible:
# - mismo entorno
# - mismo bot de entrenamiento
# - mismo presupuesto de muestras (timesteps)
# - misma arquitectura base de red
# - mismo seed
# - evaluación alternando lados
SEED = 7
#TOTAL_TIMESTEPS = 300_000 
EVAL_MATCHES_PER_SIDE = 25  # total = 50 partidos (25 con cada orientación)

#POLICY_KWARGS = dict(net_arch=[128, 128])



# ============================================================
# HIPERPARÁMETROS POR ALGORITMO EXPERTOS
# ============================================================
# TIMESTEPS
TOTAL_TIMESTEPS = 2_000_000

# POLICY
POLICY_KWARGS = dict(net_arch=[256, 256, 256])

# DQN#
DQN_KWARGS = dict(
    # LR más pequeña = aprendizaje más estable
    learning_rate=1e-4,

    # Buffer enorme para tener mucha diversidad de experiencias
    buffer_size=300_000,

    # Espera más antes de empezar a aprender
    # para que el buffer no esté tan "verde"
    learning_starts=20_000,

    # Batch grande = gradientes más estables

    batch_size=256,
    # Horizonte un poco más largo
    gamma=0.995,

    # Cada 4 pasos del entorno, entrena

    train_freq=4,
    # Hace varias actualizaciones por ciclo de entrenamiento
    # para exprimir más el replay buffer

    gradient_steps=4,
    # Actualiza más lentamente la target network

    target_update_interval=5_000,
    # Explora durante más tiempo
    exploration_fraction=0.30,

    # Empieza totalmente exploratoria
    exploration_initial_eps=1.0,

    # Termina casi determinista
    exploration_final_eps=0.01,
)


# PPO
PPO_KWARGS = dict(
    # LR más baja para estabilidad

    learning_rate=1e-4,
    # Rollout muy largo
    n_steps=8192,

    # Batch grande = updates más estables
    batch_size=256,

    # Horizonte algo más largo
    gamma=0.995,

    # Ventajas más suaves
    gae_lambda=0.98,

    # Clipping más conservador
    clip_range=0.15,

    # Sin entropía extra
    ent_coef=0.0,

    # Peso estándar para value function
    vf_coef=0.5,

    # Gradiente recortado para estabilidad
    max_grad_norm=0.5,

    # Más épocas por rollout
    n_epochs=15,

    # KL target para evitar updates demasiado agresivos
    target_kl=0.03,
)

# ============================================================
# HIPERPARÁMETROS POR ALGORITMO SIMPLES
# ============================================================
# DQN suele aprovechar mejor datos escasos porque reutiliza experiencia.
#DQN_KWARGS = dict(
  #  buffer_size=100_000,
 #   learning_rate=5e-4,
   # learning_starts=2_000,
    #gamma=0.99,
   ## batch_size=64,
    #train_freq=4,
    #gradient_steps=1,
    #target_update_interval=1_000,
    #exploration_fraction=0.15,
    #exploration_initial_eps=1.0,
    #exploration_final_eps=0.05,

#)

# PPO queda un poco menos castigada con un rollout algo más largo
# y menos entropía al final del aprendizaje.
#PPO_KWARGS = dict(
 #   learning_rate=3e-4,
  #  n_steps=2048,
   # batch_size=64,
    #gamma=0.99,
    #gae_lambda=0.95,
    #clip_range=0.2,
    #ent_coef=0.001,
    #vf_coef=0.5,
    #max_grad_norm=0.5,
#)



# ============================================================
# TEMA / UI
# ============================================================
DEFAULT_THEME = Theme(
    background=(0, 102, 0),
    ball=(128,255,0),
    center_line=(255, 255, 255),
    limit_line=(255, 255, 255),
    left_paddle=(90, 176, 255),
    right_paddle=(255, 128, 128),
    score_panel=(20, 26, 42),
    button=(42, 55, 90),
    button_hover=(66, 82, 132),
    pause_button=(70, 84, 125),
    pause_button_hover=(92, 110, 160),
    left_name="DQN",
    right_name="PPO",
)

# ============================================================
# ENTORNO COMPARTIDO
# ============================================================
ENV_KWARGS = dict(
    max_score=5,
    width=900,
    height=640,
    paddle_h=110,
    paddle_speed_y=8,
    paddle_speed_x=6,
    ball_size=14,
    ball_speed_x=4.5,
    track_ball_reward=0.01,
    track_ball_penalty=0.005,
    hit_reward=0.20,
    score_reward=1.0,
    lose_reward=-1.0,
    match_win_bonus=2.0,
    match_lose_penalty=-2.0,
    center_margin=24,
    left_x_initial=30,
    right_x_margin=30,
)

# ============================================================
# BOT DE ENTRENAMIENTO COMPARTIDO
# ============================================================
# En vez de compartir una instancia global mutable, usamos una fábrica.
# Así cada entrenamiento recibe su propio bot, pero con la misma configuración.
def make_bot_controller() -> HeuristicBotController:
    paddle_half_height_norm = 0.5 * (ENV_KWARGS["paddle_h"] / ENV_KWARGS["height"])
    return HeuristicBotController(
        deadzone_y=0.035,
        deadzone_x=0.035,
        return_x=ENV_KWARGS["left_x_initial"] / ENV_KWARGS["width"],
        paddle_half_height_norm=paddle_half_height_norm,
    )
