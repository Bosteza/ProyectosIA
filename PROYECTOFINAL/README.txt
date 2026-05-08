PONG RL ARENA — CONFIGURACIÓN EQUILIBRADA DQN vs PPO
===================================================

Qué cambió para hacerlo lo más justo posible:
- Mismo entorno para DQN y PPO.
- Mismo bot heurístico congelado para entrenar.
- Mismo seed.
- Mismo presupuesto de entrenamiento: 250000 timesteps para cada una.
- Misma arquitectura base de red: [128, 128].
- Mismo espacio de observación y mismo espacio de acciones.

Archivos principales
--------------------
- pong_core.py      -> física, render y tema visual.
- pongenv.py        -> wrapper Gymnasium para entrenar una IA contra bot.
- controllers.py    -> humano, bot heurístico, DQN y PPO.
- common_config.py  -> configuración compartida para igualdad y rutas.
- train_dqn.py      -> entrena DQN.
- train_ppo.py      -> entrena PPO.
- game_app.py       -> menú con 3 modos y pausa.

Orden de uso
------------
1) python train_dqn.py
2) python train_ppo.py
3) python game_app.py

Opciones del menú
-----------------
- Jugar contra DQN
- Jugar contra PPO
- Ver DQN vs PPO

Pausa
-----
- ESC o botón Pausa
- Reanudar
- Salir al menú

Cómo cambiar colores, nombres y estilo visual
---------------------------------------------
Lo más fácil es abrir common_config.py y editar DEFAULT_THEME.

Ejemplos:
- background=(8, 12, 20)   -> cambia el color del fondo
- left_name="DQN Azul"     -> cambia el nombre mostrado a la izquierda
- right_name="PPO Rosa"    -> cambia el nombre mostrado a la derecha
- left_paddle=(0, 200, 255)
- right_paddle=(255, 80, 120)

Notas sobre “justicia”
----------------------
Nunca existe una justicia perfecta entre algoritmos distintos, pero esta versión iguala
los factores más importantes: entorno, rival de entrenamiento, arquitectura base, seed y
timesteps de experiencia.
