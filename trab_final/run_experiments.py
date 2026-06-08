"""
run_experiments.py
Experimento: Impacto da Exploração em SAC vs TD3 — Pendulum-v1
Autores: Daniel Higa & Luan
Disciplina: IA368 — UNICAMP 2026

Configuração completa:
  - SAC: α ∈ {0.01, 0.05, 0.10, 0.20, 'auto'}  × 10 seeds × 100k steps
  - TD3: σ ∈ {0.05, 0.10, 0.20, 0.30}            × 10 seeds × 100k steps
Preset completo:
  - Todas as 9 configurações × 10 seeds = 90 execuções balanceadas
  - Ambiente: Pendulum-v1
"""

import os
import json
import time
import argparse
import warnings
import traceback

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psutil
import torch
import gymnasium as gym

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

# ─────────────────────────── Configuração global ────────────────────────────

ENV_NAME      = "Pendulum-v1"
TIMESTEPS     = 100_000
SEEDS         = list(range(1, 11))
EVAL_EPISODES = 20
SAVE_MODELS   = True

# Diretórios
PROJECT_DIR = "project"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
LOGS_DIR    = os.path.join(PROJECT_DIR, "logs")

for d in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Hiperparâmetros base (idênticos para SAC e TD3)
BASE_PARAMS = dict(
    learning_rate  = 3e-4,
    buffer_size    = 100_000,
    learning_starts= 1_000,
    batch_size     = 256,
    tau            = 0.005,
    gamma          = 0.99,
    train_freq     = 1,
    gradient_steps = 1,
    verbose        = 0,
)

# Configurações de exploração
SAC_CONFIGS = [
    {"label": "SAC-1", "ent_coef": 0.01},
    {"label": "SAC-2", "ent_coef": 0.05},
    {"label": "SAC-3", "ent_coef": 0.10},
    {"label": "SAC-4", "ent_coef": 0.20},
    {"label": "SAC-5", "ent_coef": "auto"},
]

TD3_CONFIGS = [
    {"label": "TD3-1", "sigma": 0.05},
    {"label": "TD3-2", "sigma": 0.10},
    {"label": "TD3-3", "sigma": 0.20},
    {"label": "TD3-4", "sigma": 0.30},
]

EXPERIMENT_METADATA = {}

# ──────────────────────────── Callback de episódios ─────────────────────────

class EpisodeRewardCallback(BaseCallback):
    """Registra recompensa e comprimento de cada episódio."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int]   = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True

# ──────────────────────────── Funções auxiliares ────────────────────────────

def make_env(seed: int) -> Monitor:
    env = gym.make(ENV_NAME)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def evaluate_agent(model, n_episodes: int = EVAL_EPISODES, seed: int = 0) -> np.ndarray:
    """Avalia o agente deterministicamente e retorna array de recompensas."""
    rewards = []
    eval_env = gym.make(ENV_NAME)
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        done = truncated = False
        total = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = eval_env.step(action)
            total += r
        rewards.append(total)
    eval_env.close()
    return np.array(rewards)


def measure_ram_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024 / 1024


def run_sac(config: dict, seed: int) -> dict:
    """Treina um agente SAC e retorna métricas."""
    label     = config["label"]
    ent_coef  = config["ent_coef"]
    exp_value = ent_coef  # valor numérico ou 'auto'

    env = make_env(seed)
    ram_before = measure_ram_mb()
    t_start    = time.perf_counter()

    model = SAC(
        policy    = "MlpPolicy",
        env       = env,
        seed      = seed,
        ent_coef  = ent_coef,
        **BASE_PARAMS,
    )

    cb = EpisodeRewardCallback()
    model.learn(total_timesteps=TIMESTEPS, callback=cb, progress_bar=False)

    t_elapsed  = time.perf_counter() - t_start
    ram_after  = measure_ram_mb()
    ram_peak   = max(ram_before, ram_after)

    eval_rewards = evaluate_agent(model, seed=seed)

    # Número de episódios até atingir limiar de -200 (convergência Pendulum)
    threshold    = -200.0
    ep_rewards   = np.array(cb.episode_rewards)
    conv_episode = None
    for i, r in enumerate(ep_rewards):
        if r >= threshold:
            conv_episode = i + 1
            break

    if SAVE_MODELS:
        model_path = os.path.join(MODELS_DIR, f"{label}_seed{seed}.zip")
        model.save(model_path)
    env.close()

    return {
        "algorithm"     : "SAC",
        "label"         : label,
        "exploration"   : str(exp_value),
        "seed"          : seed,
        "mean_reward"   : float(eval_rewards.mean()),
        "std_reward"    : float(eval_rewards.std()),
        "max_reward"    : float(eval_rewards.max()),
        "min_reward"    : float(eval_rewards.min()),
        "final_reward"  : float(ep_rewards[-1]) if len(ep_rewards) > 0 else np.nan,
        "conv_episode"  : conv_episode,
        "n_episodes"    : len(ep_rewards),
        "train_time_s"  : t_elapsed,
        "ram_peak_mb"   : ram_peak,
        "episode_rewards": ep_rewards.tolist(),
        "episode_lengths": cb.episode_lengths,
    }


def run_td3(config: dict, seed: int) -> dict:
    """Treina um agente TD3 e retorna métricas."""
    label = config["label"]
    sigma = config["sigma"]

    env        = make_env(seed)
    n_actions  = env.action_space.shape[0]
    noise      = NormalActionNoise(
        mean  = np.zeros(n_actions),
        sigma = sigma * np.ones(n_actions),
    )

    ram_before = measure_ram_mb()
    t_start    = time.perf_counter()

    model = TD3(
        policy       = "MlpPolicy",
        env          = env,
        seed         = seed,
        action_noise = noise,
        **BASE_PARAMS,
    )

    cb = EpisodeRewardCallback()
    model.learn(total_timesteps=TIMESTEPS, callback=cb, progress_bar=False)

    t_elapsed  = time.perf_counter() - t_start
    ram_after  = measure_ram_mb()
    ram_peak   = max(ram_before, ram_after)

    eval_rewards = evaluate_agent(model, seed=seed)

    threshold    = -200.0
    ep_rewards   = np.array(cb.episode_rewards)
    conv_episode = None
    for i, r in enumerate(ep_rewards):
        if r >= threshold:
            conv_episode = i + 1
            break

    if SAVE_MODELS:
        model_path = os.path.join(MODELS_DIR, f"{label}_seed{seed}.zip")
        model.save(model_path)
    env.close()

    return {
        "algorithm"     : "TD3",
        "label"         : label,
        "exploration"   : str(sigma),
        "seed"          : seed,
        "mean_reward"   : float(eval_rewards.mean()),
        "std_reward"    : float(eval_rewards.std()),
        "max_reward"    : float(eval_rewards.max()),
        "min_reward"    : float(eval_rewards.min()),
        "final_reward"  : float(ep_rewards[-1]) if len(ep_rewards) > 0 else np.nan,
        "conv_episode"  : conv_episode,
        "n_episodes"    : len(ep_rewards),
        "train_time_s"  : t_elapsed,
        "ram_peak_mb"   : ram_peak,
        "episode_rewards": ep_rewards.tolist(),
        "episode_lengths": cb.episode_lengths,
    }


# ───────────────────────────── Loop principal ───────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa o estudo SAC vs TD3 sobre impacto da exploracao."
    )
    parser.add_argument("--env", default=ENV_NAME, help="Ambiente Gymnasium.")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS, help="Passos de treino por execucao.")
    parser.add_argument("--seeds", default="1-10", help="Seeds: use '1-10' ou '1,2,3'.")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES, help="Episodios de avaliacao.")
    parser.add_argument("--quick", action="store_true", help="Atalho: 2 seeds, 2.000 passos e 5 episodios de avaliacao.")
    parser.add_argument("--medium", action="store_true", help="Atalho completo: 10 seeds, 100.000 passos e 20 episodios de avaliacao (90 execucoes).")
    parser.add_argument("--no-save-models", action="store_true", help="Nao salva modelos .zip.")
    return parser.parse_args()


def parse_seed_spec(spec: str) -> list[int]:
    if "-" in spec:
        start, end = spec.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


def configure_from_args(args) -> None:
    global ENV_NAME, TIMESTEPS, SEEDS, EVAL_EPISODES, SAVE_MODELS, EXPERIMENT_METADATA

    if args.quick and args.medium:
        raise ValueError("Use apenas um preset: --quick ou --medium.")

    if args.quick:
        args.timesteps = 2_000
        args.seeds = "1,2"
        args.eval_episodes = 5
        args.no_save_models = True
    elif args.medium:
        args.timesteps = 100_000
        args.seeds = "1-10"
        args.eval_episodes = 20

    ENV_NAME = args.env
    TIMESTEPS = args.timesteps
    SEEDS = parse_seed_spec(args.seeds)
    EVAL_EPISODES = args.eval_episodes
    SAVE_MODELS = not args.no_save_models
    EXPERIMENT_METADATA = {
        "env_name": ENV_NAME,
        "timesteps": TIMESTEPS,
        "seeds": SEEDS,
        "eval_episodes": EVAL_EPISODES,
        "save_models": SAVE_MODELS,
        "sac_configs": SAC_CONFIGS,
        "td3_configs": TD3_CONFIGS,
    }


def main():
    configure_from_args(parse_args())
    all_results   = []
    episode_data  = {}   # label → list of episode reward arrays (per seed)

    total = len(SAC_CONFIGS) * len(SEEDS) + len(TD3_CONFIGS) * len(SEEDS)
    done  = 0

    print(f"{'='*60}")
    print(f" Experimento: SAC vs TD3 — {ENV_NAME}")
    print(f" Total de execuções: {total}")
    print(f"{'='*60}\n")

    # ── SAC ─────────────────────────────────────────────────────────────────
    for cfg in SAC_CONFIGS:
        label = cfg["label"]
        episode_data[label] = []

        for seed in SEEDS:
            done += 1
            print(f"[{done:2d}/{total}] {label} | α={cfg['ent_coef']} | seed={seed}", end=" ... ")
            try:
                result = run_sac(cfg, seed)
                all_results.append(result)
                episode_data[label].append(result["episode_rewards"])
                print(f"✓  mean_eval={result['mean_reward']:.1f}  t={result['train_time_s']:.0f}s")
            except Exception as e:
                print(f"✗  ERRO: {e}")
                traceback.print_exc()

        # Salva resultados parciais após cada configuração
        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("episode_rewards", "episode_lengths")}
                           for r in all_results])
        df.to_csv(os.path.join(RESULTS_DIR, "results_partial.csv"), index=False)

    # ── TD3 ─────────────────────────────────────────────────────────────────
    for cfg in TD3_CONFIGS:
        label = cfg["label"]
        episode_data[label] = []

        for seed in SEEDS:
            done += 1
            print(f"[{done:2d}/{total}] {label} | σ={cfg['sigma']} | seed={seed}", end=" ... ")
            try:
                result = run_td3(cfg, seed)
                all_results.append(result)
                episode_data[label].append(result["episode_rewards"])
                print(f"✓  mean_eval={result['mean_reward']:.1f}  t={result['train_time_s']:.0f}s")
            except Exception as e:
                print(f"✗  ERRO: {e}")
                traceback.print_exc()

        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("episode_rewards", "episode_lengths")}
                           for r in all_results])
        df.to_csv(os.path.join(RESULTS_DIR, "results_partial.csv"), index=False)

    # ── Salvamento final ─────────────────────────────────────────────────────
    df_final = pd.DataFrame([{k: v for k, v in r.items() if k not in ("episode_rewards", "episode_lengths")}
                              for r in all_results])
    df_final.to_csv(os.path.join(RESULTS_DIR, "results_all.csv"), index=False)

    # Salva séries de episódios para plotagem
    with open(os.path.join(RESULTS_DIR, "episode_series.json"), "w") as f:
        json.dump(episode_data, f)

    with open(os.path.join(RESULTS_DIR, "metadata.json"), "w") as f:
        json.dump(EXPERIMENT_METADATA, f, indent=2)

    print(f"\n{'='*60}")
    print(" Experimento concluído!")
    print(f" Resultados: {RESULTS_DIR}/results_all.csv")
    print(f"{'='*60}\n")
    print(df_final.groupby(["algorithm", "label", "exploration"])["mean_reward"].mean().round(2))


if __name__ == "__main__":
    main()
