import argparse
import json
import os
import yaml
import numpy as np
import torch

import jsp_instance_utils.instances as benchmark_instances
from graph_jsp_env.disjunctive_graph_jsp_env import DisjunctiveGraphJspEnv

from jsp_rl.encoder import Encoder, VariationalEncoder
from jsp_rl.rl_model import JSPActorCritic
from jsp_rl.env_wrapper import ObservationWrapper


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_benchmark_instance(name):
    if not hasattr(benchmark_instances, name):
        raise ValueError(f"Unknown instance '{name}' in jsp_instance_utils.instances")

    instance = getattr(benchmark_instances, name)

    if callable(instance):
        instance = instance()

    return instance


def load_pretrained_encoder(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    enc_cfg = ckpt["config"]

    deg = ckpt.get("deg", None)
    if deg is not None:
        deg = deg.to(device)

    encoder_type = enc_cfg.get("model", "gae")
    EncoderClass = VariationalEncoder if encoder_type == "vgae" else Encoder

    encoder = EncoderClass(
        in_channels=enc_cfg["in_dim"],
        hidden_channels=enc_cfg["hidden_dim"],
        out_channels=enc_cfg["latent_dim"],
        gnn_type=enc_cfg["gnn_type"],
        deg=deg,
    ).to(device)

    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad_(False)

    return encoder, enc_cfg["latent_dim"]


def build_encoder_path(instance_size, encoder_name):
    return os.path.join(
        "trained_encoder_weights",
        instance_size,
        f"{encoder_name}.pt",
    )


def load_policy(config, policy_path, token_dim, n_tokens, device):
    model = JSPActorCritic(
        token_dim=token_dim,
        hidden_dim=config["model"]["hidden_dim"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        dropout=config["model"]["dropout"],
        n_tokens=n_tokens,
    ).to(device)

    state = torch.load(policy_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    return model


@torch.no_grad()
def rollout_policy(model, instance, config, device, encoder=None, latent_dim=None):
    env = DisjunctiveGraphJspEnv(
        jps_instance=instance,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
        reward_function="zero",
    )

    env = ObservationWrapper(
        env,
        instance,
        obs_mode=config["observation"]["mode"],
        encoder=encoder,
        latent_dim=latent_dim,
        device=device,
        sample_latent=config["encoder"].get("sample_latent", False),
    )

    obs, _ = env.reset()
    done = False
    truncated = False
    actions = []

    while not (done or truncated):
        mask = env.unwrapped.valid_action_mask()

        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        mask_t = torch.tensor(
            mask,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)

        logits, _ = model.get_logits_and_value(obs_t, mask_t)
        action = int(torch.argmax(logits, dim=1).item())

        obs, reward, done, truncated, info = env.step(action)
        actions.append(action)

    return int(info["makespan"])


def select_heuristic_action(env, instance, rule):
    valid_actions = env.unwrapped.valid_action_list()

    if len(valid_actions) == 0:
        raise RuntimeError("No valid actions available.")

    if rule == "fifo":
        return int(valid_actions[0])

    machines = instance[0]
    durations = instance[1]

    n_jobs = instance.shape[1]
    n_machines = instance.shape[2]

    def job_of(action):
        return action // n_machines

    def op_of(action):
        return action % n_machines

    if rule == "spt":
        return int(min(
            valid_actions,
            key=lambda a: durations[job_of(a), op_of(a)]
        ))

    if rule == "mwkr":
        return int(max(
            valid_actions,
            key=lambda a: durations[job_of(a), op_of(a):].sum()
        ))

    raise ValueError(f"Unknown heuristic rule: {rule}")

def rollout_heuristic(instance, rule):
    env = DisjunctiveGraphJspEnv(
        jps_instance=instance,
        perform_left_shift_if_possible=True,
        normalize_observation_space=True,
        flat_observation_space=False,
        action_mode="task",
        reward_function="zero",
    )

    obs, _ = env.reset()
    done = False
    truncated = False

    while not (done or truncated):
        action = select_heuristic_action(env, instance, rule)
        obs, reward, done, truncated, info = env.step(action)

    return int(info["makespan"])

def evaluate_instances(
    model,
    instance_names,
    config,
    device,
    encoder=None,
    latent_dim=None,
):
    gaps = []
    heuristic_gaps = {
    "fifo": [],
    "spt": [],
    "mwkr": [],
    }

    for name in instance_names:

        instance = load_benchmark_instance(name)

        makespan = rollout_policy(
            model=model,
            instance=instance,
            config=config,
            device=device,
            encoder=encoder,
            latent_dim=latent_dim,
        )

        optimum = getattr(
            benchmark_instances,
            f"{name}_makespan"
        )

        gap = 100.0 * (makespan - optimum) / optimum
        print(optimum)
        gaps.append(gap)

        print(
            f"{name}: "
            f"makespan={makespan}, "
            f"opt={optimum}, "
            f"gap={gap:.2f}%"
        )
        for rule in ["fifo", "spt", "mwkr"]:
            heuristic_makespan = rollout_heuristic(instance, rule)
            heuristic_gap = 100.0 * (heuristic_makespan - optimum) / optimum
            heuristic_gaps[rule].append(heuristic_gap)

            print(
                f"{name} {rule.upper()}: "
                f"makespan={heuristic_makespan}, "
                f"gap={heuristic_gap:.2f}%"
            )

    print()
    print(f"Average gap: {np.mean(gaps):.2f}%")
    print("\nHeuristic average gaps:")
    for rule, rule_gaps in heuristic_gaps.items():
        print(f"{rule.upper()}: {np.mean(rule_gaps):.2f}%")
    return np.mean(gaps)


if __name__ == "__main__":

    # Parse command-line arguments and load configuration.
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--policy", type=str, required=True)

    parser.add_argument(
        "--instances",
        nargs="+",
        required=True,
        help="Benchmark instance names from jsp_instance_utils.instances.",
    )

    parser.add_argument(
        "--observation_mode",
        type=str,
        default=None,
        choices=["raw_graph", "graph_features", "handcrafted", "encoder"],
    )

    parser.add_argument("--encoder_name", type=str, default=None)
    parser.add_argument("--instance_size", type=str, default=None)
    parser.add_argument("--out", type=str, default="benchmark_results.json")

    args = parser.parse_args()

    config = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder if latent representations are used.
    if args.observation_mode is not None:
        config["observation"]["mode"] = args.observation_mode

    encoder = None
    latent_dim = None

    if config["observation"]["mode"] == "encoder":
        encoder_path = build_encoder_path(
            instance_size=args.instance_size,
            encoder_name=args.encoder_name,
        )

        config["encoder"]["path"] = encoder_path
        encoder, latent_dim = load_pretrained_encoder(
            encoder_path,
            device,
        )

    # Infer observation dimensions from the first benchmark instance.
    first_instance = load_benchmark_instance(args.instances[0])

    n_jobs = first_instance.shape[1]
    n_machines = first_instance.shape[2]
    n_tokens = n_jobs * n_machines

    if config["observation"]["mode"] == "encoder":
        token_dim = latent_dim
    else:
        dummy_env = DisjunctiveGraphJspEnv(
            jps_instance=first_instance,
            perform_left_shift_if_possible=True,
            normalize_observation_space=True,
            flat_observation_space=False,
            action_mode="task",
            reward_function="zero",
        )

        dummy_env = ObservationWrapper(
            dummy_env,
            first_instance,
            obs_mode=config["observation"]["mode"],
            encoder=encoder,
            latent_dim=latent_dim,
            device=device,
            sample_latent=config["encoder"].get("sample_latent", False),
        )

        token_dim = dummy_env.observation_space.shape[-1]

    # Load trained PPO policy.
    model = load_policy(
        config=config,
        policy_path=args.policy,
        token_dim=token_dim,
        n_tokens=n_tokens,
        device=device,
    )

    print(f"Loaded policy: {args.policy}")

    # Evaluate on all benchmark instances and compare with heuristics.
    avg_gap = evaluate_instances(
        model=model,
        instance_names=args.instances,
        config=config,
        device=device,
        encoder=encoder,
        latent_dim=latent_dim,
    )

    print(f"\nAverage optimality gap: {avg_gap:.2f}%")