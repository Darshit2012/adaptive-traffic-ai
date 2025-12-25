"""Adaptive AI-Based Traffic Signal Optimization System entrypoint.

Provides a CLI to simulate traffic with different controllers:
- fixed: time-of-day based fixed durations
- neural: lightweight feedforward learner
- rl: tabular Q-learning controller

Simulation logs are written to data/run_log.csv for dashboard and analysis.
"""
import argparse
import pathlib
import random
from typing import List, Dict

import pandas as pd

from data.traffic_generator import generate_stream
from models.neural_controller import NeuralController
from models.rl_controller import QLearningController
from simulation.multi_intersection import MultiIntersectionNetwork
from utils.metrics import summarize_metrics, comparison_table


class FixedTimeController:
    """Simple time-of-day fixed plan controller."""

    def __init__(self):
        # Morning/afternoon/night nominal durations in seconds
        self.plans = {
            "morning": 12,
            "afternoon": 10,
            "night": 6,
        }

    def decide(self, state: Dict) -> Dict:
        duration = self.plans.get(state.get("time_of_day"), 10)
        explain = "Fixed plan based on time-of-day profile."
        return {"duration": duration, "explanation": explain}

    def learn(self, *args, **kwargs):
        # Fixed controller does not adapt
        return None


def build_controller(controller_type: str):
    if controller_type == "fixed":
        return FixedTimeController()
    if controller_type == "neural":
        return NeuralController()
    if controller_type == "rl":
        return QLearningController()
    raise ValueError(f"Unknown controller type: {controller_type}")


def run_once(args, controller_name: str, suffix: str = "") -> Dict:
    controller = build_controller(controller_name)

    generator = generate_stream(
        steps=args.steps,
        intersections=args.intersections,
        profile=args.profile,
        emergency_rate=args.emergency_rate,
        seed=args.seed,
        varied_time=args.varied_time,
    )

    network = MultiIntersectionNetwork(
        intersections=args.intersections,
        controller_name=controller_name,
        service_rate=args.service_rate,
    )

    log_rows: List[Dict] = []
    last_states = {}

    for step_arrivals in generator:
        time_of_day = step_arrivals[0].get("time_of_day", "afternoon")
        decisions = []

        for inter_id, arrivals in enumerate(step_arrivals):
            state = network.observe_state(inter_id, time_of_day, arrivals["emergency"])
            decision = controller.decide(state)
            decisions.append(decision)
            last_states[inter_id] = state

        step_logs, rewards = network.step(step_arrivals, decisions)
        log_rows.extend(step_logs)

        if hasattr(controller, "learn"):
            for inter_id, reward in rewards.items():
                controller.learn(
                    state=last_states[inter_id],
                    reward=reward,
                    next_state=network.observe_state(inter_id, time_of_day, step_arrivals[inter_id]["emergency"]),
                )

    df = pd.DataFrame(log_rows)
    log_name = f"run_log{suffix}.csv"
    log_path = pathlib.Path("data") / log_name
    df.to_csv(log_path, index=False)
    metrics = summarize_metrics(df)
    return {"log_path": log_path, "metrics": metrics}


def run_simulation(args):
    controllers = [args.controller]
    if args.controllers:
        controllers = [c.strip() for c in args.controllers.split(",") if c.strip()]

    comparison = {}
    last_log = None

    for idx, ctrl in enumerate(controllers):
        suffix = "" if len(controllers) == 1 and ctrl == args.controller else f"_{ctrl}"
        result = run_once(args, ctrl, suffix)
        comparison[ctrl] = result["metrics"]
        last_log = result["log_path"]
        print(f"[{ctrl}] metrics:")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}")

    if len(controllers) > 1:
        comp_path = pathlib.Path("data/comparison.csv")
        comparison_table(comparison).to_csv(comp_path)
        print("Comparison table ->", comp_path)

    if last_log and last_log.name != "run_log.csv":
        # keep a default for the dashboard
        pathlib.Path("data/run_log.csv").write_bytes(last_log.read_bytes())
        print("Dashboard log synced -> data/run_log.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive AI-Based Traffic Signal Optimization System")
    parser.add_argument("--controller", choices=["fixed", "neural", "rl"], default="fixed", help="single controller to run")
    parser.add_argument("--controllers", type=str, default=None, help="comma-separated list to run comparisons, e.g., 'fixed,neural,rl'")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--intersections", type=int, default=2)
    parser.add_argument("--service_rate", type=int, default=1, help="vehicles served per second")
    parser.add_argument("--profile", choices=["morning", "afternoon", "night"], default="afternoon")
    parser.add_argument("--varied_time", action="store_true", help="cycle through morning/afternoon/night periods")
    parser.add_argument("--emergency_rate", type=float, default=0.02, help="probability of emergency per step")
    parser.add_argument("--seed", type=int, default=7)

    run_simulation(parser.parse_args())
