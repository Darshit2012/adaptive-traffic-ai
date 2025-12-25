"""Lightweight feedforward neural controller.
Reuses ANN ideas but stabilizes learning with bounded targets and small steps.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math
import random


@dataclass
class NeuralController:
    input_size: int = 5  # ns queue, ew queue, phase flag, time-of-day bucket, emergency flag
    hidden_size: int = 6
    lr: float = 0.01
    memory: List = field(default_factory=list)

    def __post_init__(self):
        rng = random.Random(42)
        self.w1 = [[rng.uniform(-0.5, 0.5) for _ in range(self.input_size)] for _ in range(self.hidden_size)]
        self.b1 = [0.0 for _ in range(self.hidden_size)]
        self.w2 = [rng.uniform(-0.5, 0.5) for _ in range(self.hidden_size)]
        self.b2 = 0.0

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def _forward(self, features: List[float]) -> Tuple[float, List[float]]:
        h = []
        for i in range(self.hidden_size):
            s = sum(self.w1[i][j] * features[j] for j in range(self.input_size)) + self.b1[i]
            h.append(self._sigmoid(s))
        raw = sum(self.w2[i] * h[i] for i in range(self.hidden_size)) + self.b2
        pred_norm = self._sigmoid(raw)  # 0..1
        return pred_norm, h

    def _encode(self, state: Dict) -> List[float]:
        ns = min(1.0, state.get("queue_ns", 0) / 30.0)
        ew = min(1.0, state.get("queue_ew", 0) / 30.0)
        phase = 1.0 if state.get("phase") == "NS" else 0.0
        tod = {"morning": 1.0, "afternoon": 0.5, "night": 0.2}.get(state.get("time_of_day"), 0.5)
        emergency = 1.0 if state.get("emergency") else 0.0
        return [ns, ew, phase, tod, emergency]

    def decide(self, state: Dict) -> Dict:
        feats = self._encode(state)
        pred_norm, _ = self._forward(feats)
        duration = 4.0 + 12.0 * pred_norm  # 4s to 16s
        self.memory.append({"features": feats, "pred_norm": pred_norm})
        explanation = (
            f"Neural policy: queues NS/EW=({state.get('queue_ns')},{state.get('queue_ew')}), "
            f"time={state.get('time_of_day')}, emergency={state.get('emergency')}"
        )
        return {"duration": duration, "explanation": explanation}

    def learn(self, state: Dict, reward: float, next_state: Dict):
        if not self.memory:
            return
        last = self.memory[-1]
        feats = last["features"]
        pred_norm, h = self._forward(feats)

        # reward shaping: compress to [-1, 1]
        reward_scaled = max(-1.0, min(1.0, reward / 20.0))
        target = max(0.0, min(1.0, pred_norm + 0.3 * reward_scaled))

        error = pred_norm - target

        # Backprop to hidden and input
        grad_out = error * pred_norm * (1 - pred_norm)
        for i in range(self.hidden_size):
            self.w2[i] -= self.lr * grad_out * h[i]
            grad_h = grad_out * self.w2[i] * h[i] * (1 - h[i])
            for j in range(self.input_size):
                self.w1[i][j] -= self.lr * grad_h * feats[j]
            self.b1[i] -= self.lr * grad_h
        self.b2 -= self.lr * grad_out
