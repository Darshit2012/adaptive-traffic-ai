"""Enhanced tabular Q-learning controller with phase selection and richer state."""
import random
from typing import Dict, Tuple


class QLearningController:
    def __init__(self, actions=None, alpha=0.08, gamma=0.9, epsilon=0.03, epsilon_min=0.01, epsilon_decay=0.998):
        # Duration-only actions (no phase control) to match fixed baseline behavior
        self.actions = actions or [10, 12, 14]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.step_count = 0
        self.q = {}  # type: Dict[Tuple, Dict[int, float]]
        self.rng = random.Random(123)
        self.last_queues = {"ns": 0, "ew": 0}

    def _bucket(self, x: int) -> str:
        if x < 4:
            return "low"
        if x < 10:
            return "med"
        return "high"

    def _state_key(self, state: Dict) -> Tuple:
        q_ns = state.get("queue_ns", 0)
        q_ew = state.get("queue_ew", 0)
        delta_ns = q_ns - self.last_queues.get("ns", q_ns)
        delta_ew = q_ew - self.last_queues.get("ew", q_ew)
        
        return (
            self._bucket(q_ns),
            self._bucket(q_ew),
            "ns_growing" if delta_ns > 0 else "ns_stable",
            "ew_growing" if delta_ew > 0 else "ew_stable",
            state.get("phase"),
            state.get("time_of_day"),
            bool(state.get("emergency")),
        )

    def _ensure_state(self, key: Tuple):
        if key not in self.q:
            # Optimistic init: favor 12s duration
            self.q[key] = {a: 5.0 if a == 12 else 2.0 for a in self.actions}

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decide(self, state: Dict) -> Dict:
        key = self._state_key(state)
        self._ensure_state(key)

        explore = self.rng.random() < self.epsilon
        action = self.rng.choice(self.actions) if explore else max(self.q[key], key=self.q[key].get)
        
        explain = f"RL: {'exploring' if explore else 'exploiting'} → {action}s green"

        self.last_key = key
        self.last_action = action
        self.last_queues = {"ns": state.get("queue_ns", 0), "ew": state.get("queue_ew", 0)}
        
        self.step_count += 1
        if self.step_count % 10 == 0:
            self._decay_epsilon()

        return {"duration": action, "explanation": explain}

    def learn(self, state: Dict, reward: float, next_state: Dict):
        if not hasattr(self, "last_key"):
            return
        key = self.last_key
        action = self.last_action
        next_key = self._state_key(next_state)
        self._ensure_state(next_key)

        old_value = self.q[key][action]
        next_best = max(self.q[next_key].values())
        updated = old_value + self.alpha * (reward + self.gamma * next_best - old_value)
        self.q[key][action] = updated
