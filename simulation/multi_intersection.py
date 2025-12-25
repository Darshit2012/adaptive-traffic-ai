"""Network of intersections with light coordination via shared downstream signals."""
from typing import Dict, List, Tuple
from simulation.intersection import Intersection


class MultiIntersectionNetwork:
    def __init__(self, intersections: int, controller_name: str, service_rate: int = 1):
        self.intersections = [Intersection(i, service_rate=service_rate) for i in range(intersections)]
        self.controller_name = controller_name

    def observe_state(self, idx: int, time_of_day: str, emergency: bool) -> Dict:
        return self.intersections[idx].observe_state(time_of_day, emergency)

    def step(self, arrivals: List[Dict], decisions: List[Dict]) -> Tuple[List[Dict], Dict]:
        logs = []
        rewards = {}
        downstream_inflows = [0 for _ in self.intersections]

        # First pass: apply upstream arrivals plus downstream inflow spillover
        for i, _ in enumerate(self.intersections):
            arrivals[i]["arrivals_ns"] += downstream_inflows[i]

        for i, inter in enumerate(self.intersections):
            log, reward = inter.step(arrivals[i], decisions[i])
            logs.append(log)
            rewards[i] = reward

            # Simple coordination: pass a fraction of throughput downstream as anticipated arrival
            if i + 1 < len(self.intersections):
                downstream_inflows[i + 1] += int(0.6 * log["throughput"])

        return logs, rewards
