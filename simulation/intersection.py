"""Intersection simulator with emergency override, time-of-day bias, and explainability."""
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class Intersection:
    intersection_id: int
    service_rate: int = 1

    def __post_init__(self):
        self.queue_ns = 0
        self.queue_ew = 0
        self.phase = "NS"  # NS or EW has green

    def observe_state(self, time_of_day: str, emergency: bool) -> Dict:
        return {
            "intersection": self.intersection_id,
            "queue_ns": self.queue_ns,
            "queue_ew": self.queue_ew,
            "phase": self.phase,
            "time_of_day": time_of_day,
            "emergency": emergency,
        }

    def _apply_emergency(self, emergency: bool) -> Tuple[bool, str]:
        if not emergency:
            return False, ""
        # give green to the larger queue as proxy for emergency direction
        self.phase = "NS" if self.queue_ns >= self.queue_ew else "EW"
        return True, "Emergency override: granted immediate green to priority approach."

    def step(self, arrivals: Dict, decision: Dict) -> Tuple[Dict, float]:
        self.queue_ns += arrivals["arrivals_ns"]
        self.queue_ew += arrivals["arrivals_ew"]
        emergency = arrivals["emergency"]

        overridden, override_msg = self._apply_emergency(emergency)

        duration = decision["duration"] if not overridden else max(5, decision["duration"])
        
        # Allow controller to choose phase (or auto-flip if not specified)
        if "target_phase" in decision and not overridden:
            phase_used = decision["target_phase"]
        else:
            phase_used = self.phase

        served_ns = served_ew = 0
        stops = 0

        if phase_used == "NS":
            served_ns = min(self.queue_ns, int(self.service_rate * duration))
            self.queue_ns -= served_ns
            stops += self.queue_ew  # all EW vehicles wait
            self.queue_ew += 0  # unchanged but explicit
        else:
            served_ew = min(self.queue_ew, int(self.service_rate * duration))
            self.queue_ew -= served_ew
            stops += self.queue_ns

        throughput = served_ns + served_ew
        avg_wait = (self.queue_ns + self.queue_ew) / 2.0 if (self.queue_ns + self.queue_ew) else 0

        explanation = decision.get("explanation", "")
        if override_msg:
            explanation += f" | {override_msg}"

        # Update phase for next step (RL controls; fixed/neural auto-flip)
        if "target_phase" in decision:
            self.phase = phase_used
        elif not emergency:
            self.phase = "EW" if self.phase == "NS" else "NS"

        log = {
            "step": arrivals["step"],
            "intersection": self.intersection_id,
            "time_of_day": arrivals["time_of_day"],
            "phase_used": phase_used,
            "duration": duration,
            "queue_ns": self.queue_ns,
            "queue_ew": self.queue_ew,
            "served_ns": served_ns,
            "served_ew": served_ew,
            "throughput": throughput,
            "avg_wait_proxy": avg_wait,
            "stops": stops,
            "emergency": emergency,
            "explanation": explanation,
        }

        reward = throughput - avg_wait - 0.3 * stops
        return log, reward
