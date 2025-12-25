"""Traffic demand generator for simulations.
Generates per-step arrivals for each intersection with time-of-day profiles and emergency events.
"""
import random
from typing import Dict, List

TIME_PROFILES = {
    "morning": {"ns_rate": (3, 6), "ew_rate": (2, 4)},
    "afternoon": {"ns_rate": (2, 4), "ew_rate": (2, 4)},
    "night": {"ns_rate": (1, 2), "ew_rate": (0, 1)},
}


def _draw_rate(rng: random.Random, rate_range):
    return rng.randint(rate_range[0], rate_range[1])


def generate_stream(
    steps: int = 200,
    intersections: int = 2,
    profile: str = "afternoon",
    emergency_rate: float = 0.02,
    seed: int = 7,
    varied_time: bool = False,
):
    """Generate traffic stream with optional time-of-day variation.
    
    Args:
        steps: Number of simulation steps
        intersections: Number of intersections
        profile: Single time profile (ignored if varied_time=True)
        emergency_rate: Probability of emergency vehicle per step
        seed: Random seed for reproducibility
        varied_time: If True, cycles through morning/afternoon/night periods
    """
    rng = random.Random(seed)
    
    # Define time period cycle (each period lasts 1/3 of total steps)
    time_periods = ["morning", "afternoon", "night"]
    period_length = steps // 3 if varied_time else steps

    for step in range(steps):
        # Determine current time period
        if varied_time:
            period_idx = (step // period_length) % 3
            current_profile = time_periods[period_idx]
        else:
            current_profile = profile
            
        profile_cfg = TIME_PROFILES.get(current_profile, TIME_PROFILES["afternoon"])
        
        step_arrivals: List[Dict] = []
        for _ in range(intersections):
            arrivals_ns = _draw_rate(rng, profile_cfg["ns_rate"])
            arrivals_ew = _draw_rate(rng, profile_cfg["ew_rate"])
            emergency = rng.random() < emergency_rate
            step_arrivals.append(
                {
                    "step": step,
                    "time_of_day": current_profile,
                    "arrivals_ns": arrivals_ns,
                    "arrivals_ew": arrivals_ew,
                    "emergency": emergency,
                }
            )
        yield step_arrivals
