## Adaptive AI-Based Traffic Signal Optimization System

Viva-ready, end-to-end AI traffic signal optimizer with multi-intersection simulation, emergency handling, explainability, and a polished Streamlit dashboard. Legacy C# code has been removed; this repository is now a clean Python project ready for GitHub.

### Problem Statement
Fixed-timing signals ignore live conditions, creating queues, delays, and unsafe emergency handling. This project learns from traffic state (queues, waits, throughput, emergencies) to adapt green durations, remain explainable, and coordinate across intersections.

### Features
- Controllers: fixed baseline, lightweight neural policy, tabular Q-learning
- Multi-intersection coordination: upstream outflow informs downstream demand
- Emergency priority: instant override with event logging
- Time-of-day intelligence: morning / afternoon / night demand profiles (or auto-cycling)
- Explainability: every decision stores a reason string
- Metrics: average wait proxy, throughput, stops, emergencies handled
- Streamlit dashboard: six tabs with live/run log visualization and comparisons

### Repository Layout
- data/traffic_generator.py — synthetic demand generator (time-of-day, emergencies, optional auto-cycle)
- models/neural_controller.py — lightweight feedforward learner
- models/rl_controller.py — tabular Q-learning controller
- simulation/intersection.py — single junction with emergency override and explainability
- simulation/multi_intersection.py — coordinated network wrapper
- utils/metrics.py — evaluation utilities and comparison table helper
- dashboard/app.py — Streamlit dashboard reading data/run_log.csv and data/comparison.csv
- main.py — CLI entrypoint for single or multi-controller runs

### Quickstart
```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Run a simulation (single controller)
python main.py --controller rl --steps 600 --intersections 3 --profile morning

# 3) Run a simulation with auto time-of-day cycling and comparisons
python main.py --controllers "fixed,neural,rl" --steps 600 --intersections 3 --varied_time

# 4) Launch the dashboard
python -m streamlit run dashboard/app.py
```
Outputs land in data/run_log.csv and data/comparison.csv for the dashboard.

### How It Works
- State: queues (NS/EW), current phase, time-of-day, emergency flag
- Actions: green duration choice (tabular RL: {10,12,14}s; neural outputs 4–16s)
- Reward: $r = \text{throughput} - \text{avg\_wait} - 0.3 \times \text{stops}$
- Controllers:
     - Fixed: profile-based static duration
     - Neural: small ANN trained online from rewards
     - RL (Q-learning): tabular action values with epsilon-greedy
- Coordination: upstream throughput shared as anticipated inflow downstream

### Evaluation
- Per-step log in data/run_log.csv with throughput, avg_wait_proxy, stops, emergencies
- utils.metrics.summarize_metrics builds KPI summaries; comparison_table writes data/comparison.csv
- Dashboard Tab 4 shows controller comparison; Tab 5 shows time-of-day adaptation and emergencies

### Explainability
- Each decision stores a readable explanation string (queues, time-of-day, emergency overrides)
- Dashboard Tab 3 lists the latest decisions for viva-ready discussion

### Future Scope
- Camera/IoT sensing for live counts
- DQN with replay for richer policies
- Pedestrian phases and amber/all-red safety timing
- Weather/incident features in state representation

### License
MIT License (matches the original repository)
