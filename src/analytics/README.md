# Analytics Vertical

Owner: Mohamed Ziyadh Mohamed

## Overview
This module consumes trajectory telemetry records to monitor training progress and detect reward hacking.

## Telemetry Interface
Ingests `TelemetryRecord` streams emitted to `logs/trajectory_telemetry.jsonl` by the RL pipeline.

## Responsibilities
- Track reward convergence and median or peak speedups across benchmark tasks.
- Perform Roofline model utilization audits against theoretical peak limits.
- Flag false gains, memory leaks, and compilation bypass attempts.

## Usage
```bash
python -m src.analytics logs/trajectory_telemetry.jsonl --gpu T4
```
Prints per-task pass rates and speedups, then lists records flagged as fake gains.

- `telemetry.py`: `append_record`, `load_records`, `summarize`, `reward_curve`.
- `roofline.py`: minimum bytes moved, minimum possible runtime, and bandwidth utilization per task.
- `fake_gain.py`: `audit` flags reward without correctness, implausible speedups, and bandwidth above the GPU's peak.
