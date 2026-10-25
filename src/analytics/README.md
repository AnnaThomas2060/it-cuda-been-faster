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
