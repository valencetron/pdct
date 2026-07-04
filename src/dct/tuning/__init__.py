"""PDCT self-tuning engine (Build 106).

Public surface:
    dct.tuning.harness   — Tier 1 reference benchmark (shipped fixtures, deterministic)
    dct.tuning.engine    — shadow-replay tuner core (propose / evaluate / promote / revert)
    dct.tuning.watchdog  — drift detection + convergence state machine
    dct.tuning.telemetry — opt-in, allowlisted local telemetry
"""
