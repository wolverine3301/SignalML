"""Production pipeline stages (contracts: docs/PIPELINE_AND_CONTRACTS.md §3).

Uniform job shape: read manifest -> select work -> process per song -> write artifacts
+ update manifest status -> emit summary stats. Each stage is idempotent,
config-driven (one YAML in configs/), and runnable via ``signalml <stage>``.

Modules are stubs until their migration phase lands:
acquire (P1) · separate (P2) · clean (P3) · features (P4) · align (P5) · dataset (P7).
"""
