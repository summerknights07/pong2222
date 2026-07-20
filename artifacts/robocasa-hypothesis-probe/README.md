# RoboCasa hypothesis probe

This directory contains an Adapt-1 API probe and its raw, key-free JSONL traces.
Each completed condition used 12 structured RoboCasa fixture-manipulation
transitions. These are API-level samples derived from public RoboCasa atomic task
semantics, not MuJoCo rollouts.

## Results

- Without a Domain, `/memory/explain` returned no hypotheses through sample 12.
- With a Domain but no declared candidates, `/query` and `/explain` returned no
  ranked hypotheses through sample 12.
- Declared Domain candidates appeared on the first query, before any samples.
- Ordinary `/events` trained structured-transition state but did not change the
  candidates' hypothesis evidence through sample 12.
- `/feedback` changed candidate policy scores and selection after one sample.
- `core_support.mechanistic.hypotheses` remained empty in all tested conditions.

## Contents

- `run_robocasa_hypothesis_probe.py`: reusable probe; reads the credential from
  `ADAPT1_API_KEY` and never writes it to output.
- `runs/robocasa-hyp-1784571986-6888`: initial request rejected by Cloudflare
  before any write was accepted.
- `runs/robocasa-hyp-1784572014-3721`: no-Domain, Domain-without-candidates, and
  initial declared-candidate comparison.
- `runs/robocasa-hyp-1784572404-1688`: corrected declared-candidate context run.
- `runs/robocasa-hyp-1784572634-7579`: feedback-driven candidate run.

Each run includes `manifest.json` and `trace.jsonl`.
