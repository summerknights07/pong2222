# RoboCasa hypothesis probe

This repository contains an Adapt-1 API probe and its raw, key-free traces.
The probe uses structured fixture-manipulation transitions derived from public
RoboCasa atomic task semantics. It does not run the MuJoCo simulator.

## Findings

- Without a Domain, `/memory/explain` returned no hypotheses through sample 12.
- With a Domain but no declared candidates, `/query` and `/explain` returned no
  ranked hypotheses through sample 12.
- Declared Domain candidates appeared on the first query, before any samples.
- Ordinary `/events` updated structured-transition state but did not change the
  candidates' hypothesis evidence through sample 12.
- `/feedback` changed candidate policy scores and selection after one sample.
- `core_support.mechanistic.hypotheses` remained empty in all tested conditions.

## Contents

The complete probe is in [`artifacts/robocasa-hypothesis-probe`](artifacts/robocasa-hypothesis-probe):

- `run_robocasa_hypothesis_probe.py` runs the API conditions.
- `README.md` describes the conditions and recorded runs.
- `runs/` contains manifests and raw JSONL traces.

The probe reads its credential from `ADAPT1_API_KEY`. It does not write the
credential to output.

## Validate the script

```bash
python -m py_compile artifacts/robocasa-hypothesis-probe/run_robocasa_hypothesis_probe.py
```
