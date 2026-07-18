# Adapt-1 Pong V1–V3 evaluation

A reproducible Pong contact-dynamics evaluation for three Adapt-1 Domain
contracts. The three versions use the same deterministic experience set and
the same held-out queries while changing how the action and physical state are
represented.

This repository contains evaluation code only. It includes no run results,
API responses, traces, credentials, or pre-existing Domain state.

## Versions

| Version | Learner input | Action handling | Neighbors |
|---|---|---|---:|
| V1 | Incoming ball state and pre-action paddle center | Hard group by `executed_action` | 8 |
| V2 | V1 state plus signed `action_delta_y` and `delta_t` | Numeric input, no hard group | 3 |
| V3 | Incoming ball state plus observed post-action paddle center and `delta_t` | Action retained only for audit metadata | 8 |

The harness records observed transitions. It does not send Boolean reward
feedback, provide the collision threshold, provide a projected contact point,
name a correct action, or supply a solution hypothesis. Evaluation queries are
deterministic and set both `allow_exploration` and `update_memory_state` to
`false`.

## Repository layout

- `pong_collision_followup.py` — V1 fresh grouped-action Domain, phase-one
  training, and fixed evaluation.
- `pong_collision_phase2.py` — V1 exploratory boundary-bracketing continuation
  on the same Domain.
- `pong_domain_v2_shared_action.py` — V2 fresh numeric-action Domain using all
  phase-one and phase-two experiences.
- `pong_domain_v2_resume.py` — interruption-safe V2 continuation after checking
  the saved event count against the live structured-transition version.
- `pong_domain_v3_contact_state.py` — V3 fresh post-actuation-state Domain with
  built-in safe continuation.
- `METHODS.md` — exact dataset construction, Domain contracts, controls, and
  interpretation boundaries.
- `tests/test_harness.py` — offline checks for row counts, action balance,
  held-out separation, and query controls.

## Requirements

- Python 3.10 or newer.
- Access to an Adapt-1/NeuroAdapt API exposing `/api/v1/domains`.
- An API key entered at the hidden terminal prompt.

The harness uses only the Python standard library. The default API base URL is
the endpoint used to construct the evaluation. Override it with
`ADAPT1_BASE_URL` or `--base-url`.

Never place an API key in source, command-line arguments, output paths, or
committed files. These scripts accept it only through `getpass`.

## Validate offline

```bash
python -m unittest discover -s tests -v
python -m py_compile *.py
```

## Run V1

Phase one creates a fresh Domain, records 90 observations, and runs the fixed
81-row evaluation:

```bash
python pong_collision_followup.py \
  --results artifacts/v1-phase1.jsonl
```

The first `design` record contains the generated `domain_id`. Use it to
continue the same Domain with the exploratory boundary-bracketing phase. This
phase uses its own session identifier, matching the original evaluation design;
override it with `--session-id` only when needed.

```bash
python pong_collision_phase2.py \
  --domain-id <domain_id> \
  --results artifacts/v1-phase2.jsonl
```

## Run V2

```bash
python pong_domain_v2_shared_action.py \
  --results artifacts/v2.jsonl
```

If a transport failure interrupts training, keep the partial output and run:

```bash
python pong_domain_v2_resume.py \
  --results artifacts/v2.jsonl
```

The resume utility reads the Domain and session identifiers from the saved
`design` record and refuses to continue unless the logged event count equals
the live model version.

## Run V3

```bash
python pong_domain_v3_contact_state.py \
  --results artifacts/v3.jsonl
```

V3 is resumable with the same command and output path. It refuses to continue
if the local acknowledgements and live model version disagree.

## Generated artifacts

Runs write JSONL locally so every accepted observation and held-out prediction
can be audited. JSONL, results, logs, traces, environment files, and common
credential-file patterns are excluded by `.gitignore` and are not part of this
repository.

Phase two was selected after inspecting the unresolved boundary in phase one.
It is an exploratory intervention, not an untouched confirmatory evaluation.
V2 also changes more than one factor at once, so it should not be presented as
a clean single-variable ablation. See `METHODS.md` for the full scope.
