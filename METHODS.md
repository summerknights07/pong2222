# Methods

## Evaluation question

The evaluation asks how three Domain contracts affect online transition
learning at a sharp Pong paddle-collision boundary. It is designed to separate:

- learning from a cold-state guess;
- action effects from per-action sampling bias;
- categorical action routing from shared physical-state reuse;
- smooth numeric prediction from return/miss classification; and
- cross-action consistency from correctness.

The external harness chooses interventions and produces simulator outcomes.
Adapt-1 does not choose what experiment to run next, so this is not an
evaluation of autonomous exploration or multi-step Pong play.

## Simulator boundary

Coordinates are normalized and vertical position increases downward. The
player paddle half-height and one actuator step are both `0.105`. The control
interval is `0.46`.

The evaluator computes:

```text
post_paddle_y = clamp(initial_paddle_y + action_delta_y)
post_offset   = post_paddle_y - contact_y
return        = abs(post_offset) <= 0.105
next_ball_vx  = -ball_vx on return, otherwise ball_vx
```

This rule is harness truth. The Domain does not receive the contact threshold,
`contact_y`, a Boolean collision feature, a preferred action, or a solution.

## Deterministic trajectory generation

Each base trajectory samples an incoming horizontal speed, a crossing time,
vertical velocity, and a contact height from fixed pseudorandom seed families.
Training and evaluation use separate seed families. All versions therefore see
the same training multiset and the same held-out queries without repeating an
evaluation trajectory during training.

Two complementary designs are used:

1. `aligned_post_geometry`: each action begins from a different paddle center
   but ends at the same contact geometry. This tests whether equivalent physical
   states remain consistent across action labels.
2. `same_initial_state`: the initial ball and paddle state is fixed while the
   action changes. This tests whether the learner preserves the actuator's
   causal effect.

## Training set

Phase one contains 90 observations:

| Component | Construction | Rows |
|---|---|---:|
| Aligned post geometry | 4 trajectories × 3 actions × 5 offsets (`-0.18`, `-0.09`, `0`, `0.09`, `0.18`) | 60 |
| Same initial state | 5 trajectories × 3 actions × 2 offsets (`-0.08`, `0.08`) | 30 |

Phase two contains 72 new boundary-bracketing observations:

| Component | Construction | Rows |
|---|---|---:|
| Aligned post geometry | 4 trajectories × 3 actions × 6 offsets (`-0.15`, `-0.11`, `-0.10`, `0.10`, `0.11`, `0.15`) | 72 |

After both phases, each version has 162 observations, 54 per action. Phase two
uses new trajectory seeds and adds no held-out evaluation row to training.

## Held-out evaluation set

The fixed evaluation contains 81 rows:

| Component | Construction | Rows |
|---|---|---:|
| Aligned post geometry | 3 unseen trajectories × 3 actions × 7 offsets (`-0.16`, `-0.11`, `-0.08`, `0`, `0.08`, `0.11`, `0.16`) | 63 |
| Same initial state | 3 unseen trajectories × 3 actions × 2 offsets (`-0.08`, `0.08`) | 18 |

Every version uses this exact row list. The evaluation is crossed evenly over
the three actions, with 27 rows per action.

## Domain contracts

### V1: categorical action routing

- Event type: `collision_transition`
- Relation: `pong_collision_dynamics`
- Inputs: ball state and pre-action `paddle_center_y`
- Group: `values.executed_action`
- Targets: `next_ball_vx`, `next_paddle_center_y`, `native_outcome`
- Neighbors: 8
- Required support: 2

V1 protects intervention identity by fitting separate transition neighborhoods
for `move_up`, `move_down`, and `stay`.

### V2: numeric actuator intervention

- Event type: `player_contact_transition`
- Relation: `pong_contact_dynamics`
- Inputs: V1 state plus signed `action_delta_y` and `delta_t`
- Group: none
- Targets: `next_ball_vx`, `next_paddle_center_y`, `native_outcome`
- Neighbors: 3
- Required support: 2

V2 exposes action direction, ordering, and magnitude as numeric structure. It
also changes neighborhood size and adds explicit time, so it is an engineering
iteration rather than a single-variable ablation.

### V3: post-actuation physical state

- Event type: `post_actuation_contact_transition`
- Relation: `pong_player_contact`
- Inputs: ball state, observed `post_action_paddle_center_y`, and `delta_t`
- Group: none
- Audit-only field: `executed_action`
- Targets: `next_ball_vx`, `native_outcome`
- Neighbors: 8
- Required support: 2

V3 moves the temporal boundary: actuator motion is complete before the contact
transition begins. The resulting paddle position is available to the contact
model, while the action label is excluded from learner inputs.

## Evaluation controls

Every held-out query sets:

```text
selection_mode      = deterministic
allow_exploration   = false
update_memory_state = false
return_fields       = transition_prediction, learning_state
```

The scripts probe each fresh Domain before training and reject an unexpected
cold prediction. They require accepted event acknowledgements and expose the
structured-transition model version in generated audit output. No feedback
request is sent.

## Interpretation boundaries

- Phase two is post-hoc and exploratory because its offsets were chosen after
  the phase-one boundary behavior was inspected.
- Accuracy and cross-action consistency answer different questions; a model can
  be consistently wrong.
- Removing a hard group does not by itself prove that retrieved evidence is
  shared across action labels.
- Confidence returned by the API is logged for diagnosis, not assumed to be a
  calibrated probability.
- The harness supplies the intervention schedule. It does not test autonomous
  experiment selection, long-horizon planning, or full-game control.
