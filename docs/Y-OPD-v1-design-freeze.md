# Y-OPD v1 design freeze

Date: 2026-06-11

This page freezes the intended Y-OPD v1 scope so implementation work does not
drift back into RPI-OPD v4.

## Frozen definition

Y-OPD v1 means:

- online state conditioning on `(hat kappa, d)` only;
- yield reward from delayed teacher/student diagnostics;
- no Mode A asynchronous teacher feature pipeline;
- no lagged teacher feature `hat z` in the online controller;
- KL rail before online control;
- epsilon-floor exploration, not arm top-p truncation;
- A/B logged separately, calibrated from probe-pool thresholds, and normalized
  before reward composition.

Teacher-derived quantities may be used to score completed samples and update
delayed reward estimates. They must not be added as online routing features for
Y-OPD v1.

## Controller Inputs

Allowed online features:

- `hat kappa`: the current student-side confidence/uncertainty proxy used for
  cell assignment;
- `d`: depth or position bucket.

Forbidden online features:

- `hat z`;
- teacher disagreement from a lagged teacher pass;
- Mode A asynchronous scoring outputs;
- any feature that requires restoring the RPI-OPD v4 teacher-side online
  pipeline.

## Reward And Calibration

The yield reward is not a free-form hand-tuned reward search. Before online
control resumes:

- `tau_H` and `tau_D` must be calibrated from a fixed probe pool;
- the calibration checkpoint and time must be logged;
- A-rate and B-rate must be persisted as separate metrics;
- reward composition should use normalized per-cell utilities after M1/M2
  validates which components predict learning gain.

Post-degradation B-rate movement from the failed online run is a contaminated
signal. It must not be used as evidence for or against B manipulability.

## Feasible Domain

Y-OPD v1 online control requires a rail before the controller is allowed to
change the training distribution:

- use a small smoke action grid, e.g. `[0.7, 1.0, 1.3]`;
- compute per-response cumulative `D_KL(q_T || q_1)` from rollout logits;
- reject or clip actions outside the KL budget;
- set the KL budget by length bucket, or normalize it by response length;
- log trajectory KL by step, length bucket, and temperature.

The Lagrangian rail can come later. The first implementation should be a hard
constraint so failure attribution stays simple.

## Experiment Order

The execution order is part of the design:

1. Freeze the failed run as an M2 off-distribution negative control.
2. Audit early-window propensities and sample counts, then run M1-lite on early
   post-warmup data only with step stratification, IPW from logged
   propensities, and low/mid/high temperature buckets if per-arm B counts are
   too sparse.
3. Run M2 on vanilla/on-distribution trajectories from mid-training checkpoints
   where teacher-student gap remains measurable, with buckets matched on length
   distribution and prompt-difficulty strata.
4. Add the hard KL rail.
5. Replace arm top-p with epsilon-floor exploration.
6. Re-enable online smoke with student-health gates.
7. Only then move to chunked/state-conditioned routing using `(hat kappa, d)`.

The current sequence-level temperature controller is a global-bandit smoke
test. It is not evidence for or against the state-conditioned Y-OPD v1 claim.
