# Y-OPD failure diagnosis and next-change plan

Date: 2026-06-11

This note records my current understanding of the interrupted Y-OPD run. It is
not an implementation patch. The purpose is to pin down the failure mechanism,
connect it to the code and logs, and state what I would change next.

## Bottom line

The current Y-OPD run does not show that the controller wiring is broken. It
shows the opposite: the bandit is optimizing the reward it was given. The
failure is that the reward and feasible action set bypassed the protocol gates
that were meant to prevent off-distribution training.

The key mechanism is not "temperature fakes the entropy proxy." Entropy here is
the student's own entropy at the visited states. Temperature cannot directly
fake that value. High-temperature rollout instead changes state visitation: it
pushes trajectories into bad prefixes where the student is genuinely confused.
Those states produce real high A-rate, but they are not valuable training
states. This is the proposal's M2/off-distribution fragility risk.

The run then enters a two-loop positive feedback:

1. Controller loop: high temperature visits worse states, A-rate and raw Y rise,
   and the policy shifts further toward high temperature.
2. Student loop: OPD distills on those bad high-temperature trajectories with
   very large actor updates. The student itself degrades, actor entropy rises
   broadly, and A-rate rises even more.

So the observed "Y goes up while true reward stays low" is not a proxy-forgery
artifact. It is off-distribution reward hacking by state visitation, followed by
student damage.

Scope freeze for this diagnosis: Y-OPD v1 is not RPI-OPD v4. The online
conditioning for Y-OPD v1 is only `(hat kappa, d)`: the local confidence proxy
and depth. Teacher-derived signals enter delayed yield/reward only. They do not
enter the online controller state, and there is no Mode A asynchronous teacher
feature path in Y-OPD v1. In particular, `hat z` is an RPI-OPD v4 artifact and
must not be reintroduced into the Y-OPD v1 implementation.
The frozen implementation scope is recorded in
[Y-OPD v1 design freeze](../Y-OPD-v1-design-freeze.md).

## Evidence from the logs

Run:

- Y-OPD log: `logs/y_opd_full_20260611_174711.log`
- Vanilla OPD comparison log: `logs/thunlp_opd_full_20260610_183210.log`
- Y-OPD checkpoints saved before abort: `global_step_20`, `global_step_40`
- Y-OPD latest checkpoint marker after abort: `40`
- Y-OPD run was stopped after logging `training/global_step=51`.

The early warmup phase was normal: at step 1 and 10, Y-OPD used forced
temperature 1.0 and looked close to vanilla.

| Run | Step | actor entropy | grad norm | true reward mean |
|---|---:|---:|---:|---:|
| vanilla | 1 | 0.641 | 2.010 | 0.0938 |
| Y-OPD | 1 | 0.571 | 1.908 | 0.1016 |
| vanilla | 10 | 0.611 | 1.886 | 0.1484 |
| Y-OPD | 10 | 0.545 | 1.766 | 0.1719 |

After warmup, the failure appears immediately. By step 20, actor entropy and
grad norm are already far outside the vanilla regime, while true reward drops.

| Step | Run | A-rate | B-rate | Y | P(T=1.7) | actor entropy | grad norm | true reward mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20 | vanilla | n/a | n/a | n/a | n/a | 0.779 | 1.786 | 0.2031 |
| 20 | Y-OPD | 0.388 | 0.00181 | 0.397 | 0.0807 | 4.131 | 46.716 | 0.0664 |
| 40 | vanilla | n/a | n/a | n/a | n/a | 0.789 | 1.186 | 0.2969 |
| 40 | Y-OPD | 0.487 | 0.00371 | 0.506 | 0.0898 | 5.661 | 33.609 | 0.0820 |
| 51 | vanilla | n/a | n/a | n/a | n/a | 0.686 | 0.946 | 0.3516 |
| 51 | Y-OPD | 0.509 | 0.01258 | 0.572 | 0.0956 | 5.528 | 31.027 | 0.0703 |

The late B-rate rise is not clean evidence that B is manipulable by temperature.
B-rate rises from 0.00181 to 0.01258, but the probability of the hottest arm
only moves from 0.0807 to 0.0956. The more plausible confound is student
degradation: once the student is damaged, many visited states satisfy the
"teacher disagreement" part of B. This curve must not be cited either as
positive evidence for B-manipulability or as negative evidence against it.

This would still have tripped student-health smoke gates at step 20:

- actor entropy is 5.3x vanilla, not <= 1.5x;
- grad norm is 26x vanilla, not <= 5x.

The true-reward collapse is also informative, but it should not be an
independent primary gate for Y-OPD. The method deliberately seeks confident
student errors, so rollout task reward can fall mildly even in a useful run. A
hard `true_reward >= 0.9 * vanilla` gate only becomes logically valid when the
KL rail is active; the rail says distribution shift is small, so true reward
should not collapse.

## Code-level causes

### 1. No KL rail was active

The run used `actor_rollout_ref.actor.use_kl_loss=False` via
`baselines/thunlp-opd/on_policy_distillation.sh`. More importantly, the Y-OPD
controller has no trajectory-KL rail for the behavior policy at all. The
proposal's rail is not an entropy penalty; it is a feasible-domain constraint:
compute cumulative `D_KL(q_T || q_1)` from logits, then either hard-filter
actions by budget or use a Lagrangian `U' = U - lambda D`.

Current implementation only changes sampling temperature and updates logits
from reward. There is no per-response KL accounting and no rail budget.

### 2. Temperature grid includes unsafe arms

The default Y-OPD action set is:

```text
[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
```

That is wider than the proposal's smoke/action range. The upper tail, especially
1.5 to 1.7, creates a direct path to off-distribution prefixes before any M1/M2
evidence or rail has validated that those states are useful.

### 3. `policy_top_p=0.7` is anti-exploration

The rollout worker truncates the controller distribution before sampling:

```python
policy_top_p = float(policy.get("policy_top_p", 1.0))
...
keep_idx = sorted_idx[:max(1, cutoff)]
probs = truncated / truncated.sum()
```

This is the opposite of the proposal's epsilon floor. An epsilon floor gives
every arm a known nonzero propensity and lets IPW remain meaningful. Top-p over
arms can set low-ranked arms to zero in the actual behavior policy. Even though
the current virtual-credit update still computes scores for every temperature,
the rollout data no longer has a clean per-arm propensity floor.

### 4. Threshold provenance is missing

The current A/B definitions depend on online thresholds such as `entropy_high`,
`entropy_low`, and `disagreement_high`. In the current implementation these are
derived from batch statistics/EMA inside the controller instead of from a frozen
probe-pool calibration.

That makes the scale of A/B a moving target. The observed A-rate of 0.39 to 0.51
means that roughly half the tokens are being labeled "high entropy." On a
healthy student this is suspicious: TIP-style 50% retention is a loose upper
bound, not a normal operating point. Before changing the reward, `tau_H` and
`tau_D` must come from preregistered probe-pool percentiles, and the calibration
checkpoint/time must be logged.

### 5. Raw A/B rates make A dominate Y

The controller currently uses:

```python
y = a_rate + self.b_weight * b_rate
```

with `b_weight=5.0`. This looks B-heavy but is not B-heavy in practice because
A is common and B is sparse. At step 40, A contributes 0.487 while `5B`
contributes only 0.0185. At step 51, A contributes 0.509 while `5B` contributes
0.0629. The reward is therefore mostly an A-rate reward.

This is exactly why the proposal specified per-cell normalization before
combining components. A/B must be logged and analyzed separately, then combined
as z-scored utilities if M1/M2 validates them.

### 6. Sequence-level temperature only tests global bandit

This implementation assigns one temperature to the whole response. It can only
test a global teachability bandit. It cannot test the main state-conditioned
Y-OPD v1 claim, which requires chunk/cell conditioning on `(hat kappa, d)`. A
bad result here should not be used to reject state-conditioned Y-OPD v1; it only
says this global bandit reward and action set are unsafe.

### 7. Virtual likelihood credit is not the right first estimator

The implementation uses virtual likelihood credit plus sparsemax over
temperatures. For a smoke/global-bandit setting, the sampled temperature has
known propensity. A plain per-arm estimate with epsilon-floor propensity and
IPW is simpler and cleaner. Virtual credit may still be useful later, but it is
not the first estimator to trust when debugging the reward/rail protocol.

## What I would change next

This is the plan only. I would not make these code changes until we agree on
the protocol.

### Step 0: Freeze the current run as a negative control

Do not continue the current configuration and do not run strict eval on all
current Y-OPD checkpoints as if it were a candidate method. Keep `global_step_20`
and `global_step_40`, the controller states, and the log as an M2
off-distribution negative control.

The scientific value of this run is mechanism evidence: the controller works
mechanically, but without rail and M1/M2 gates it optimizes a harmful reward.

### Step 1: Run M1-lite before any more controller training

Goal: check whether B-rate is actually manipulable by decoding action.

Immediate version:

- Use the existing Y-OPD summary logs to plot A-rate, B-rate, Y, `P(T)`,
  actor entropy, grad norm, and true reward over steps.
- Treat this only as failure-mode evidence. The post-degradation B-rate curve is
  polluted by student damage and is not clean M1 evidence.

Proper M1-lite:

- First audit the actual behavior propensities and sample counts per arm in the
  early window. The data was collected with `policy_top_p=0.7`; with 13 nearly
  uniform arms, top-p can zero out the bottom 3-4 arms, so those arms may have
  no samples at all.
- Use only the early post-warmup window, approximately steps 11-20, where the
  temperature policy is still near-uniform and the student has not yet broadly
  degraded.
- Stratify by step, then use logged propensities for IPW by sampled
  `y_opd_temp_id`.
- Because B is rare, with base rate around 0.2%, 10 steps spread across 13 arms
  may leave per-arm B counts in the single digits. If so, merge arms into three
  temperature buckets, low/mid/high, and estimate only a directional curve.
- If per-sample tensors are available in artifacts, compute A-rate/B-rate
  separately by T from that early window only.
- If not, add a diagnostic-only dump in the next smoke run. The required tensors
  already exist in `batch.batch`: `response_mask`, `entropys`,
  `teacher_on_student_log_probs`, `student_top_k_log_probs`, `old_log_probs`,
  `student_top_k_ids`, `responses`, and `y_opd_temp_id`.
- Do not update the controller in this diagnostic run; only collect enough
  samples to estimate A/B curves.

Decision rule: M1-lite may justify continuing or stopping engineering work, but
it cannot formally decide B-manipulability. The formal decision requires clean
paired same-prefix M1 estimates.

### Step 2: Run M2 on vanilla/on-distribution trajectories

Use the vanilla OPD trajectory distribution at checkpoints where the teacher
student gap is still large enough to measure learning gain. The current best
checkpoint, `global_step_220`, has weighted score 0.5944 versus teacher 0.6012,
so it is statistically low-power as the primary M2 point: there may be little
learnable signal left.

Primary M2 should use a mid-training checkpoint band, roughly steps 100-150, or
multiple checkpoints in that range if available. Keep `global_step_220` only as
an additional observation about late-stage yield-signal decay near convergence.

The test should bucket natural vanilla trajectories by candidate utility, then
match buckets on length distribution and prompt-difficulty strata before
running one-step or few-step distillation. This controls two known confounds:
long trajectories mechanically accumulate more A/B opportunities, and hard
prompts naturally create more disagreement.

After matching, measure:

- same-context KL reduction;
- held-out native-rollout loss;
- answer/verifier score on a small subset;
- repetition/format degeneration.

If high utility does not predict real improvement on the natural distribution,
do not enter online control.

### Step 3: Implement the rail as a hard constraint first

Before re-enabling online control:

- shrink the smoke action grid to `[0.7, 1.0, 1.3]` or the proposal's small
  action set;
- compute per-response cumulative trajectory KL from logits for each action;
- reject or clip actions that exceed the KL budget;
- set the KL budget by length bucket, or normalize it by response length,
  because cumulative response KL grows roughly with length;
- log trajectory KL by step, length bucket, and temperature.

Only after hard-rail behavior is stable should the Lagrangian version be added.
I would not add ad-hoc entropy penalties first; that weakens the protocol and
reintroduces reward-search ambiguity.

### Step 4: Replace arm top-p with epsilon-floor sampling

Remove `policy_top_p` over controller arms. Use:

```text
q_tilde = (1 - eps) * q + eps / |A|
```

Start with high exploration, e.g. `eps=0.5`, and anneal toward `0.1`. Log the
actual propensity for every sample. This makes per-arm estimates and IPW
well-defined.

### Step 5: Replace raw A/B sum with separated, normalized utilities

First freeze the threshold source: `tau_H` and `tau_D` must come from
preregistered probe-pool percentiles and must be logged with the calibration
checkpoint/time. Do not let a degraded online batch redefine what A/B mean.

Then, do not switch directly to `B_rate - penalty` as another hand-tuned reward.
That can also be gamed by repetitive low-entropy states.

Instead:

- log A and B as first-class separate metrics per action;
- compute per-cell or per-probe-pool z-scores;
- combine only after M2 says which component predicts real learning gain;
- report A-only, B-only, and combined versions as protocol variants, not
  opportunistic reward edits.

### Step 6: Add hard smoke abort gates

For any future online run, the primary gates should protect student health.
Abort within 20 steps if either of these break against the synchronized vanilla
run:

- `actor_entropy > 1.5 * vanilla_actor_entropy`;
- `actor_grad_norm > 5 * vanilla_actor_grad_norm`.

Also run a small eval checkpoint gate as soon as the first smoke checkpoint is
available. This catches damage that does not show up in a single rollout metric.

Use `true_reward_mean < 0.9 * vanilla_true_reward_mean` only together with an
active KL rail. Without the rail, a mild rollout-reward drop can be expected
from exposing confident wrong trajectories. With the rail, large task-reward
collapse means the feasible-domain constraint is not doing its job. Under the
health gates alone, the current run would still have stopped at step 20.

### Step 7: Only then move from global bandit to state-conditioned control

The current sequence-level controller is useful only as a global-bandit smoke.
After M1, M2, rail, epsilon-floor, and reward normalization are in place, the
next real method step is chunked/state-conditioned routing:

- chunk-level decode instead of one temperature for the whole response;
- cell features from `hat kappa` and depth only;
- teacher-derived signals in delayed reward only, not online conditioning;
- no Mode A asynchronous teacher feature path;
- per-cell action counts, ESS, and IPW;
- online-vs-paired agreement against M1 paired estimates.

Until then, results from this global bandit should be treated as a failure-mode
diagnostic, not evidence for or against the main Y-OPD v1 claim.

## Immediate no-go list

I would not do the following next:

- do not run strict eval on all current Y-OPD checkpoints as if this were a
  viable method;
- do not widen temperature above 1.3 before rail and M1/M2;
- do not reintroduce `hat z`, Mode A, or lagged teacher features into Y-OPD v1
  online conditioning;
- do not derive `tau_H`/`tau_D` from degraded online batches;
- do not cite post-degradation B-rate movement as B-manipulability evidence;
- do not add an entropy penalty or clip as the main fix;
- do not tune reward variants by hand from failed online runs;
- do not use this sequence-level result to judge state-conditioned Y-OPD v1.
