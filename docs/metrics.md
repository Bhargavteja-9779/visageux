# Metrics: Formal Definitions

Let windows be 5s slices indexed by t. For window t, primitives are:
- rage_t ∈ ℕ (rage clicks in t), dead_t ∈ ℕ (dead clicks), stall_t ∈ ℕ (hover-stall count)
- osc_t ∈ ℕ (scroll direction changes), v_t ∈ ℝ (scroll velocity px/s)
- σ_t ≥ 0 (cursor speed std), clicks_t ∈ ℕ

We use clamped/normalized terms:
r̃age_t = min(rage_t / 2, 1)
d̃ead_t = min(dead_t / 2, 1)
s̃tall_t = min(stall_t / 1, 1)
ōsc_t   = min(osc_t / 3, 1)
j̃it_t  = sigmoid(σ_t / 200) · 1[clicks_t = 0], where sigmoid(x)=1/(1+e^{−x})

## UFI (User Friction Index) ∈ [0,1]
UFI_t = 0.35·r̃age_t + 0.20·d̃ead_t + 0.15·s̃tall_t + 0.15·ōsc_t + 0.15·j̃it_t
Bounds follow from convex combination of [0,1] terms.

## RCS (Reading Confidence Score) ∈ [0,1]
RCS_t = clamp(0..1){ 0.6·sigmoid(−σ_t / 150) + 0.4·sigmoid(v_t / 200) } · (1 − ōsc_t)

High when jitter is low and scrolling is smooth forward; penalized by oscillations.

## MIV (Micro-interaction Velocity) [seconds]
Let τ_stop be last time within [t−1s,t] where |v|<20 px/s (scroll “stopped”).
Let τ_click be first click time in window t (if any). Then:
MIV_t = τ_click − τ_stop, undefined if no click. Report per-session median of defined MIV_t.
