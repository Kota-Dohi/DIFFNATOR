# DiffNator: Generating Structured Explanations of Time‑Series Differences

> Notes: The details that the paper states are “provided on the project page”—such as the duration bounds of component functions, the sampling ranges for each parameter, and the offset/ratio modification rules—are organized as dedicated sections below in Data → Component Functions and Data → Parameter Ranges & Modification Rules.

<p align="center">
  <img src="docs/teaser.png" alt="DiffNator Teaser (reference vs target with JSON output)" width="75%">
</p>

## Summary

**DiffNator** is a framework that explains differences between two time series as a **structured list of JSON objects**. Each object defines the type of difference (Type1 or Type2), the type of component function that corresponds to certain physical phenomena (e.g., LINEAR_INCREASE, QUADRATIC_INCREASE, TRIANGULAR_WAVE, SPIKE, ...), the start and end index of the difference, etc..

* **Task**: Explain pairwise differences (reference vs target) in **list of JSON**.
* **Backbone**: Frozen LLM (Mistral‑7B‑Instruct) conditioned by learnable **time‑series encoders** (Informer / Dilated‑TCN) via **difference‑aware merging**.
* **Outputs**: Validated JSON adhering to a unified schema; easy to evaluate and integrate downstream.

> This repository ships configs and specification tables of the component functions and the method for generating synthetic pairs of reference and target time-series pairs from the original time series.
---

## Data

DiffNator works with real IoT time series and with synthetic perturbations.

### TORI: Time‑series Observations of Real‑world IoT

* **Description**: 1,690,485 real‑world sensor time series used as source corpus.
* **Splits**: 1,000 validation + 1,000 test randomly sampled; remaining for on‑the‑fly training sampling.

> **Note**: The paper uses **z‑normalized** base series, resampled/truncated to length **T=300**.

### Component Functions (taxonomy)

> *This corresponds to Table 2 in the paper; capitalized names are for readability — actual JSON uses UPPERCASE.*

| Category    | Component functions                                                                                                                                                                                                                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Trend       | Linear\_increase, Linear\_decrease, Quadratic\_increase, Quadratic\_decrease, Cubic\_increase, Cubic\_decrease, Exponential\_growth, Inverted\_exponential\_growth, Exponential\_decay, Inverted\_exponential\_decay, Log\_increase, Log\_decrease, Sigmoid, Inverted\_sigmoid, Gaussian, Inverted\_gaussian |
| Periodic    | Sinusoidal, Sawtooth, Square\_wave, Triangle\_wave                                                                                                                                                                                                                                                           |
| Fluctuation | Gaussian\_noise, Laplace\_noise                                                                                                                                                                                                                                                                              |
| Event       | Spike, Drop, Positive\_step, Negative\_step, Positive\_pulse, Negative\_pulse                                                                                                                                                                                                                                |

## Mathematical Definitions (by component)

Let the active interval be indices `t in [s, e]` (inclusive). Define its length `L = e - s` and the normalized time `u = (t - s) / L` in `[0, 1]`. Each component adds an increment `Δ(t)` to the current series `x(t)` as specified below. For components that **persist** after the interval (e.g., trends or steps), also apply `x(t) <- x(t) + Δ(e)` for all `t > e`.

### Trend

* **Linear increase / decrease** — slope `C` over `[s, e]`: `Δ(t) = C * u` for `t in [s, e]`. (Decrease: use `C < 0`.)
  *Type-2 changeable param*: `slope`.

* **Polynomial (degree n)** — `p(u) = sum_{i=0..n} a_i * u^i`, then `Δ(t) = p(u) - p(0)`.
  Quadratic increase/decrease: `p(u) = ± A * u^2`.
  Cubic increase/decrease: `p(u) = ± A * u^3`.
  *Type-2 changeable param*: `amplitude` (`A`).

* **Logarithmic** — shift `sigma > 0`, amplitude `A` (normalized so `Δ` reaches `A` at `u = 1`):
  `Δ(t) = A * ( log(u + sigma) - log(sigma) ) / ( log(1 + sigma) - log(sigma) )`.
  *Type-2 changeable param*: `amplitude` (`A`).

* **Exponential trend** (time constant `tau > 0`) —
  Upward: `Δ(t) = A * (1 - exp(-u / tau))`.
  Downward: `Δ(t) = A * exp(-u / tau) - A`.

* **Exponential growth / inverted growth** — rate `g > 0`:
  `Δ(t) = A * (exp(g * u) - 1)`, inverted `Δ_inv(t) = -A * (exp(g * u) - 1)`.
  *Type-2 changeable params*: `amplitude` (`A`), `growth_rate` (`g`).

* **Exponential decay / inverted decay** — rate `lambda > 0`:
  `Δ(t) = A * (exp(-lambda * u) - 1)`, inverted `Δ_inv(t) = -A * (exp(-lambda * u) - 1)`.
  *Type-2 changeable params*: `amplitude` (`A`), `decay_rate` (`lambda`).

* **Sigmoid / inverted sigmoid** — amplitude `A > 0`, steepness `k > 0`, midpoint `c in (0,1)`, offset `d`:
  `s(u) = A / (1 + exp(-k * (u - c))) + d`, then `Δ(t) = s(u) - s(0)`.
  Inverted: `s_inv(u) = -A / (1 + exp(-k * (u - c))) + d`, `Δ(t) = s_inv(u) - s_inv(0)`.
  *Type-2 changeable params*: `amplitude` (`A`), `steepness` (`k`), `midpoint` (`c`).

* **Gaussian / inverted Gaussian** — amplitude `A > 0`, center `b in [0,1]`, width `w > 0`, offset `d`:
  `g(u) = A * exp(-((u - b)/w)^2) + d`, then `Δ(t) = g(u) - g(0)`.
  Inverted: `g_inv(u) = -A * exp(-((u - b)/w)^2) + d`, `Δ(t) = g_inv(u) - g_inv(0)`.
  *Type-2 changeable params*: `amplitude` (`A`), `center` (`b`), `width` (`w`).

### Periodic (frequency `f` cycles over the interval, common phase `phi`)

* **Sinusoidal** — `Δ(t) = A * sin(2*pi*f*u + phi)`.
  *Type-2 changeable params*: `amplitude` (`A`), `frequency` (`f`).

* **Sawtooth** — define `saw(x) = 2*(x - floor(x)) - 1`, then `Δ(t) = A * saw(f*u + phi/(2*pi))`.
  *Type-2 changeable params*: `amplitude` (`A`), `frequency` (`f`).

* **Square** — `Δ(t) = A * sign( sin(2*pi*f*u + phi) )`.
  *Type-2 changeable params*: `amplitude` (`A`), `frequency` (`f`).

* **Triangle** — define `tri(x) = 2*abs( 2*(x - floor(x)) - 1 ) - 1`, then `Δ(t) = A * tri(f*u + phi/(2*pi))`.
  *Type-2 changeable params*: `amplitude` (`A`), `frequency` (`f`).

### Fluctuation

* **Gaussian noise** (iid, scale `sigma`): `Δ(t) ~ Normal(0, sigma^2)` for `t in [s, e]`.
  *Type-2 changeable param*: `scale` (`sigma`).

* **Laplace noise** (iid, scale `b`): `Δ(t) ~ Laplace(0, b)` for `t in [s, e]`.
  *Type-2 changeable param*: `scale` (`b`).

### Event

* **Spike / Drop** — `Δ(t) = ±|M|` for `t in [s, e]` and `0` otherwise.
  *Type-2 changeable param*: `magnitude` (`M`).

* **Positive / Negative step** (linear ramp on `[s, e]`, persists after `e`):
  Positive: `Δ_pos(t) = |M| * u` for `t in [s, e]`, `= |M|` for `t > e`, `= 0` for `t < s`.
  Negative: `Δ_neg(t) = -|M| * u` for `t in [s, e]`, `= -|M|` for `t > e`, `= 0` for `t < s`.
  *Type-2 changeable param*: `magnitude` (`M`).

* **Positive / Negative pulse** (temporary within `[s, e]` only): `Δ(t) = ±|M|` for `t in [s, e]` and `0` otherwise.
  *Type-2 changeable param*: `magnitude` (`M`).

> Parameter names follow the executors: `amplitude`, `frequency`, `phase(=phi)`, `slope` (`C`), `tau`, `growth_rate`/`decay_rate`, `steepness` (`k`), `midpoint` (`c`), `center` (`b`), `width` (`w`), `scale`/`std` (`sigma`), `magnitude` (`M`).

---
### Duration Bounds (per component function)
* When `relative_to_length: true`, `min_duration`/`max_duration` are fractions of `T` (e.g., `0.25` → `0.25 * T`).
* When `false`, values are absolute timesteps.

#### Duration bound table

|               Function | Category    | min\_duration | max\_duration | relative\_to\_length |
| ----------------------------: | :---------- | ------------: | ------------: | :------------------- | 
|              LINEAR\_INCREASE | TREND       |          0.25 |          0.75 | true                 | 
|              LINEAR\_DECREASE | TREND       |          0.25 |          0.75 | true                 |
|           QUADRATIC\_INCREASE | TREND       |          0.25 |          0.75 | true                 | 
|           QUADRATIC\_DECREASE | TREND       |          0.25 |          0.75 | true                 |
|               CUBIC\_INCREASE | TREND       |          0.25 |          0.75 | true                 | 
|               CUBIC\_DECREASE | TREND       |          0.25 |          0.75 | true                 | 
|           EXPONENTIAL\_GROWTH | TREND       |          0.25 |          0.75 | true                 | 
| INVERTED\_EXPONENTIAL\_GROWTH | TREND       |          0.25 |          0.75 | true                 | 
|            EXPONENTIAL\_DECAY | TREND       |          0.25 |          0.75 | true                 | 
|  INVERTED\_EXPONENTIAL\_DECAY | TREND       |          0.25 |          0.75 | true                 |
|                 LOG\_INCREASE | TREND       |          0.25 |          0.75 | true                 | 
|                 LOG\_DECREASE | TREND       |          0.25 |          0.75 | true                 |
|                       SIGMOID | TREND       |          0.25 |          0.75 | true                 | 
|             INVERTED\_SIGMOID | TREND       |          0.25 |          0.75 | true                 | 
|                      GAUSSIAN | TREND       |          0.25 |          0.75 | true                 |
|            INVERTED\_GAUSSIAN | TREND       |          0.25 |          0.75 | true                 |
|                    SINUSOIDAL | PERIODIC    |          0.25 |          0.75 | true                 |
|                      SAWTOOTH | PERIODIC    |          0.25 |          0.75 | true                 |
|                  SQUARE\_WAVE | PERIODIC    |          0.25 |          0.75 | true                 |
|                TRIANGLE\_WAVE | PERIODIC    |          0.25 |          0.75 | true                 |
|               GAUSSIAN\_NOISE | FLUCTUATION |          0.25 |          0.75 | true                 |
|                LAPLACE\_NOISE | FLUCTUATION |          0.25 |          0.75 | true                 |
|                         SPIKE | EVENT       |             1 |             3 | false                |
|                          DROP | EVENT       |             1 |             3 | false                | 
|                POSITIVE\_STEP | EVENT       |             1 |             3 | false                | 
|                NEGATIVE\_STEP | EVENT       |             1 |             3 | false                |
|               POSITIVE\_PULSE | EVENT       |            10 |            20 | false                |
|               NEGATIVE\_PULSE | EVENT       |            10 |            20 | false                |


### Parameter Ranges & Modification Rules
We specify, per component and parameter, both the **base range** and the **Type‑2 modification rule** (offset or ratio) with its range.

### Trend

#### Linear / Polynomial

| Component           | Param     | Base range | Type-2 mod | Mod range |
| ------------------- | --------- | ---------: | ---------- | --------: |
| LINEAR\_INCREASE    | slope     |    1.0–3.0 | ratio      |   1.5–2.0 |
| LINEAR\_DECREASE    | slope     |    1.0–3.0 | ratio      |   1.5–2.0 |
| QUADRATIC\_INCREASE | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
| QUADRATIC\_DECREASE | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
| CUBIC\_INCREASE     | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
| CUBIC\_DECREASE     | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |

#### Exponential (decay/growth) and Log

| Component                     | Param        | Base range | Type-2 mod | Mod range |
| ----------------------------- | ------------ | ---------: | ---------- | --------: |
| EXPONENTIAL\_DECAY            | amplitude    |    1.0–3.0 | ratio      |   1.5–2.0 |
|                               | decay\_rate  |    3.0–5.0 | ratio      |   2.0–3.0 |
| INVERTED\_EXPONENTIAL\_DECAY  | amplitude    |    1.0–3.0 | ratio      |   1.5–2.0 |
|                               | decay\_rate  |    3.0–5.0 | ratio      |   2.0–3.0 |
| EXPONENTIAL\_GROWTH           | amplitude    |    1.0–3.0 | ratio      |   1.5–2.0 |
|                               | growth\_rate |    1.0–2.0 | ratio      |   1.2–1.5 |
| INVERTED\_EXPONENTIAL\_GROWTH | amplitude    |    1.0–3.0 | ratio      |   1.5–2.0 |
|                               | growth\_rate |    1.0–2.0 | ratio      |   1.2–1.5 |
| LOG\_INCREASE                 | amplitude    |    1.0–3.0 | ratio      |   1.5–2.0 |
| LOG\_DECREASE                 | amplitude    |    1.0–3.0 | ratio      |   1.5–2.0 |

#### Sigmoid / Gaussian

| Component          | Param     | Base range | Type-2 mod | Mod range |
| ------------------ | --------- | ---------: | ---------- | --------: |
| SIGMOID            | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
|                    | steepness |   5.0–10.0 | ratio      |   2.0–3.0 |
|                    | midpoint  |    0.3–0.7 | offset     |   0.1–0.2 |
| INVERTED\_SIGMOID  | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
|                    | steepness |   5.0–10.0 | ratio      |   2.0–3.0 |
|                    | midpoint  |    0.3–0.7 | offset     |   0.1–0.2 |
| GAUSSIAN           | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
|                    | center    |    0.4–0.6 | offset     |   0.1–0.2 |
|                    | width     |    0.1–0.3 | ratio      |   1.5–2.0 |
| INVERTED\_GAUSSIAN | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
|                    | center    |    0.4–0.6 | offset     |   0.1–0.2 |
|                    | width     |    0.1–0.3 | ratio      |   1.5–2.0 |

---

### Periodic

| Component      | Param     | Base range | Type-2 mod | Mod range |
| -------------- | --------- | ---------: | ---------- | --------: |
| SINUSOIDAL     | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
|                | frequency |    2.0–8.0 | ratio      |   1.1–1.3 |
| SAWTOOTH       | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
|                | frequency |    2.0–8.0 | ratio      |   1.1–1.3 |
| SQUARE\_WAVE   | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
|                | frequency |    2.0–8.0 | ratio      |   1.1–1.3 |
| TRIANGLE\_WAVE | amplitude |    1.0–3.0 | ratio      |   1.5–2.0 |
|                | frequency |    2.0–8.0 | ratio      |   1.1–1.3 |

---

### Fluctuation

| Component       | Param | Base range | Type-2 mod | Mod range |
| --------------- | ----- | ---------: | ---------- | --------: |
| GAUSSIAN\_NOISE | scale |    0.5–1.0 | ratio      |   1.5–2.0 |
| LAPLACE\_NOISE  | scale |    0.5–1.0 | ratio      |   1.5–2.0 |

---

### Event

| Component       | Param     | Base range | Type-2 mod | Mod range |
| --------------- | --------- | ---------: | ---------- | --------: |
| SPIKE           | magnitude |    0.5–3.0 | ratio      |   1.5–2.0 |
| DROP            | magnitude |    0.5–3.0 | ratio      |   1.5–2.0 |
| POSITIVE\_STEP  | magnitude |    0.5–3.0 | ratio      |   1.5–2.0 |
| NEGATIVE\_STEP  | magnitude |  −3.0––0.5 | ratio      |   1.5–2.0 |
| POSITIVE\_PULSE | magnitude |    0.5–3.0 | ratio      |   1.5–2.0 |
| NEGATIVE\_PULSE | magnitude |  −3.0––0.5 | ratio      |   1.5–2.0 |

---

## JSON Schema

All model outputs must conform to this schema (also in `diffnator/schema/diffnator.schema.json`).

```jsonc
[
  {
    "type": "TYPE1 or TYPE2",
    "func": "<UPPERCASE_COMPONENT_FUNCTION_NAME>",
    "start": "<START_INDEX>",
    "end": "<END_INDEX>",
    "presence": "null or PRESENT or ABSENT",   // Type1 only
    "param": "null or <PARAM_NAME>",            // Type2 only
    "magnitude": "null or LARGER or SMALLER"   // Type2 only
  }
]
```

---

## Model

* **Encoders**: Informer (3 enc layers, hidden 512) or Dilated‑TCN (8 conv layers, k=3, hidden 512)
* **Merging**:

  * *mean*: mean of per‑time hidden differences `D_t = H_tgt(t) − H_ref(t)`
  * *attn*: `J=6` learnable queries; softmax attention over `D_t`; concatenated
* **Adapter**: 2‑layer MLP + GELU + LayerNorm → project to LLM dim 4096
* **LLM**: `Mistral‑7B‑Instruct‑v0.1` (frozen)
* **Prompting**: `inst_part1` (task header) + \[TS embedding] + `inst_part2` (schema contract) + `question`

### Prompt template
```text
inst_part1:
Task: You are a time-series difference explainer. Input:

[TS_EMBEDDING_INSERTED_HERE]

inst_part2:
Output strictly valid JSON list of objects with fields: type, func, start, end,
presence (Type1 only), param & magnitude (Type2 only).

question:
Explain the differences between the two time-series data.
```

---

## Training
* Optimizer: **AdamW**, lr = 1e‑4, batch size = 2
* Scheduler: cosine annealing with warm restarts
* Epochs: 500
* Data: **on‑the‑fly** synthesis, 10,000 pairs/epoch
  
---

## Evaluation

Metrics implemented per paper:
* **Field accuracies**: `type`, `func`, `presence` (Type1), `param` & `magnitude` (Type2)
* **Interval IoU** (percent)
* **Match accuracy** (all fields correct + IoU ≥ 0.8), overall and by category
* **OPR/UPR** (for K\_max>1)

Alignment rule (K\_max>1): category‑first greedy; fallback to position‑based; remaining are unmatched

---

## Citation

The citation information is coming soon.

---

