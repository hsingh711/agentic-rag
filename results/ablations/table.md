| configuration | n | relaxed acc | vs full [95% CI] | skip | false abst | model s | lite calls | what changed |
|---|---|---|---|---|---|---|---|---|
| full | 103 | 0.903 | (baseline) | 0.40 | 0.01 | 8.6 | 1.30 | current default (agentic_v2) |
| no_classifier | 103 | 0.699 | -0.204 [-0.291, -0.117] | 0.00 | 0.18 | 25.3 | 3.61 | block 1 off: always retrieve |
| no_reformulation | 103 | 0.922 | +0.019 [-0.019, +0.058] | 0.40 | 0.02 | 4.3 | 0.61 | block 2 off: search the question as typed |
| no_decompose | 103 | 0.903 | +0.000 [+0.000, +0.000] | 0.40 | 0.01 | 8.6 | 1.30 | block 2 partial: rewrite only, never split |
| no_critic | 103 | 0.893 | -0.010 [-0.029, +0.000] | 0.40 | 0.02 | 4.5 | 0.60 | blocks 3-4 off: evidence accepted unjudged |
| retries_0 | 103 | 0.893 | -0.010 [-0.029, +0.000] | 0.40 | 0.02 | 8.2 | 1.20 | block 4 off: judge once, no retry |
| retries_1 | 103 | 0.903 | +0.000 [+0.000, +0.000] | 0.40 | 0.02 | 8.4 | 1.26 | at most one retry |
| contexts_3 | 103 | 0.913 | +0.010 [-0.019, +0.049] | 0.40 | 0.02 | 6.3 | 1.30 | 3 chunks shown instead of 6 |
| contexts_10 | 103 | 0.903 | +0.000 [-0.029, +0.029] | 0.40 | 0.00 | 8.4 | 1.30 | 10 chunks shown instead of 6 |
| top_k_10 | 103 | 0.913 | +0.010 [+0.000, +0.029] | 0.40 | 0.01 | 6.2 | 1.30 | retrieve 10 chunks per query instead of 5 |
| int8 | 103 | 0.903 | +0.000 [+0.000, +0.000] | 0.40 | 0.01 | 8.6 | 1.30 | retrieve from the int8-quantized collection |
