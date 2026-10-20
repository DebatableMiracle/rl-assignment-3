Shared threshold H = 0.9 * max_m mean_s F = 900.00  (best method mean = 1000.00)

| Method | Description | final eval mean ± std (seeds) | reached | not reached | threshold episode mean ± std (successful) |
|---|---|---|---|---|---|
| M1 | Discretized policy + full-return REINFORCE | 147.82 ± 44.33 | 0 | 5 | undefined |
| M2 | Neural policy + full-return REINFORCE | 150.73 ± 76.97 | 0 | 5 | undefined |
| M3 | Neural policy + leave-one-out batch baseline | 1000.00 ± 0.00 | 5 | 0 | 3168 ± 444 |
| M4 | Neural policy + moving-average baseline, beta=0.01 | 237.97 ± 355.43 | 2 | 3 | 4560 ± 339 |
| M5 | Neural policy + moving-average baseline, beta=0.1 | 815.71 ± 353.22 | 3 | 2 | 3627 ± 333 |
| M6 | Neural policy + return-to-go | 615.35 ± 507.75 | 5 | 0 | 4288 ± 1486 |

| Method | seed | F (final-20 eval mean) | episodes to reach H |
|---|---|---|---|
| M1 | 0 | 169.40 | not reached |
| M1 | 1 | 176.35 | not reached |
| M1 | 2 | 88.90 | not reached |
| M1 | 3 | 113.05 | not reached |
| M1 | 4 | 191.40 | not reached |
| M2 | 0 | 155.20 | not reached |
| M2 | 1 | 156.85 | not reached |
| M2 | 2 | 41.80 | not reached |
| M2 | 3 | 258.80 | not reached |
| M2 | 4 | 141.00 | not reached |
| M3 | 0 | 1000.00 | 3520 |
| M3 | 1 | 1000.00 | 2720 |
| M3 | 2 | 1000.00 | 3200 |
| M3 | 3 | 1000.00 | 2720 |
| M3 | 4 | 1000.00 | 3680 |
| M4 | 0 | 98.45 | not reached |
| M4 | 1 | 56.90 | 4320 |
| M4 | 2 | 75.90 | not reached |
| M4 | 3 | 873.20 | 4800 |
| M4 | 4 | 85.40 | not reached |
| M5 | 0 | 888.80 | not reached |
| M5 | 1 | 1000.00 | 3360 |
| M5 | 2 | 1000.00 | 4000 |
| M5 | 3 | 1000.00 | 3520 |
| M5 | 4 | 189.75 | not reached |
| M6 | 0 | 1000.00 | 4320 |
| M6 | 1 | 76.90 | 6240 |
| M6 | 2 | 957.50 | 3680 |
| M6 | 3 | 1000.00 | 2240 |
| M6 | 4 | 42.35 | 4960 |
