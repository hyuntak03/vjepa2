| cell | pilotA1c_e10_tsp | copy | ctx-mean8 | oracle / upper | chance | valid |
|---|---|---|---|---|---|---|
| carry gain all | 0.03 (n 350) | 0.01 | -0.40 | 0.84 | 0 = stays · 1 = true | upper bound |
| object-region visible/moving (n 60) | 55.0 [42, 67] | 55.0 | 51.7 | 95.0 | 50.0 | yes |
| object-region late/moving (n 180) | 50.6 [43, 58] | 50.6 | 51.1 | 98.3 | 50.0 | yes |
| object-region late/moving/permanence (n 60) | 50.0 [38, 63] | 50.0 | 50.0 | 100.0 | 50.0 | yes |
| object-region late/moving/shape (n 60) | 50.0 [37, 63] | 51.7 | 50.0 | 98.3 | 50.0 | yes |
| object-region late/moving/color (n 60) | 51.7 [38, 63] | 50.0 | 53.3 | 96.7 | 50.0 | yes |
| object-region late/static (n 36) | 86.1 [75, 97] | 69.4 | 100.0 | 91.7 | 50.0 | yes |
| all-token late/moving/permanence (n 60) | 85.0 [75, 93] | 53.3 | 93.3 | 55.0 | 50.0 | **판정 불가** |
| all-token gravity arc (n 36) | 22.2 [8, 36] | 11.1 | 33.3 | 72.2 | 11.1 | yes |
| all-token gravity arc/apex30 (n 12) | 0.0 [0, 0] | 0.0 | 0.0 | 66.7 | 11.1 | yes |
| all-token gravity arc/apex48 (n 12) | 16.7 [0, 42] | 0.0 | 33.3 | 58.3 | 11.1 | yes |
| all-token gravity arc/apex66 (n 12) | 50.0 [25, 75] | 33.3 | 66.7 | 91.7 | 11.1 | yes |
| all-token gravity ledge (n 72) | 5.6 [1, 11] | 2.8 | 5.6 | 15.3 | 50.0 | **판정 불가** |
| object-region gravity arc (n 36) | 0.0 [0, 0] | 0.0 | 0.0 | 97.2 | 11.1 | yes |
| object-region gravity ledge (n 72) | 43.1 [32, 54] | 36.1 | 44.4 | 100.0 | 50.0 | yes |
| natural L1 (mean over steps, n 200) | 0.288 (copy −22.3 %) | 0.371 | 0.303 | — | — | yes |
| natural carry gain k1 | 0.56 (n 42) | 0.39 | 0.48 | 0.75 | 0 = stays · 1 = true | upper bound |
| natural carry gain all | 0.37 (n 277) | 0.30 | 0.28 | 0.81 | 0 = stays · 1 = true | upper bound |
| all-token ip1late all (n 90) | 76.7 [68, 84] | 82.2 | 76.7 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/occluded (n 30) | 53.3 [37, 70] | 63.3 | 66.7 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/visible (n 30) | 76.7 [60, 90] | 90.0 | 63.3 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/occluded (n 18) | 100.0 [100, 100] | 88.9 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/visible (n 12) | 100.0 [100, 100] | 100.0 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| false carry static clips: mean \|p − ctx-mean8\| (n 36; lower = less invented change) | 0.122 | copy 0.196 | 0 | oracle 0.187 · truth 0.119 | — | yes |
| false carry far cells (moving clips): mean \|p − ctx-mean8\| (n 200; lower = less invented change) | 0.104 | copy 0.181 | 0 | oracle 0.181 · truth 0.183 | — | yes |
| natural carry: steps where flow and target readout agree | 277 / 1108 moved steps | | | | | |
