| cell | fairB0s0_e4 | copy | ctx-mean8 | oracle / upper | chance | valid |
|---|---|---|---|---|---|---|
| carry gain all | 0.28 (n 350) | 0.00 | -0.47 | 0.89 | 0 = stays · 1 = true | upper bound |
| object-region visible/moving (n 60) | 86.7 [78, 95] | 68.3 | 55.0 | 98.3 | 50.0 | yes |
| object-region late/moving (n 180) | 51.1 [44, 58] | 51.7 | 51.7 | 96.1 | 50.0 | yes |
| object-region late/moving/permanence (n 60) | 50.0 [38, 63] | 50.0 | 50.0 | 100.0 | 50.0 | yes |
| object-region late/moving/shape (n 60) | 51.7 [38, 65] | 51.7 | 55.0 | 98.3 | 50.0 | yes |
| object-region late/moving/color (n 60) | 51.7 [38, 65] | 53.3 | 50.0 | 90.0 | 50.0 | yes |
| object-region late/static (n 36) | 75.0 [61, 89] | 72.2 | 100.0 | 91.7 | 50.0 | yes |
| all-token late/moving/permanence (n 60) | 90.0 [82, 97] | 53.3 | 93.3 | 53.3 | 50.0 | **판정 불가** |
| all-token gravity arc (n 36) | 33.3 [19, 50] | 11.1 | 36.1 | 44.4 | 11.1 | yes |
| all-token gravity arc/apex30 (n 12) | 0.0 [0, 0] | 0.0 | 0.0 | 16.7 | 11.1 | **판정 불가** |
| all-token gravity arc/apex48 (n 12) | 25.0 [0, 50] | 0.0 | 25.0 | 50.0 | 11.1 | yes |
| all-token gravity arc/apex66 (n 12) | 75.0 [50, 100] | 33.3 | 83.3 | 66.7 | 11.1 | yes |
| all-token gravity ledge (n 72) | 6.9 [1, 14] | 4.2 | 4.2 | 9.7 | 50.0 | **판정 불가** |
| object-region gravity arc (n 36) | 19.4 [8, 33] | 0.0 | 16.7 | 100.0 | 11.1 | yes |
| object-region gravity ledge (n 72) | 31.9 [22, 43] | 36.1 | 48.6 | 100.0 | 50.0 | yes |
| natural L1 (mean over steps, n 200) | 0.560 (copy −22.2 %) | 0.720 | 0.571 | — | — | yes |
| natural carry gain k1 | 0.54 (n 40) | 0.38 | 0.34 | 0.83 | 0 = stays · 1 = true | upper bound |
| natural carry gain all | 0.42 (n 272) | 0.21 | 0.29 | 0.78 | 0 = stays · 1 = true | upper bound |
| all-token ip1late all (n 90) | 83.3 [76, 91] | 84.4 | 78.9 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/occluded (n 30) | 63.3 [47, 80] | 63.3 | 66.7 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/visible (n 30) | 90.0 [77, 100] | 90.0 | 70.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/occluded (n 18) | 94.4 [83, 100] | 100.0 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/visible (n 12) | 100.0 [100, 100] | 100.0 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| false carry static clips: mean \|p − ctx-mean8\| (n 36; lower = less invented change) | 0.421 | copy 0.423 | 0 | oracle 0.409 · truth 0.273 | — | yes |
| false carry far cells (moving clips): mean \|p − ctx-mean8\| (n 200; lower = less invented change) | 0.347 | copy 0.413 | 0 | oracle 0.413 · truth 0.427 | — | yes |
| natural carry: steps where flow and target readout agree | 272 / 1108 moved steps | | | | | |
