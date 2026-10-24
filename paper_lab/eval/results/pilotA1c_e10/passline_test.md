| cell | pilotA1c_e10 | copy | ctx-mean8 | oracle / upper | chance | valid |
|---|---|---|---|---|---|---|
| carry gain all | 0.06 (n 350) | 0.00 | -0.47 | 0.89 | 0 = stays · 1 = true | upper bound |
| object-region visible/moving (n 60) | 65.0 [53, 77] | 68.3 | 55.0 | 98.3 | 50.0 | yes |
| object-region late/moving (n 180) | 51.1 [44, 59] | 51.7 | 51.7 | 96.1 | 50.0 | yes |
| object-region late/moving/permanence (n 60) | 51.7 [40, 65] | 50.0 | 50.0 | 100.0 | 50.0 | yes |
| object-region late/moving/shape (n 60) | 51.7 [38, 65] | 51.7 | 55.0 | 98.3 | 50.0 | yes |
| object-region late/moving/color (n 60) | 50.0 [37, 63] | 53.3 | 50.0 | 90.0 | 50.0 | yes |
| object-region late/static (n 36) | 69.4 [53, 83] | 72.2 | 100.0 | 91.7 | 50.0 | yes |
| all-token late/moving/permanence (n 60) | 50.0 [37, 62] | 53.3 | 93.3 | 53.3 | 50.0 | **판정 불가** |
| all-token gravity arc (n 36) | 36.1 [22, 53] | 11.1 | 36.1 | 44.4 | 11.1 | yes |
| all-token gravity arc/apex30 (n 12) | 16.7 [0, 42] | 0.0 | 0.0 | 16.7 | 11.1 | **판정 불가** |
| all-token gravity arc/apex48 (n 12) | 33.3 [8, 58] | 0.0 | 25.0 | 50.0 | 11.1 | yes |
| all-token gravity arc/apex66 (n 12) | 58.3 [33, 83] | 33.3 | 83.3 | 66.7 | 11.1 | yes |
| all-token gravity ledge (n 72) | 26.4 [17, 38] | 4.2 | 4.2 | 9.7 | 50.0 | **판정 불가** |
| object-region gravity arc (n 36) | 30.6 [17, 47] | 0.0 | 16.7 | 100.0 | 11.1 | yes |
| object-region gravity ledge (n 72) | 31.9 [22, 43] | 36.1 | 48.6 | 100.0 | 50.0 | yes |
| natural L1 (mean over steps, n 200) | 0.621 (copy −13.7 %) | 0.720 | 0.571 | — | — | yes |
| natural carry gain k1 | 0.60 (n 40) | 0.38 | 0.34 | 0.83 | 0 = stays · 1 = true | upper bound |
| natural carry gain all | 0.37 (n 272) | 0.21 | 0.29 | 0.78 | 0 = stays · 1 = true | upper bound |
| all-token ip1late all (n 90) | 61.1 [50, 71] | 84.4 | 78.9 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/occluded (n 30) | 50.0 [33, 67] | 63.3 | 66.7 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/visible (n 30) | 56.7 [40, 73] | 90.0 | 70.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/occluded (n 18) | 61.1 [39, 83] | 100.0 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/visible (n 12) | 100.0 [100, 100] | 100.0 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| false carry static clips: mean \|p − ctx-mean8\| (n 36; lower = less invented change) | 0.594 | copy 0.423 | 0 | oracle 0.409 · truth 0.273 | — | yes |
| false carry far cells (moving clips): mean \|p − ctx-mean8\| (n 200; lower = less invented change) | 0.522 | copy 0.413 | 0 | oracle 0.413 · truth 0.427 | — | yes |
| natural carry: steps where flow and target readout agree | 272 / 1108 moved steps | | | | | |
