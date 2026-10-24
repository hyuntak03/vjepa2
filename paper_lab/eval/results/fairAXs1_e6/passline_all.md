| cell | fairAXs1_e6 | copy | ctx-mean8 | oracle / upper | chance | valid |
|---|---|---|---|---|---|---|
| carry gain all | 0.30 (n 700) | -0.02 | -0.50 | 0.90 | 0 = stays · 1 = true | upper bound |
| object-region visible/moving (n 120) | 83.3 [77, 90] | 68.3 | 56.7 | 99.2 | 50.0 | yes |
| object-region late/moving (n 360) | 51.1 [46, 56] | 51.1 | 50.8 | 95.3 | 50.0 | yes |
| object-region late/moving/permanence (n 120) | 50.0 [41, 59] | 50.0 | 50.0 | 100.0 | 50.0 | yes |
| object-region late/moving/shape (n 120) | 52.5 [43, 61] | 51.7 | 52.5 | 95.0 | 50.0 | yes |
| object-region late/moving/color (n 120) | 50.8 [42, 60] | 51.7 | 50.0 | 90.8 | 50.0 | yes |
| object-region late/static (n 72) | 79.2 [69, 89] | 63.9 | 100.0 | 90.3 | 50.0 | yes |
| all-token late/moving/permanence (n 120) | 89.2 [83, 94] | 53.3 | 94.2 | 53.3 | 50.0 | **판정 불가** |
| all-token gravity arc (n 72) | 29.2 [19, 40] | 9.7 | 37.5 | 37.5 | 11.1 | yes |
| all-token gravity arc/apex30 (n 24) | 0.0 [0, 0] | 0.0 | 0.0 | 8.3 | 11.1 | **판정 불가** |
| all-token gravity arc/apex48 (n 24) | 16.7 [4, 33] | 0.0 | 37.5 | 45.8 | 11.1 | yes |
| all-token gravity arc/apex66 (n 24) | 70.8 [50, 88] | 29.2 | 75.0 | 58.3 | 11.1 | yes |
| all-token gravity ledge (n 144) | 7.6 [3, 12] | 5.6 | 4.9 | 11.8 | 50.0 | **판정 불가** |
| object-region gravity arc (n 72) | 27.8 [18, 39] | 0.0 | 16.7 | 100.0 | 11.1 | yes |
| object-region gravity ledge (n 144) | 27.8 [21, 35] | 31.2 | 41.0 | 100.0 | 50.0 | yes |
| natural L1 (mean over steps, n 200) | 0.560 (copy −22.2 %) | 0.720 | 0.571 | — | — | yes |
| natural carry gain k1 | 0.52 (n 40) | 0.38 | 0.34 | 0.83 | 0 = stays · 1 = true | upper bound |
| natural carry gain all | 0.40 (n 272) | 0.21 | 0.29 | 0.78 | 0 = stays · 1 = true | upper bound |
| all-token ip1late all (n 180) | 87.2 [82, 92] | 87.2 | 81.1 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/occluded (n 60) | 75.0 [63, 87] | 71.7 | 71.7 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/visible (n 60) | 90.0 [82, 97] | 90.0 | 71.7 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/occluded (n 30) | 93.3 [83, 100] | 100.0 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/visible (n 30) | 100.0 [100, 100] | 100.0 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| false carry static clips: mean \|p − ctx-mean8\| (n 72; lower = less invented change) | 0.418 | copy 0.414 | 0 | oracle 0.400 · truth 0.280 | — | yes |
| false carry far cells (moving clips): mean \|p − ctx-mean8\| (n 400; lower = less invented change) | 0.349 | copy 0.413 | 0 | oracle 0.413 · truth 0.427 | — | yes |
| natural carry: steps where flow and target readout agree | 272 / 1108 moved steps | | | | | |
