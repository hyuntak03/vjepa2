| cell | ext_ar_vw_shortctx_e30_c8 | copy | ctx-mean8 | oracle / upper | chance | valid |
|---|---|---|---|---|---|---|
| carry gain all | 0.10 (n 700) | -0.02 | -0.50 | 0.90 | 0 = stays · 1 = true | upper bound |
| object-region visible/moving (n 120) | 72.5 [64, 80] | 68.3 | 56.7 | 99.2 | 50.0 | yes |
| object-region late/moving (n 360) | 52.6 [48, 58] | 51.1 | 50.8 | 95.3 | 50.0 | yes |
| object-region late/moving/permanence (n 120) | 50.0 [41, 59] | 50.0 | 50.0 | 100.0 | 50.0 | yes |
| object-region late/moving/shape (n 120) | 53.3 [45, 62] | 51.7 | 52.5 | 95.0 | 50.0 | yes |
| object-region late/moving/color (n 120) | 54.6 [46, 64] | 51.7 | 50.0 | 90.8 | 50.0 | yes |
| object-region late/static (n 72) | 100.0 [100, 100] | 63.9 | 100.0 | 90.3 | 50.0 | yes |
| all-token late/moving/permanence (n 120) | 94.2 [90, 98] | 53.3 | 94.2 | 53.3 | 50.0 | **판정 불가** |
| all-token gravity arc (n 72) | 37.5 [26, 49] | 9.7 | 37.5 | 37.5 | 11.1 | yes |
| all-token gravity arc/apex30 (n 24) | 0.0 [0, 0] | 0.0 | 0.0 | 8.3 | 11.1 | **판정 불가** |
| all-token gravity arc/apex48 (n 24) | 41.7 [25, 62] | 0.0 | 37.5 | 45.8 | 11.1 | yes |
| all-token gravity arc/apex66 (n 24) | 70.8 [50, 88] | 29.2 | 75.0 | 58.3 | 11.1 | yes |
| all-token gravity ledge (n 144) | 10.4 [6, 16] | 5.6 | 4.9 | 11.8 | 50.0 | **판정 불가** |
| object-region gravity arc (n 72) | 20.8 [12, 31] | 0.0 | 16.7 | 100.0 | 11.1 | yes |
| object-region gravity ledge (n 144) | 29.9 [22, 38] | 31.2 | 41.0 | 100.0 | 50.0 | yes |
| natural L1 (mean over steps, n 200) | 0.562 (copy −21.8 %) | 0.720 | 0.571 | — | — | yes |
| natural carry gain k1 | 0.60 (n 40) | 0.38 | 0.34 | 0.83 | 0 = stays · 1 = true | upper bound |
| natural carry gain all | 0.45 (n 272) | 0.21 | 0.29 | 0.78 | 0 = stays · 1 = true | upper bound |
| all-token ip1late all (n 180) | 85.6 [80, 91] | 87.2 | 81.1 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/occluded (n 60) | 71.7 [60, 83] | 71.7 | 71.7 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/visible (n 60) | 88.3 [80, 97] | 90.0 | 71.7 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/occluded (n 30) | 93.3 [83, 100] | 100.0 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/visible (n 30) | 100.0 [100, 100] | 100.0 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| false carry static clips: mean \|p − ctx-mean8\| (n 72; lower = less invented change) | 0.421 | copy 0.414 | 0 | oracle 0.400 · truth 0.280 | — | yes |
| false carry far cells (moving clips): mean \|p − ctx-mean8\| (n 400; lower = less invented change) | 0.370 | copy 0.413 | 0 | oracle 0.413 · truth 0.427 | — | yes |
| natural carry: steps where flow and target readout agree | 272 / 1108 moved steps | | | | | |
