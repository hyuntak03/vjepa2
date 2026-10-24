| cell | pilotA1_e10_tsp | copy | ctx-mean8 | oracle / upper | chance | valid |
|---|---|---|---|---|---|---|
| carry gain all | 0.21 (n 700) | -0.00 | -0.46 | 0.84 | 0 = stays · 1 = true | upper bound |
| object-region visible/moving (n 120) | 60.8 [52, 69] | 57.5 | 51.7 | 95.0 | 50.0 | yes |
| object-region late/moving (n 360) | 51.9 [47, 57] | 51.1 | 51.4 | 96.1 | 50.0 | yes |
| object-region late/moving/permanence (n 120) | 50.0 [41, 59] | 50.0 | 50.0 | 100.0 | 50.0 | yes |
| object-region late/moving/shape (n 120) | 52.5 [44, 61] | 51.7 | 50.8 | 95.0 | 50.0 | yes |
| object-region late/moving/color (n 120) | 53.3 [44, 62] | 51.7 | 53.3 | 93.3 | 50.0 | yes |
| object-region late/static (n 72) | 76.4 [65, 86] | 65.3 | 100.0 | 93.1 | 50.0 | yes |
| all-token late/moving/permanence (n 120) | 85.8 [79, 92] | 54.2 | 92.5 | 55.0 | 50.0 | **판정 불가** |
| all-token gravity arc (n 72) | 18.1 [10, 28] | 11.1 | 36.1 | 70.8 | 11.1 | yes |
| all-token gravity arc/apex30 (n 24) | 0.0 [0, 0] | 0.0 | 0.0 | 62.5 | 11.1 | yes |
| all-token gravity arc/apex48 (n 24) | 12.5 [0, 25] | 0.0 | 41.7 | 70.8 | 11.1 | yes |
| all-token gravity arc/apex66 (n 24) | 41.7 [25, 62] | 33.3 | 66.7 | 79.2 | 11.1 | yes |
| all-token gravity ledge (n 144) | 4.2 [1, 8] | 4.9 | 4.9 | 15.3 | 50.0 | **판정 불가** |
| object-region gravity arc (n 72) | 0.0 [0, 0] | 0.0 | 1.4 | 98.6 | 11.1 | yes |
| object-region gravity ledge (n 144) | 39.6 [32, 48] | 31.9 | 42.4 | 100.0 | 50.0 | yes |
| natural L1 (mean over steps, n 200) | 0.290 (copy −21.8 %) | 0.371 | 0.303 | — | — | yes |
| natural carry gain k1 | 0.53 (n 42) | 0.39 | 0.48 | 0.75 | 0 = stays · 1 = true | upper bound |
| natural carry gain all | 0.41 (n 277) | 0.30 | 0.28 | 0.81 | 0 = stays · 1 = true | upper bound |
| all-token ip1late all (n 180) | 80.0 [74, 86] | 84.4 | 78.3 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/occluded (n 60) | 65.0 [53, 77] | 68.3 | 70.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late moving/visible (n 60) | 78.3 [68, 88] | 88.3 | 65.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/occluded (n 30) | 93.3 [83, 100] | 93.3 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| all-token ip1late static/visible (n 30) | 100.0 [100, 100] | 100.0 | 100.0 | nan | 50.0 | no oracle (no object labels) — vs predictor-free baselines only |
| false carry static clips: mean \|p − ctx-mean8\| (n 72; lower = less invented change) | 0.140 | copy 0.189 | 0 | oracle 0.181 · truth 0.121 | — | yes |
| false carry far cells (moving clips): mean \|p − ctx-mean8\| (n 400; lower = less invented change) | 0.122 | copy 0.181 | 0 | oracle 0.181 · truth 0.183 | — | yes |
| natural carry: steps where flow and target readout agree | 277 / 1108 moved steps | | | | | |
