# E2 — boundary tubelet identity at the predictor input under occlusion (IntPhysGen v11_full, ViT-H)

Every number below is recomputed from `results.json` written by `z_research/scripts/analysis/ctxenc_boundary_identity_occlusion.py` (seed 0, 100 blocks = up to 200 clips per (condition, k) cell, 2079 clips total).
Readout = closed-form one-vs-rest ridge (argmax), block-held-out, lambda chosen on an inner block-held-out validation fold. `shuf` = same fit with train labels permuted.
Reps: `z` = ctx_masked (predictor input), `h` = target encoder (ceiling). Feats: B7 = 3x3 at true object position @t7, B7all = all 256 tokens @t7, C = all 2048 context tokens, BG = fixed far 3x3 @t7 (control).

## Cells

| cell | condition | k | n clips | n blocks | hidden frac of t7 frames (42,45) | 3x3 window size | obj x @t7 (px) |
|---|---|---|---:|---:|---:|---:|---|
| static_visible | static_visible | all | 169 | 100 | 0.00 | 9.0 | 66–222 |
| static_late | static_occlusion | all | 679 | 400 | 0.88 | 9.0 | 66–222 |
| static_late | static_occlusion | 1 | 165 | 100 | 0.50 | 9.0 | 66–222 |
| static_late | static_occlusion | 2 | 174 | 100 | 1.00 | 9.0 | 66–222 |
| static_late | static_occlusion | 3 | 169 | 100 | 1.00 | 9.0 | 66–222 |
| static_late | static_occlusion | 4 | 171 | 100 | 1.00 | 9.0 | 66–222 |
| static_early_k4 | static_occlusion_early | all | 176 | 100 | 0.00 | 9.0 | 66–222 |
| moving_flat_visible | moving_visible_flat | all | 173 | 100 | 0.00 | 9.0 | 136–152 |
| moving_flat_late | moving_occlusion_flat | all | 701 | 400 | 0.87 | 9.0 | 136–152 |
| moving_flat_late | moving_occlusion_flat | 1 | 184 | 100 | 0.50 | 9.0 | 136–152 |
| moving_flat_late | moving_occlusion_flat | 2 | 175 | 100 | 1.00 | 9.0 | 136–152 |
| moving_flat_late | moving_occlusion_flat | 3 | 172 | 100 | 1.00 | 9.0 | 136–152 |
| moving_flat_late | moving_occlusion_flat | 4 | 170 | 100 | 1.00 | 9.0 | 136–152 |
| moving_flat_early_k4 | moving_occlusion_flat_early | all | 181 | 100 | 0.00 | 9.0 | 136–152 |

## shape_pre — accuracy % (shuffle control in parentheses), chance 14.3%

### static

| fit | train → test | k | n_train | n_test | z:B7 | z:B7all | z:C | z:BG | h:B7 | h:B7all | h:C | h:BG |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| self | static_visible → static_visible | all | 169 | 169 | 87.6 (10.7) | 85.2 (14.2) | 91.1 (18.3) | 43.8 (17.2) | 94.7 (20.1) | 88.8 (17.2) | 91.1 (16.0) | 49.7 (14.2) |
| self | static_early_k4 → static_early_k4 | all | 176 | 176 | 92.0 (10.2) | 74.4 (13.1) | 90.3 (22.7) | 34.7 (15.3) | 92.0 (9.7) | 78.4 (15.9) | 86.4 (15.9) | 33.5 (17.6) |
| self | static_late → static_late | all | 679 | 679 | 88.7 (13.3) | 87.0 (15.3) | 98.4 (13.4) | 56.1 (11.9) | 85.3 (15.3) | 91.0 (12.2) | 98.7 (12.1) | 67.3 (12.1) |
| self | static_late → static_late | 1 | 165 | 165 | 80.0 (10.3) | 74.5 (12.1) | 80.0 (22.4) | 33.3 (13.3) | 77.6 (15.8) | 77.6 (10.3) | 86.1 (13.9) | 36.4 (10.9) |
| self | static_late → static_late | 2 | 174 | 174 | 76.4 (14.4) | 74.7 (13.8) | 89.7 (13.8) | 40.2 (12.1) | 77.0 (13.2) | 72.4 (15.5) | 88.5 (20.7) | 48.3 (13.8) |
| self | static_late → static_late | 3 | 169 | 169 | 71.6 (17.8) | 65.1 (15.4) | 79.9 (4.7) | 29.6 (10.1) | 52.7 (10.7) | 69.8 (16.0) | 85.8 (9.5) | 39.1 (17.8) |
| self | static_late → static_late | 4 | 171 | 171 | 74.3 (9.4) | 70.2 (17.5) | 80.7 (15.2) | 34.5 (15.8) | 58.5 (16.4) | 74.9 (20.5) | 88.9 (12.9) | 44.4 (22.2) |
| transfer_visible->late | static_visible → static_late | all | 169 | 679 | 20.2 (15.6) | 25.5 (10.8) | 20.9 (20.8) | 23.6 (13.1) | 15.5 (15.2) | 29.0 (14.7) | 50.7 (16.3) | 30.2 (12.5) |
| transfer_visible->late | static_visible → static_late | 1 | 169 | 165 | 33.9 (17.6) | 22.4 (12.7) | 22.4 (22.4) | 25.5 (14.5) | 12.7 (16.4) | 14.5 (13.3) | 66.1 (13.9) | 31.5 (9.7) |
| transfer_visible->late | static_visible → static_late | 2 | 169 | 174 | 14.9 (17.8) | 31.6 (9.2) | 21.3 (25.3) | 24.7 (10.3) | 14.9 (17.8) | 21.8 (16.1) | 55.7 (17.8) | 31.0 (10.9) |
| transfer_visible->late | static_visible → static_late | 3 | 169 | 169 | 14.2 (11.2) | 26.6 (10.7) | 20.1 (16.0) | 24.9 (16.6) | 20.1 (11.8) | 32.0 (16.6) | 36.7 (17.2) | 29.6 (11.2) |
| transfer_visible->late | static_visible → static_late | 4 | 169 | 171 | 18.1 (15.8) | 21.1 (10.5) | 19.9 (19.3) | 19.3 (11.1) | 14.0 (14.6) | 47.4 (12.9) | 44.4 (16.4) | 28.7 (18.1) |
| transfer_visible->early | static_visible → static_early_k4 | all | 169 | 176 | 76.1 (21.6) | 54.5 (4.5) | 41.5 (19.3) | 30.1 (14.8) | 57.4 (17.0) | 60.2 (13.6) | 54.5 (14.8) | 34.7 (13.6) |
| transfer_early->late | static_early_k4 → static_late | all | 176 | 679 | 19.0 (12.1) | 15.9 (14.1) | 41.2 (11.0) | 30.3 (14.4) | 15.2 (14.0) | 38.6 (15.5) | 79.8 (14.4) | 37.0 (16.1) |
| transfer_early->late | static_early_k4 → static_late | 1 | 176 | 165 | 26.1 (10.9) | 17.0 (15.2) | 55.8 (10.9) | 37.6 (13.3) | 15.2 (10.9) | 40.6 (14.5) | 78.8 (17.0) | 36.4 (20.0) |
| transfer_early->late | static_early_k4 → static_late | 2 | 176 | 174 | 20.1 (14.4) | 15.5 (13.8) | 54.0 (9.2) | 27.0 (16.7) | 9.8 (17.2) | 33.3 (14.4) | 83.3 (11.5) | 32.2 (10.9) |
| transfer_early->late | static_early_k4 → static_late | 3 | 176 | 169 | 16.6 (9.5) | 16.0 (11.8) | 31.4 (11.2) | 24.9 (11.8) | 19.5 (14.8) | 40.8 (14.2) | 76.9 (16.0) | 40.2 (16.6) |
| transfer_early->late | static_early_k4 → static_late | 4 | 176 | 171 | 13.5 (13.5) | 15.2 (15.8) | 24.0 (12.9) | 32.2 (15.8) | 16.4 (12.9) | 39.8 (18.7) | 80.1 (13.5) | 39.2 (17.0) |

### moving_flat

| fit | train → test | k | n_train | n_test | z:B7 | z:B7all | z:C | z:BG | h:B7 | h:B7all | h:C | h:BG |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| self | moving_flat_visible → moving_flat_visible | all | 173 | 173 | 98.8 (14.5) | 99.4 (15.0) | 99.4 (16.2) | 74.6 (18.5) | 100.0 (17.3) | 98.3 (13.9) | 99.4 (10.4) | 73.4 (11.6) |
| self | moving_flat_early_k4 → moving_flat_early_k4 | all | 181 | 181 | 98.9 (17.7) | 99.4 (15.5) | 97.2 (16.6) | 67.4 (8.8) | 100.0 (17.1) | 99.4 (16.0) | 98.9 (16.6) | 48.1 (11.6) |
| self | moving_flat_late → moving_flat_late | all | 701 | 701 | 82.2 (14.3) | 99.1 (15.0) | 99.7 (16.7) | 76.5 (14.4) | 77.9 (16.0) | 99.3 (10.6) | 100.0 (13.8) | 82.6 (12.4) |
| self | moving_flat_late → moving_flat_late | 1 | 184 | 184 | 90.8 (20.7) | 96.2 (18.5) | 97.8 (11.4) | 58.7 (11.4) | 97.3 (15.2) | 96.7 (13.6) | 99.5 (8.2) | 57.1 (16.3) |
| self | moving_flat_late → moving_flat_late | 2 | 175 | 175 | 72.0 (16.0) | 90.3 (11.4) | 94.3 (19.4) | 46.9 (13.7) | 66.3 (11.4) | 94.3 (14.9) | 97.7 (16.0) | 43.4 (12.0) |
| self | moving_flat_late → moving_flat_late | 3 | 172 | 172 | 49.4 (14.0) | 89.5 (16.3) | 97.1 (16.3) | 53.5 (9.9) | 42.4 (15.7) | 90.1 (10.5) | 98.3 (14.5) | 50.6 (15.1) |
| self | moving_flat_late → moving_flat_late | 4 | 170 | 170 | 54.7 (11.8) | 89.4 (13.5) | 95.9 (20.0) | 48.8 (12.4) | 44.1 (20.0) | 87.1 (14.1) | 97.6 (18.8) | 50.0 (15.3) |
| transfer_visible->late | moving_flat_visible → moving_flat_late | all | 173 | 701 | 16.0 (16.7) | 19.5 (12.7) | 23.7 (12.7) | 33.4 (9.8) | 14.4 (15.3) | 24.1 (16.0) | 44.7 (7.1) | 46.4 (12.8) |
| transfer_visible->late | moving_flat_visible → moving_flat_late | 1 | 173 | 184 | 16.3 (20.7) | 27.7 (9.8) | 31.0 (14.7) | 40.2 (7.1) | 14.1 (11.4) | 29.9 (19.0) | 40.8 (7.1) | 50.5 (10.9) |
| transfer_visible->late | moving_flat_visible → moving_flat_late | 2 | 173 | 175 | 16.6 (20.0) | 18.3 (13.7) | 24.0 (10.9) | 36.0 (10.9) | 16.0 (15.4) | 26.9 (11.4) | 46.3 (8.6) | 48.6 (11.4) |
| transfer_visible->late | moving_flat_visible → moving_flat_late | 3 | 173 | 172 | 18.0 (10.5) | 15.1 (13.4) | 20.9 (11.6) | 29.1 (11.6) | 11.6 (19.8) | 18.6 (15.7) | 45.9 (6.4) | 47.1 (15.7) |
| transfer_visible->late | moving_flat_visible → moving_flat_late | 4 | 173 | 170 | 12.9 (15.3) | 16.5 (14.1) | 18.2 (13.5) | 27.6 (10.0) | 15.9 (14.7) | 20.6 (17.6) | 45.9 (6.5) | 38.8 (13.5) |
| transfer_visible->early | moving_flat_visible → moving_flat_early_k4 | all | 173 | 181 | 94.5 (27.6) | 64.1 (14.4) | 47.5 (11.6) | 52.5 (12.2) | 98.3 (11.6) | 78.5 (26.0) | 56.9 (4.4) | 56.4 (15.5) |
| transfer_early->late | moving_flat_early_k4 → moving_flat_late | all | 181 | 701 | 16.0 (12.4) | 35.9 (14.1) | 56.5 (8.8) | 45.6 (12.7) | 14.6 (17.5) | 25.8 (13.6) | 65.5 (10.3) | 44.9 (15.0) |
| transfer_early->late | moving_flat_early_k4 → moving_flat_late | 1 | 181 | 184 | 17.4 (13.0) | 57.1 (13.0) | 72.3 (12.5) | 53.3 (13.0) | 14.1 (22.3) | 37.5 (12.0) | 81.5 (8.2) | 47.8 (16.3) |
| transfer_early->late | moving_flat_early_k4 → moving_flat_late | 2 | 181 | 175 | 19.4 (13.1) | 31.4 (13.7) | 50.9 (5.7) | 42.3 (12.0) | 16.0 (14.9) | 23.4 (17.1) | 62.3 (10.9) | 46.3 (15.4) |
| transfer_early->late | moving_flat_early_k4 → moving_flat_late | 3 | 181 | 172 | 14.5 (13.4) | 26.2 (14.0) | 52.9 (11.0) | 42.4 (14.0) | 11.6 (15.7) | 20.9 (15.1) | 62.8 (14.0) | 43.0 (14.0) |
| transfer_early->late | moving_flat_early_k4 → moving_flat_late | 4 | 181 | 170 | 12.4 (10.0) | 27.6 (15.9) | 48.8 (5.9) | 44.1 (11.8) | 16.5 (17.1) | 20.6 (10.0) | 54.1 (8.2) | 42.4 (14.1) |

## color_pre — accuracy % (shuffle control in parentheses), chance 12.5%

### static

| fit | train → test | k | n_train | n_test | z:B7 | z:B7all | z:C | z:BG | h:B7 | h:B7all | h:C | h:BG |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| self | static_visible → static_visible | all | 169 | 169 | 58.0 (16.6) | 72.2 (14.2) | 84.0 (13.6) | 19.5 (12.4) | 60.4 (13.6) | 71.6 (7.7) | 83.4 (16.6) | 19.5 (11.2) |
| self | static_early_k4 → static_early_k4 | all | 176 | 176 | 82.4 (14.2) | 47.2 (9.1) | 75.0 (9.7) | 13.1 (12.5) | 69.9 (11.4) | 56.2 (5.1) | 75.0 (11.4) | 10.2 (12.5) |
| self | static_late → static_late | all | 679 | 679 | 64.2 (13.7) | 76.9 (12.4) | 89.0 (15.2) | 28.4 (11.8) | 65.2 (13.1) | 75.1 (11.3) | 89.2 (12.7) | 28.6 (14.3) |
| self | static_late → static_late | 1 | 165 | 165 | 52.7 (16.4) | 57.0 (13.9) | 69.7 (14.5) | 14.5 (17.6) | 54.5 (16.4) | 54.5 (12.1) | 60.0 (17.0) | 17.0 (21.8) |
| self | static_late → static_late | 2 | 174 | 174 | 47.1 (16.1) | 52.3 (13.8) | 70.7 (13.8) | 14.9 (12.1) | 48.3 (13.8) | 52.3 (13.8) | 68.4 (9.8) | 19.0 (12.6) |
| self | static_late → static_late | 3 | 169 | 169 | 49.7 (14.8) | 46.2 (16.6) | 62.1 (7.7) | 14.8 (11.2) | 33.1 (14.8) | 42.0 (11.8) | 65.7 (13.0) | 10.1 (18.9) |
| self | static_late → static_late | 4 | 171 | 171 | 45.0 (15.8) | 36.3 (8.8) | 63.7 (8.8) | 11.1 (12.3) | 29.8 (13.5) | 53.2 (10.5) | 67.8 (12.9) | 6.4 (14.0) |
| transfer_visible->late | static_visible → static_late | all | 169 | 679 | 12.5 (11.2) | 17.8 (10.9) | 32.4 (14.7) | 16.9 (11.2) | 13.1 (10.8) | 13.7 (14.9) | 32.1 (11.2) | 17.4 (11.9) |
| transfer_visible->late | static_visible → static_late | 1 | 169 | 165 | 15.2 (11.5) | 18.8 (13.9) | 33.3 (12.1) | 15.2 (9.1) | 12.7 (9.1) | 9.7 (15.8) | 33.3 (7.9) | 16.4 (6.7) |
| transfer_visible->late | static_visible → static_late | 2 | 169 | 174 | 11.5 (9.8) | 12.1 (8.0) | 32.8 (19.0) | 20.1 (12.6) | 11.5 (12.1) | 11.5 (12.1) | 34.5 (11.5) | 19.5 (10.3) |
| transfer_visible->late | static_visible → static_late | 3 | 169 | 169 | 11.8 (9.5) | 18.3 (7.1) | 34.3 (10.1) | 16.6 (8.9) | 14.2 (13.0) | 17.2 (17.2) | 29.6 (15.4) | 15.4 (16.6) |
| transfer_visible->late | static_visible → static_late | 4 | 169 | 171 | 11.7 (14.0) | 22.2 (14.6) | 29.2 (17.5) | 15.8 (14.0) | 14.0 (8.8) | 16.4 (14.6) | 31.0 (9.9) | 18.1 (14.0) |
| transfer_visible->early | static_visible → static_early_k4 | all | 169 | 176 | 52.8 (14.8) | 27.8 (11.9) | 33.5 (11.9) | 14.2 (11.4) | 42.6 (6.2) | 22.7 (13.6) | 20.5 (10.8) | 13.6 (10.8) |
| transfer_early->late | static_early_k4 → static_late | all | 176 | 679 | 14.3 (13.0) | 17.1 (14.3) | 53.6 (12.4) | 16.3 (11.6) | 10.6 (10.2) | 15.9 (15.9) | 43.3 (11.6) | 13.7 (14.7) |
| transfer_early->late | static_early_k4 → static_late | 1 | 176 | 165 | 19.4 (10.9) | 21.8 (11.5) | 52.7 (18.8) | 15.8 (13.3) | 11.5 (11.5) | 13.3 (15.2) | 47.3 (11.5) | 10.3 (16.4) |
| transfer_early->late | static_early_k4 → static_late | 2 | 176 | 174 | 11.5 (13.2) | 14.9 (11.5) | 57.5 (13.8) | 18.4 (8.6) | 2.3 (9.8) | 13.8 (16.7) | 48.9 (13.8) | 17.8 (16.7) |
| transfer_early->late | static_early_k4 → static_late | 3 | 176 | 169 | 14.2 (13.0) | 14.8 (17.8) | 52.1 (9.5) | 14.2 (13.6) | 11.2 (8.9) | 18.3 (18.3) | 46.7 (11.8) | 14.8 (13.6) |
| transfer_early->late | static_early_k4 → static_late | 4 | 176 | 171 | 12.3 (14.6) | 17.0 (16.4) | 52.0 (7.6) | 17.0 (11.1) | 17.5 (10.5) | 18.1 (13.5) | 30.4 (9.4) | 11.7 (12.3) |

### moving_flat

| fit | train → test | k | n_train | n_test | z:B7 | z:B7all | z:C | z:BG | h:B7 | h:B7all | h:C | h:BG |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| self | moving_flat_visible → moving_flat_visible | all | 173 | 173 | 97.1 (12.1) | 95.4 (9.8) | 96.5 (12.7) | 38.2 (11.0) | 94.8 (15.0) | 92.5 (19.1) | 96.5 (19.1) | 45.1 (14.5) |
| self | moving_flat_early_k4 → moving_flat_early_k4 | all | 181 | 181 | 91.7 (7.7) | 88.4 (12.2) | 92.8 (9.9) | 23.2 (16.0) | 93.9 (14.9) | 84.5 (12.2) | 83.4 (11.0) | 24.9 (16.6) |
| self | moving_flat_late → moving_flat_late | all | 701 | 701 | 67.2 (12.1) | 95.3 (11.4) | 98.3 (12.6) | 55.5 (12.6) | 69.5 (11.8) | 96.4 (10.4) | 99.4 (11.0) | 58.5 (12.4) |
| self | moving_flat_late → moving_flat_late | 1 | 184 | 184 | 69.0 (12.0) | 89.1 (14.7) | 91.8 (15.2) | 34.8 (11.4) | 75.5 (13.0) | 93.5 (10.9) | 96.7 (9.2) | 17.4 (10.9) |
| self | moving_flat_late → moving_flat_late | 2 | 175 | 175 | 45.7 (15.4) | 85.7 (12.6) | 93.7 (13.7) | 17.1 (9.7) | 45.7 (12.0) | 81.1 (13.7) | 95.4 (13.1) | 25.7 (16.0) |
| self | moving_flat_late → moving_flat_late | 3 | 172 | 172 | 29.1 (9.3) | 80.2 (9.3) | 84.9 (14.0) | 22.1 (18.6) | 40.1 (18.0) | 87.2 (8.7) | 94.8 (13.4) | 27.9 (9.3) |
| self | moving_flat_late → moving_flat_late | 4 | 170 | 170 | 21.2 (16.5) | 68.2 (12.9) | 82.9 (14.7) | 27.6 (13.5) | 30.0 (12.9) | 72.9 (11.2) | 95.9 (18.2) | 23.5 (12.4) |
| transfer_visible->late | moving_flat_visible → moving_flat_late | all | 173 | 701 | 14.0 (13.0) | 31.5 (13.8) | 39.1 (12.6) | 23.0 (11.6) | 12.1 (11.7) | 33.5 (11.3) | 38.9 (12.3) | 27.4 (13.4) |
| transfer_visible->late | moving_flat_visible → moving_flat_late | 1 | 173 | 184 | 16.8 (12.0) | 35.9 (13.6) | 36.4 (11.4) | 27.2 (7.6) | 10.9 (13.6) | 48.9 (9.2) | 33.7 (9.8) | 24.5 (18.5) |
| transfer_visible->late | moving_flat_visible → moving_flat_late | 2 | 173 | 175 | 13.7 (10.9) | 33.7 (14.9) | 35.4 (17.1) | 21.1 (10.9) | 14.3 (7.4) | 37.1 (9.7) | 38.3 (12.6) | 29.1 (13.7) |
| transfer_visible->late | moving_flat_visible → moving_flat_late | 3 | 173 | 172 | 14.5 (15.7) | 25.6 (12.8) | 42.4 (12.2) | 22.1 (16.3) | 11.6 (11.6) | 23.8 (12.2) | 41.9 (11.0) | 27.9 (9.9) |
| transfer_visible->late | moving_flat_visible → moving_flat_late | 4 | 173 | 170 | 10.6 (13.5) | 30.6 (14.1) | 42.4 (9.4) | 21.2 (11.8) | 11.8 (14.1) | 22.9 (14.1) | 42.4 (15.9) | 28.2 (11.2) |
| transfer_visible->early | moving_flat_visible → moving_flat_early_k4 | all | 173 | 181 | 85.1 (14.4) | 77.3 (7.7) | 45.3 (14.4) | 30.9 (11.0) | 89.0 (13.8) | 70.7 (7.7) | 39.2 (14.4) | 29.8 (14.9) |
| transfer_early->late | moving_flat_early_k4 → moving_flat_late | all | 181 | 701 | 13.0 (12.4) | 29.5 (15.0) | 53.2 (12.3) | 24.0 (10.0) | 12.1 (12.3) | 41.1 (11.6) | 60.1 (13.6) | 27.5 (11.4) |
| transfer_early->late | moving_flat_early_k4 → moving_flat_late | 1 | 181 | 184 | 13.6 (15.2) | 35.9 (14.1) | 57.6 (10.9) | 24.5 (8.2) | 11.4 (10.9) | 47.3 (11.4) | 68.5 (12.0) | 26.1 (10.3) |
| transfer_early->late | moving_flat_early_k4 → moving_flat_late | 2 | 181 | 175 | 12.6 (12.0) | 32.0 (16.6) | 53.1 (14.9) | 26.9 (7.4) | 12.6 (12.6) | 49.1 (12.6) | 65.7 (14.3) | 28.6 (13.1) |
| transfer_early->late | moving_flat_early_k4 → moving_flat_late | 3 | 181 | 172 | 14.5 (9.3) | 27.9 (14.5) | 55.2 (14.0) | 23.8 (12.2) | 14.0 (11.6) | 39.5 (8.7) | 54.1 (12.2) | 29.7 (8.7) |
| transfer_early->late | moving_flat_early_k4 → moving_flat_late | 4 | 181 | 170 | 11.2 (12.9) | 21.8 (14.7) | 46.5 (9.4) | 20.6 (12.4) | 10.6 (14.1) | 27.6 (13.5) | 51.2 (15.9) | 25.9 (13.5) |

## 재현

```
cd /data/hyuntak/project/2026/2027_cvpr/vjepa2
/data/hyuntak/anaconda3/envs/vjepa2/bin/python z_research/scripts/analysis/ctxenc_boundary_identity_occlusion.py --blocks 100
```
