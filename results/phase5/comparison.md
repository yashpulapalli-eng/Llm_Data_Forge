Seeds: [1, 2] | seed-to-seed spread (noise): clean 0.0579, raw 0.0658

| Variant | Docs | Tokens | Clean loss (ppl) | vs raw | Raw-test loss (ppl) | vs raw | Train time |
|---|---|---|---|---|---|---|---|
| Raw | 19,000 | 12,100,833 | 5.8407 (344.0) | - | 5.8614 (351.2) | - | 300s |
| Filtered | 17,728 | 11,818,704 | 5.8196 (336.9) | -0.0211 (within noise) | 5.8364 (342.6) | -0.0250 (within noise) | 286s |
| Deduplicated | 18,984 | 12,097,228 | 5.8344 (341.9) | -0.0063 (within noise) | 5.8546 (348.8) | -0.0068 (within noise) | 291s |
| Filtered + dedup | 17,712 | 11,815,099 | 5.8395 (343.6) | -0.0012 (within noise) | 5.8584 (350.1) | -0.0030 (within noise) | 286s |
