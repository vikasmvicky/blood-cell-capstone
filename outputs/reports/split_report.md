# Blood Cell Dataset Split Report

## Dataset summary

- Usable images: 17090
- Excluded images: 2
- Split ratios: 70% train, 15% validation, 15% test
- Random seed: 123

## Split distribution

| Split | Images | Percentage |
|---|---:|---:|
| train | 11964 | 70.01% |
| val | 2564 | 15.00% |
| test | 2562 | 14.99% |

## Class distribution

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| basophil | 852 | 183 | 183 |
| eosinophil | 2181 | 468 | 467 |
| erythroblast | 1086 | 233 | 232 |
| ig | 2027 | 434 | 434 |
| lymphocyte | 850 | 182 | 182 |
| monocyte | 994 | 213 | 213 |
| neutrophil | 2330 | 499 | 499 |
| platelet | 1644 | 352 | 352 |

## Exclusions

- cross_class_duplicate_conflict: 2 image(s)

## Leakage verification

The exact-hash and duplicate-group leakage checks passed.
