# Phase 1 — inspection report

Rows **22,372** · features **22** · churn share **61.00%** · missing cells **0**

## Splits

| Split | Snapshot | Rows | Churn | Customers |
|---|---|---|---|---|
| test | 2011-09-09 | 5,253 | 56.6% | 5,253 |
| train | 2010-09-09 | 3,337 | 42.2% | 3,337 |
| train | 2010-12-09 | 4,282 | 68.9% | 4,282 |
| train | 2011-03-09 | 4,559 | 65.2% | 4,559 |
| val | 2011-06-09 | 4,941 | 67.7% | 4,941 |

## Customer overlap

- Unique customers: 5,253
- Customers with more than one snapshot row: 4,941
- Test customers already seen in train/val: 4,941 (94.1%)

## Features

| Feature | Min | Max | Skew | Zero share | Unique | corr with churn |
|---|---|---|---|---|---|---|
| recency | 0.168 | 646.549 | 1.17 | 0.0% | 20540 | 0.343 |
| orders_last_90d | 0.000 | 77.000 | 11.08 | 54.4% | 39 | -0.293 |
| n_unique_products | 1.000 | 2178.000 | 6.17 | 0.0% | 550 | -0.292 |
| is_single_order | 0.000 | 1.000 | 0.74 | 67.3% | 2 | 0.288 |
| orders_last_60d | 0.000 | 54.000 | 9.57 | 63.2% | 31 | -0.285 |
| frequency | 1.000 | 284.000 | 10.53 | 0.0% | 123 | -0.277 |
| orders_last_30d | 0.000 | 34.000 | 9.51 | 76.6% | 19 | -0.242 |
| spend_last_90d | 0.000 | 103579.500 | 23.13 | 54.4% | 9442 | -0.157 |
| monetary | 1.550 | 456780.490 | 24.62 | 0.0% | 11444 | -0.148 |
| avg_days_between_orders | 0.001 | 632.120 | 2.78 | 0.0% | 8992 | 0.132 |
| std_days_between_orders | 0.000 | 440.493 | 2.43 | 50.6% | 7341 | -0.104 |
| recency_to_gap_ratio | 0.001 | 646.449 | 10.66 | 0.0% | 22343 | 0.073 |
| tenure | 0.300 | 646.677 | 0.12 | 0.0% | 21315 | -0.073 |
| std_order_value | 0.000 | 44513.120 | 49.47 | 32.9% | 9043 | -0.071 |
| avg_order_value | 1.550 | 25784.320 | 19.66 | 0.0% | 11423 | -0.049 |
| frequency_trend | -1.000 | 1.000 | 0.01 | 42.1% | 160 | -0.036 |
| spend_trend | -1.000 | 1.000 | 0.03 | 33.0% | 5539 | -0.032 |
| avg_unit_price | 0.151 | 434.650 | 39.40 | 0.0% | 11179 | 0.026 |
| is_uk | 0.000 | 1.000 | -3.07 | 8.1% | 2 | 0.024 |
| return_rate | 0.000 | 1.000 | 7.86 | 60.4% | 5716 | 0.021 |
| weekend_share | 0.000 | 1.000 | 2.01 | 66.9% | 2942 | -0.008 |
| avg_basket_size | 1.000 | 87167.000 | 46.00 | 0.0% | 5626 | -0.001 |

## Saved model

- RandomForestClassifier · trees 100 · depth 8 · leaf 20
- threshold **0.45** · features **22** · order matches dataset: **True**

## Checks

- PASS — target not in features
- PASS — identifiers not in features
- PASS — snapshot belongs to one split
- PASS — no duplicate (customer, snapshot)
- PASS — label is binary
- PASS — train < val < test in time
