# Data Validation Report

Engine: **great_expectations 1.24.1**

Quality gate: reject-rate per table must be <= **15%** -> **PASSED**

Severity: `error` = record quarantined to `data/errors/`; `warning` = record kept and repaired in preparation.

## Summary by table

| Table | Rows in | Valid | Rejected | Reject rate | Rows with warnings |
|---|---:|---:|---:|---:|---:|
| users | 2010 | 1992 | 18 | 0.90% | 71 |
| items | 904 | 900 | 4 | 0.44% | 19 |
| interactions | 64571 | 59459 | 5112 | 7.92% | 1914 |

## All checks (pass and fail)

| Table | Check | Severity | Description | Failed / checked | Status |
|---|---|---|---|---:|---|
| users | `missing_user_id` | error | user_id must not be null | 0 / 2010 | PASS |
| users | `invalid_user_id_type` | error | user_id must be an integer | 0 / 2010 | PASS |
| users | `duplicate_user_id` | error | user_id must be unique (first occurrence kept) | 10 / 2010 | FAIL |
| users | `invalid_signup_date` | error | signup_date must be a parseable date | 8 / 2010 | FAIL |
| users | `future_signup_date` | error | signup_date must not be after reference date | 0 / 2010 | PASS |
| users | `missing_age` | warning | age should not be null (imputed in preparation) | 31 / 2010 | WARN |
| users | `invalid_age` | warning | age must be numeric in [13, 100] | 12 / 2010 | WARN |
| users | `missing_gender` | warning | gender should not be null | 8 / 2010 | WARN |
| users | `invalid_gender` | warning | gender must be a known code | 0 / 2010 | PASS |
| users | `missing_country` | warning | country should not be null | 20 / 2010 | WARN |
| items | `missing_item_id` | error | item_id must not be null | 0 / 904 | PASS |
| items | `invalid_item_id_type` | error | item_id must be an integer | 0 / 904 | PASS |
| items | `duplicate_item_id` | error | item_id must be unique (first occurrence kept) | 4 / 904 | FAIL |
| items | `missing_title` | warning | title should not be null | 4 / 904 | WARN |
| items | `missing_genres` | warning | genres should not be null | 9 / 904 | WARN |
| items | `unknown_genre` | warning | every genre token must be in the genre vocabulary | 0 / 904 | PASS |
| items | `invalid_release_year` | warning | release_year must be an integer in [1900, 2026] | 7 / 904 | WARN |
| interactions | `missing_user_id` | error | user_id must not be null | 130 / 64571 | FAIL |
| interactions | `missing_item_id` | error | item_id must not be null | 0 / 64571 | PASS |
| interactions | `missing_rating` | error | rating must not be null | 632 / 64571 | FAIL |
| interactions | `missing_timestamp` | error | timestamp must not be null | 92 / 64571 | FAIL |
| interactions | `invalid_user_id_type` | error | user_id must be an integer | 0 / 64571 | PASS |
| interactions | `invalid_item_id_type` | error | item_id must be an integer | 0 / 64571 | PASS |
| interactions | `invalid_rating_type` | error | rating must be numeric | 131 / 64571 | FAIL |
| interactions | `rating_out_of_range` | error | rating must be within [0.5, 5.0] | 508 / 64571 | FAIL |
| interactions | `rating_off_scale` | error | rating must be a multiple of 0.5 | 124 / 64571 | FAIL |
| interactions | `invalid_timestamp_format` | error | timestamp must be parseable | 448 / 64571 | FAIL |
| interactions | `timestamp_in_future` | error | timestamp must not exceed reference date | 326 / 64571 | FAIL |
| interactions | `timestamp_too_old` | error | timestamp must be >= 2015-01-01 | 194 / 64571 | FAIL |
| interactions | `orphan_user_id` | error | user_id must exist in valid users | 818 / 64571 | FAIL |
| interactions | `orphan_item_id` | error | item_id must exist in valid items | 651 / 64571 | FAIL |
| interactions | `interaction_before_signup` | error | timestamp must be >= the user's signup_date | 377 / 64571 | FAIL |
| interactions | `exact_duplicate_row` | error | identical (user,item,rating,timestamp) rows are duplicates (first kept) | 1003 / 64571 | FAIL |
| interactions | `duplicate_user_item_pair` | warning | same user rated same item more than once (latest kept in preparation) | 1914 / 64571 | WARN |

Rejected records with reasons: `data/errors/rejected_<table>.csv`; every individual issue: `data/errors/validation_issues.csv`.
