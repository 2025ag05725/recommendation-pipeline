# Data Validation Report

Engine: **great_expectations 1.24.1**

Quality gate: reject-rate per table must be <= **15%** -> **PASSED**

Severity: `error` = record quarantined to `data/errors/`; `warning` = record kept and repaired in preparation.

## Summary by table

| Table | Rows in | Valid | Rejected | Reject rate | Rows with warnings |
|---|---:|---:|---:|---:|---:|
| users | 6040 | 6040 | 0 | 0.00% | 6040 |
| items | 3883 | 3883 | 0 | 0.00% | 251 |
| interactions | 1000209 | 1000209 | 0 | 0.00% | 0 |

## All checks (pass and fail)

| Table | Check | Severity | Description | Failed / checked | Status |
|---|---|---|---|---:|---|
| users | `missing_user_id` | error | user_id must not be null | 0 / 6040 | PASS |
| users | `invalid_user_id_type` | error | user_id must be an integer | 0 / 6040 | PASS |
| users | `duplicate_user_id` | error | user_id must be unique (first occurrence kept) | 0 / 6040 | PASS |
| users | `invalid_signup_date` | error | signup_date must be a parseable date | 0 / 6040 | PASS |
| users | `future_signup_date` | error | signup_date must not be after reference date | 0 / 6040 | PASS |
| users | `missing_age` | warning | age should not be null (imputed in preparation) | 0 / 6040 | PASS |
| users | `invalid_age` | warning | age must be numeric in [13, 100] | 0 / 6040 | PASS |
| users | `missing_gender` | warning | gender should not be null | 0 / 6040 | PASS |
| users | `invalid_gender` | warning | gender must be a known code | 0 / 6040 | PASS |
| users | `missing_country` | warning | country should not be null | 6040 / 6040 | WARN |
| items | `missing_item_id` | error | item_id must not be null | 0 / 3883 | PASS |
| items | `invalid_item_id_type` | error | item_id must be an integer | 0 / 3883 | PASS |
| items | `duplicate_item_id` | error | item_id must be unique (first occurrence kept) | 0 / 3883 | PASS |
| items | `missing_title` | warning | title should not be null | 0 / 3883 | PASS |
| items | `missing_genres` | warning | genres should not be null | 0 / 3883 | PASS |
| items | `unknown_genre` | warning | every genre token must be in the genre vocabulary | 251 / 3883 | WARN |
| items | `invalid_release_year` | warning | release_year must be an integer in [1870, 2026] | 0 / 3883 | PASS |
| interactions | `missing_user_id` | error | user_id must not be null | 0 / 1000209 | PASS |
| interactions | `missing_item_id` | error | item_id must not be null | 0 / 1000209 | PASS |
| interactions | `missing_rating` | error | rating must not be null | 0 / 1000209 | PASS |
| interactions | `missing_timestamp` | error | timestamp must not be null | 0 / 1000209 | PASS |
| interactions | `invalid_user_id_type` | error | user_id must be an integer | 0 / 1000209 | PASS |
| interactions | `invalid_item_id_type` | error | item_id must be an integer | 0 / 1000209 | PASS |
| interactions | `invalid_rating_type` | error | rating must be numeric | 0 / 1000209 | PASS |
| interactions | `rating_out_of_range` | error | rating must be within [0.5, 5.0] | 0 / 1000209 | PASS |
| interactions | `rating_off_scale` | error | rating must be a multiple of 0.5 | 0 / 1000209 | PASS |
| interactions | `invalid_timestamp_format` | error | timestamp must be parseable | 0 / 1000209 | PASS |
| interactions | `timestamp_in_future` | error | timestamp must not exceed reference date | 0 / 1000209 | PASS |
| interactions | `timestamp_too_old` | error | timestamp must be >= 1990-01-01 | 0 / 1000209 | PASS |
| interactions | `orphan_user_id` | error | user_id must exist in valid users | 0 / 1000209 | PASS |
| interactions | `orphan_item_id` | error | item_id must exist in valid items | 0 / 1000209 | PASS |
| interactions | `interaction_before_signup` | error | timestamp must be >= the user's signup_date | 0 / 1000209 | PASS |
| interactions | `exact_duplicate_row` | error | identical (user,item,rating,timestamp) rows are duplicates (first kept) | 0 / 1000209 | PASS |
| interactions | `duplicate_user_item_pair` | warning | same user rated same item more than once (latest kept in preparation) | 0 / 1000209 | PASS |

Rejected records with reasons: `data/errors/rejected_<table>.csv`; every individual issue: `data/errors/validation_issues.csv`.
