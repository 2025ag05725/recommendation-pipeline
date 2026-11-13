import pandas as pd

from src.common.config import load_config
from src.validation.checks import IssueLog, validate_interactions, validate_items, validate_users

CFG = load_config()


def _df(rows, cols):
    d = pd.DataFrame(rows, columns=cols)
    d.insert(0, "_row_id", range(2, len(d) + 2))
    return d


def test_interaction_checks_catch_each_defect():
    users = _df([["1", "30", "M", "IN", "2024-01-01"], ["2", "30", "F", "US", "2024-01-01"]], ["user_id", "age", "gender", "country", "signup_date"])
    items = _df([["10", "a", "Drama", "2000"], ["11", "b", "Comedy", "2001"]], ["item_id", "title", "genres", "release_year"])
    rows = [
        ["1", "10", "4.0", "2025-01-01 10:00:00"],   # 0 ok
        ["1", "11", "6", "2025-01-01 10:00:00"],     # 1 rating out of range
        ["1", "11", "four", "2025-01-01 10:00:00"],  # 2 invalid type
        ["2", "10", "3.3", "2025-01-01 10:00:00"],   # 3 off-scale
        ["2", "11", "3", "2027-01-01 00:00:00"],     # 4 future
        ["2", "11", "3", "garbage"],                 # 5 bad timestamp
        ["99", "10", "3", "2025-01-01 10:00:00"],    # 6 orphan user
        ["1", "99", "3", "2025-01-01 10:00:00"],     # 7 orphan item
        ["1", "10", "4.0", "2025-01-01 10:00:00"],   # 8 exact duplicate of 0
        ["1", "10", "2.0", "2025-02-01 10:00:00"],   # 9 same pair, different values -> warning only
        ["2", "10", "3", "2023-01-01 00:00:00"],     # 10 before signup
        [None, "10", "3", "2025-01-01 10:00:00"],    # 11 missing user
    ]
    inter = _df(rows, ["user_id", "item_id", "rating", "timestamp"])
    log = IssueLog()
    valid = validate_interactions(inter, CFG, log, users, items)
    assert sorted(valid["_row_id"] - 2) == [0, 9]
    failed = {r["check"] for r in log.results if r["rows_failed"] and r["severity"] == "error"}
    assert {"rating_out_of_range", "invalid_rating_type", "rating_off_scale", "timestamp_in_future", "invalid_timestamp_format",
            "orphan_user_id", "orphan_item_id", "exact_duplicate_row", "interaction_before_signup", "missing_user_id"} <= failed
    assert any(r["check"] == "duplicate_user_item_pair" and r["rows_failed"] == 2 for r in log.results)


def test_users_and_items_warnings_do_not_reject():
    users = _df([["1", None, None, None, "2024-01-01"], ["2", "250", "x", "IN", "2024-01-01"], ["2", "30", "M", "IN", "2024-01-01"]],
                ["user_id", "age", "gender", "country", "signup_date"])
    log = IssueLog()
    assert len(validate_users(users, CFG, log)) == 2          # duplicate id rejected, bad age only warned
    items = _df([["1", None, None, "1850"]], ["item_id", "title", "genres", "release_year"])
    assert len(validate_items(items, CFG, IssueLog(), {"Drama"})) == 1


def test_prepare_survives_columns_that_are_entirely_missing():
    """Real MovieLens (ml-latest-small) has no age/gender/country and some movies lack years."""
    from src.preparation.prepare import prepare_items, prepare_users
    users = pd.DataFrame({"user_id": ["1", "2"], "age": [None, None], "gender": [None, None], "country": [None, None], "signup_date": ["2020-01-01"] * 2})
    stats = {}
    out = prepare_users(users, CFG, stats)
    assert out["age"].tolist() == [30, 30] and out["gender"].tolist() == ["U", "U"] and out["country"].tolist() == ["UNK", "UNK"]
    items = pd.DataFrame({"item_id": ["1"], "title": ["x"], "genres": ["(no genres listed)"], "release_year": [None]})
    out_i = prepare_items(items, CFG, stats)
    assert out_i["genres"].tolist() == ["Unknown"] and out_i["release_year"].tolist() == [2000]
