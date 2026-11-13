import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("pm", Path(__file__).resolve().parents[1] / "scripts" / "prepare_movielens.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


def test_latest_small_layout(tmp_path):
    d = tmp_path / "ml-latest-small"
    d.mkdir()
    pd.DataFrame({"userId": [1, 1, 2], "movieId": [10, 20, 10], "rating": [4.5, 3.0, 5.0], "timestamp": [964982703, 964981247, 964982224]}).to_csv(d / "ratings.csv", index=False)
    pd.DataFrame({"movieId": [10, 20], "title": ["Toy Story (1995)", "No Year Film"], "genres": ["Adventure|Animation|Children", "(no genres listed)"]}).to_csv(d / "movies.csv", index=False)
    users, items, inter = pm.load(tmp_path)
    assert list(inter.columns) == ["user_id", "item_id", "rating", "timestamp"] and inter["timestamp"].iloc[0] == "2000-07-30 18:45:03"
    assert items["release_year"].tolist() == ["1995", None] or items["release_year"].isna().tolist() == [False, True]
    assert items["genres"].isna().tolist() == [False, True]
    assert users.set_index("user_id").loc[1, "signup_date"] == "2000-07-30"          # first rating date
    assert users["age"].isna().all()


def test_ml1m_layout(tmp_path):
    d = tmp_path / "ml-1m"
    d.mkdir()
    (d / "ratings.dat").write_text("1::10::5::978300760\n1::20::3::978302109\n2::10::4::978298709\n")
    (d / "movies.dat").write_bytes("10::Toy Story (1995)::Animation|Children's|Comedy\n20::Caf\xe9 Film (1999)::Drama\n".encode("latin-1"))
    (d / "users.dat").write_text("1::F::1::10::48067\n2::M::56::16::70072\n")
    users, items, inter = pm.load(tmp_path)
    u = users.set_index("user_id")
    assert u.loc[1, "gender"] == "F" and u.loc[1, "age"] == 16 and u.loc[2, "age"] == 60
    assert items["title"].iloc[1] == "Café Film (1999)" and len(inter) == 3


def test_converted_output_passes_validation(tmp_path):
    from src.common.config import Config, load_config
    from src.validation import checks as C
    d = tmp_path / "ml"
    d.mkdir()
    pd.DataFrame({"userId": [1, 1, 2], "movieId": [10, 20, 10], "rating": [4.5, 3.0, 5.0], "timestamp": [964982703, 964981247, 964982224]}).to_csv(d / "ratings.csv", index=False)
    pd.DataFrame({"movieId": [10, 20], "title": ["Toy Story (1995)", "X (1999)"], "genres": ["Adventure|Animation|Children|Film-Noir|IMAX", "Drama"]}).to_csv(d / "movies.csv", index=False)
    users, items, inter = pm.load(tmp_path)
    for n, df in (("users", users), ("items", items), ("interactions", inter)):
        df.to_csv(tmp_path / f"{n}.csv", index=False)
    from src.common.io_utils import read_raw_csv
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "config.movielens.yaml")
    log = C.IssueLog()
    vu = C.validate_users(read_raw_csv(tmp_path / "users.csv"), cfg, log)
    vi = C.validate_items(read_raw_csv(tmp_path / "items.csv"), cfg, log, set(cfg["genre_vocabulary"]))
    vx = C.validate_interactions(read_raw_csv(tmp_path / "interactions.csv"), cfg, log, vu, vi)
    assert (len(vu), len(vi), len(vx)) == (2, 2, 3)                          # 1990s timestamps + MovieLens genres are accepted
    assert not [r for r in log.results if r["rows_failed"] and r["severity"] == "error"]
