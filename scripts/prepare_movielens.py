#!/usr/bin/env python3
"""Download a real MovieLens dataset (GroupLens) and convert it to this pipeline's raw schema.

    python scripts/prepare_movielens.py --dataset ml-latest-small      # ~1 MB: 100k ratings, 610 users, 9.7k movies (recommended)
    python scripts/prepare_movielens.py --dataset ml-1m                # 1M ratings, 6k users, 3.7k movies (has age + gender)
    python scripts/prepare_movielens.py --zip ~/Downloads/ml-1m.zip    # use a zip you downloaded yourself (no network needed)
    python scripts/prepare_movielens.py --dir ~/Downloads/ml-latest-small   # or an already-extracted folder

Writes data/movielens/raw/{users,items,interactions}.csv  (then run:  python run_pipeline.py --config configs/config.movielens.yaml)

Mapping to the pipeline schema
  interactions  user_id, item_id, rating, timestamp  <- ratings (epoch seconds -> 'YYYY-MM-DD HH:MM:SS')
  items         item_id, title, genres, release_year <- movies ('(no genres listed)' -> empty; year parsed from 'Title (1995)')
  users         user_id, age, gender, country, signup_date
                  ml-latest-small has NO user table: user_id and signup_date (= date of the user's first rating) are derived; age/gender/country are
                  left empty (validation reports them as warnings, preparation imputes them).
                  ml-1m provides gender and age *bracket*: age = midpoint of the bracket; country is empty.
MovieLens data is (c) GroupLens Research, free for research/education use but NOT to be redistributed - download it yourself and cite:
Harper & Konstan (2015), "The MovieLens Datasets: History and Context", ACM TiiS 5(4).
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/{name}.zip"
AGE_MIDPOINT = {1: 16, 18: 21, 25: 29, 35: 39, 45: 47, 50: 52, 56: 60}   # ml-1m age brackets -> approximate age


def _items(movies: pd.DataFrame) -> pd.DataFrame:
    genres = movies["genres"].where(movies["genres"] != "(no genres listed)")
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    return pd.DataFrame({"item_id": movies["item_id"], "title": movies["title"].str.strip(), "genres": genres, "release_year": year})


def _interactions(r: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(r["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    return pd.DataFrame({"user_id": r["user_id"], "item_id": r["item_id"], "rating": r["rating"], "timestamp": ts})


def _users(r: pd.DataFrame, demo: pd.DataFrame | None) -> pd.DataFrame:
    first = pd.to_datetime(r.groupby("user_id")["timestamp"].min(), unit="s").dt.strftime("%Y-%m-%d")
    users = pd.DataFrame({"user_id": first.index, "age": None, "gender": None, "country": None, "signup_date": first.to_numpy()})
    if demo is not None:
        users = users.drop(columns=["age", "gender"]).merge(demo, on="user_id", how="left")[["user_id", "age", "gender", "country", "signup_date"]]
    return users


def convert_latest(ratings: pd.DataFrame, movies: pd.DataFrame):
    """ml-latest-small / ml-latest / ml-25m layout (ratings.csv + movies.csv)."""
    r = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"})
    m = movies.rename(columns={"movieId": "item_id"})
    return _users(r, None), _items(m), _interactions(r)


def convert_1m(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame):
    """ml-1m layout (ratings.dat / movies.dat / users.dat, '::' separated)."""
    demo = pd.DataFrame({"user_id": users["user_id"], "age": users["age_code"].map(AGE_MIDPOINT), "gender": users["gender"], "country": None})
    demo = demo.drop(columns=["country"])
    return _users(ratings, demo), _items(movies), _interactions(ratings)


def load(root: Path):
    """`root` = folder containing the MovieLens files (searched one level deep)."""
    find = lambda n: next(iter(list(root.glob(n)) + list(root.glob(f"*/{n}"))), None)  # noqa: E731
    if find("ratings.csv") is not None:
        return convert_latest(pd.read_csv(find("ratings.csv")), pd.read_csv(find("movies.csv")))
    if find("ratings.dat") is not None:
        rd = lambda n, cols: pd.read_csv(find(n), sep="::", engine="python", names=cols, encoding="latin-1")  # noqa: E731
        return convert_1m(rd("ratings.dat", ["user_id", "item_id", "rating", "timestamp"]),
                          rd("movies.dat", ["item_id", "title", "genres"]),
                          rd("users.dat", ["user_id", "gender", "age_code", "occupation", "zip"]))
    raise FileNotFoundError(f"No MovieLens ratings.csv or ratings.dat found under {root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--dataset", default="ml-latest-small", help="name on files.grouplens.org (ml-latest-small, ml-1m, ...)")
    src.add_argument("--zip", help="path to a MovieLens zip you downloaded manually")
    src.add_argument("--dir", help="path to an extracted MovieLens folder")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "movielens" / "raw"))
    a = ap.parse_args()
    tmp = Path(a.out).parent / "_download"
    if a.dir:
        root = Path(a.dir)
    else:
        tmp.mkdir(parents=True, exist_ok=True)
        if a.zip:
            data = Path(a.zip).read_bytes()
        else:
            url = URL.format(name=a.dataset)
            print(f"Downloading {url} ...")
            data = urllib.request.urlopen(url, timeout=120).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
        root = tmp
    users, items, inter = load(root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    users.to_csv(out / "users.csv", index=False)
    items.to_csv(out / "items.csv", index=False)
    inter.to_csv(out / "interactions.csv", index=False)
    print(f"Wrote {out}: users={len(users):,} items={len(items):,} interactions={len(inter):,}")
    print("Next:  python run_pipeline.py --config configs/config.movielens.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
