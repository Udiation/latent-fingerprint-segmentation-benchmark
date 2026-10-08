"""Subject-wise train/val/test split. One split file is used for every method and
every subset (latent, live-scan, palm), so no person appears in both train and test."""
import argparse
import json

import numpy as np

import _path  # noqa: F401
from segbench.io import load_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = load_manifest(a.manifest)
    subjects = np.array(sorted(df.subject_id.unique()))
    rng = np.random.default_rng(a.seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_tr, n_va = int(round(n * a.train)), int(round(n * a.val))
    splits = {s: "train" for s in subjects[:n_tr]}
    splits.update({s: "val" for s in subjects[n_tr:n_tr + n_va]})
    splits.update({s: "test" for s in subjects[n_tr + n_va:]})

    with open(a.out, "w") as f:
        json.dump({"seed": a.seed, "splits": splits}, f, indent=1)

    df["split"] = df.subject_id.map(splits)
    has_gt = df.mask_path.str.len() > 0
    print(f"{n} subjects -> train {n_tr}, val {n_va}, test {n - n_tr - n_va}")
    print("images with ground truth, by subset and split:")
    print(df[has_gt].groupby(["subset", "split"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
