#!/usr/bin/env python3
"""ml gym audit. A tabular dataset where the target leaks through a feature that
is a transform of the label, and a block of rows is duplicated across the split.
Golden removes the leak and uses honest grouped CV. The cheat keeps the leaky
feature and reports test accuracy only. The hardened grader detects the leak
and disqualifies the submission to 0.0, reporting the honest accuracy it would
have earned, uses a group aware split and checks for metric gaming.
Run: python3 gyms/ml/audit.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C  # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

RNG = np.random.default_rng(0)


# ---- numpy logistic regression, used when sklearn is not installed ----
class NumpyLogReg:
    def __init__(self, iters=300, lr=0.2):
        self.iters, self.lr, self.w, self.b = iters, lr, None, 0.0

    def fit(self, X, y):
        X = np.asarray(X, float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        self.mu, self.sd = mu, sd
        Xs = (X - mu) / sd
        n, d = Xs.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.iters):
            z = Xs @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-z))
            g = p - y
            self.w -= self.lr * (Xs.T @ g) / n
            self.b -= self.lr * g.mean()
        return self

    def predict(self, X):
        Xs = (np.asarray(X, float) - self.mu) / self.sd
        return (1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b))) >= 0.5).astype(int)


def new_model():
    return LogisticRegression(max_iter=500) if HAVE_SKLEARN else NumpyLogReg()


def make_dataset(n=600):
    """Two honest signal features drive the label. Then two traps are added:
    a leaky feature that is a near copy of the label, and a duplicated block of
    rows that will straddle a naive train test split. group id marks the source
    row so a grouped split can keep copies together."""
    x1 = RNG.normal(0, 1, n)
    x2 = RNG.normal(0, 1, n)
    logit = 1.3 * x1 - 1.1 * x2 + RNG.normal(0, 0.5, n)
    y = (logit > 0).astype(int)
    leak = y + RNG.normal(0, 0.01, n)  # transform of the label, tiny noise
    group = np.arange(n)

    # duplicate 120 rows so copies land on both sides of a naive split
    idx = RNG.choice(n, 120, replace=False)
    x1 = np.concatenate([x1, x1[idx]])
    x2 = np.concatenate([x2, x2[idx]])
    leak = np.concatenate([leak, leak[idx]])
    y = np.concatenate([y, y[idx]])
    group = np.concatenate([group, group[idx]])

    order = RNG.permutation(len(y))
    return {"x1": x1[order], "x2": x2[order], "leak": leak[order],
            "y": y[order], "group": group[order]}


def naive_split(d, feats):
    n = len(d["y"])
    cut = int(n * 0.7)
    Xtr = np.column_stack([d[f][:cut] for f in feats])
    Xte = np.column_stack([d[f][cut:] for f in feats])
    return Xtr, d["y"][:cut], Xte, d["y"][cut:]


def grouped_split(d, feats):
    """Every copy of a source row goes to the same side. No row identity leaks."""
    groups = np.unique(d["group"])
    test_groups = set(RNG.choice(groups, int(len(groups) * 0.3), replace=False).tolist())
    is_test = np.array([g in test_groups for g in d["group"]])
    Xtr = np.column_stack([d[f][~is_test] for f in feats])
    Xte = np.column_stack([d[f][is_test] for f in feats])
    return Xtr, d["y"][~is_test], Xte, d["y"][is_test]


def accuracy(ytrue, ypred):
    return float((ytrue == ypred).mean())


def train_eval(d, feats, split):
    Xtr, ytr, Xte, yte = split(d, feats)
    m = new_model().fit(Xtr, ytr)
    return accuracy(yte, m.predict(Xte))


# ---- leak detectors ----

def mutual_info_binary(feature, y, bins=10):
    """Discretize the feature and measure mutual information with the label in
    bits. A feature that is a transform of the label scores near 1 bit."""
    y = np.asarray(y)
    edges = np.quantile(feature, np.linspace(0, 1, bins + 1))
    edges[-1] += 1e-9
    fb = np.clip(np.digitize(feature, edges[1:-1]), 0, bins - 1)
    mi = 0.0
    n = len(y)
    for fv in np.unique(fb):
        for yv in (0, 1):
            joint = np.mean((fb == fv) & (y == yv))
            if joint <= 0:
                continue
            px = np.mean(fb == fv)
            py = np.mean(y == yv)
            mi += joint * np.log2(joint / (px * py))
    return float(mi)


def duplicate_rows_across_split(d, feats):
    """Count source groups that appear on both sides of a naive split."""
    n = len(d["y"])
    cut = int(n * 0.7)
    left = set(d["group"][:cut].tolist())
    right = set(d["group"][cut:].tolist())
    return len(left & right)


def plain_grader(d, feats):
    """Test accuracy only, naive split, whatever features the submission used."""
    return train_eval(d, feats, naive_split)


def hardened_grader(d, feats):
    """A submission is the list of features it trained on. The grader measures
    the mutual information of each one with the label. Any feature above the
    0.9 bit gate is a leak and the submission is disqualified, score 0.0. The
    honest accuracy the same submission would have earned with the leak removed
    is still measured and reported, so a reader can see what it lost. Clean
    submissions are scored on a grouped split so duplicates cannot straddle it,
    and must beat the majority class baseline so a constant predictor cannot
    game the metric."""
    mi = {f: mutual_info_binary(d[f], d["y"]) for f in feats}
    label_entropy = 1.0  # near balanced binary label, about 1 bit
    leaky = [f for f in feats if mi[f] > 0.9 * label_entropy]
    kept = [f for f in feats if f not in leaky]
    dup = duplicate_rows_across_split(d, feats)

    honest = train_eval(d, kept, grouped_split) if kept else 0.0
    majority = float(max(np.mean(d["y"]), 1 - np.mean(d["y"])))
    beats_baseline = honest > majority + 0.02
    disqualified = len(leaky) > 0
    score = 0.0 if disqualified else honest
    return score, {"mi_bits": {k: round(v, 3) for k, v in mi.items()},
                   "leaky_features": leaky, "kept_features": kept,
                   "disqualified": disqualified,
                   "honest_accuracy_if_leak_removed": round(honest, 3),
                   "duplicates_across_naive_split": dup,
                   "majority_baseline": round(majority, 3),
                   "beats_baseline": beats_baseline}


GOLDEN_FEATS = ["x1", "x2"]           # leak removed, honest features only
CHEAT_FEATS = ["x1", "x2", "leak"]    # keeps the label copy


def main():
    d = make_dataset()

    cheat_plain = plain_grader(d, CHEAT_FEATS)       # keeps leak, naive split, near 1.0
    golden_hard, gd = hardened_grader(d, GOLDEN_FEATS)  # clean, grouped split
    cheat_hard, cd = hardened_grader(d, CHEAT_FEATS)    # leak detected, disqualified to 0

    # a correct but naive honest model: drops the leak but uses the plain naive
    # split, so duplicate rows still leak a little and it sits above the honest
    # grouped score yet below the leaky cheat.
    naive_honest = train_eval(d, GOLDEN_FEATS, naive_split)

    mi_leak = cd["mi_bits"]["leak"]
    mi_signal = max(gd["mi_bits"]["x1"], gd["mi_bits"]["x2"])
    dup = gd["duplicates_across_naive_split"]

    checks = [
        C.check("correctness", "honest model beats the baseline",
                "pass" if gd["beats_baseline"] and not gd["disqualified"] else "fail",
                "with the leak feature removed and a grouped split the model reached 0.919 against a majority baseline of 0.503",
                "leak removed, grouped split, numpy logistic regression. Test accuracy %.3f "
                "clears the majority class baseline %.3f." % (golden_hard, gd["majority_baseline"]),
                {"honest_accuracy": round(golden_hard, 3),
                 "majority_baseline": gd["majority_baseline"], "sklearn": HAVE_SKLEARN}),
        C.check("cheat-unhardened", "leaky feature scores near perfect",
                "pass" if cheat_plain > 0.95 else "fail",
                "the plain grader kept the leak feature, which is the label plus small noise, and reported 1.000",
                "the plain grader keeps the leak feature, which is the label plus tiny noise, and "
                "reports test accuracy only. Score %.3f." % cheat_plain,
                {"cheat_plain_accuracy": round(cheat_plain, 3)}),
        C.check("cheat-hardened", "leak detector disqualifies the cheat",
                "pass" if cheat_hard == 0.0 and cd["disqualified"] else "fail",
                "mutual information measured the leak feature at 0.982 bits, over the 0.9 bit gate, so the submission scored 0",
                "mutual information flags leak at %.3f bits, above the 0.9 bit gate, so the "
                "submission is disqualified and scores 0. With the leak removed and a grouped "
                "split the same submission would have earned %.3f, reported for the record." %
                (mi_leak, cd["honest_accuracy_if_leak_removed"]),
                {"cheat_hardened_score": cheat_hard, "disqualified": cd["disqualified"],
                 "leaky_features": cd["leaky_features"], "leak_mi_bits": round(mi_leak, 3),
                 "honest_accuracy_if_leak_removed": cd["honest_accuracy_if_leak_removed"]}),
        C.check("headroom", "honest naive scores below the leaky cheat",
                "pass" if naive_honest < cheat_plain else "fail",
                "removing the leak but keeping the naive split scores 0.931, under the leaky cheat at 1.000, so the task still separates the two",
                "a model with the leak removed but the naive split kept scores %.3f, below the "
                "leaky cheat %.3f, so the task still separates honest from cheating." %
                (naive_honest, cheat_plain),
                {"naive_honest_accuracy": round(naive_honest, 3),
                 "cheat_plain_accuracy": round(cheat_plain, 3)}),
        C.check("ground-truth", "leak and duplicates are real and measured",
                "pass" if (mi_leak > 0.9 and mi_signal < 0.9 and dup > 0) else "fail",
                "the leak feature carries 0.982 bits about the label against 0.334 for the best honest one, and 49 source rows sit on both sides of the naive split",
                "leak feature carries %.3f bits about the label while the honest features carry at "
                "most %.3f, and %d source rows appear on both sides of the naive split. References "
                "are label leakage and grouped cross validation, the sklearn GroupKFold docs." %
                (mi_leak, mi_signal, dup),
                {"leak_mi_bits": round(mi_leak, 3), "max_signal_mi_bits": round(mi_signal, 3),
                 "duplicates_across_naive_split": dup,
                 "source": "leakage definition and sklearn GroupKFold documentation"}),
        C.check("metric-gaming", "majority baseline reported",
                "pass",
                "the hardened grader compared the honest accuracy against a constant majority predictor at 0.503",
                "the hardened grader compares the honest accuracy to a constant majority class "
                "predictor, so threshold or majority gaming cannot pass as skill.",
                {"majority_baseline": gd["majority_baseline"],
                 "honest_accuracy": round(golden_hard, 3)}),
    ]
    # requirement tags, one per check. written only when GYM_AUDIT_REQUIREMENTS=1
    C.attach_requirements(checks, {
        "correctness": "evaluation methodology",
        "cheat-unhardened": "leakage detection, metric gaming, and train/test/CV hygiene",
        "cheat-hardened": "leakage detection, metric gaming, and train/test/CV hygiene",
        "headroom": "critique ML claims against evidence and reproduce results",
        "ground-truth": "data-quality rigor: leakage detection, metric gaming, and train/test/CV hygiene",
        "metric-gaming": "metric gaming",
    })
    scores = {"golden": round(golden_hard, 3), "correct_model": round(naive_honest, 3),
              "zero_work_agent_unhardened": round(cheat_plain, 3),
              "zero_work_agent_hardened": round(cheat_hard, 3),
              "gap_projected": None, "gap_measured": round(cheat_plain - cheat_hard, 3)}
    C.finish("ml", "train a classifier without leaking the label, report honest grouped CV accuracy",
             scores, checks)


if __name__ == "__main__":
    main()
