#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
echo_chamber_demo.py
----------------------------------
Lightweight ML to mitigate echo chambers:
- Synthetic dataset generator
- Logistic Regression click model
- Greedy vs MMR ranking
- Evaluation: Precision@k, Intra-List Similarity (ILS), Topic Coverage, Mean Slant
- Saves CSV summaries

Run:
    python echo_chamber_demo.py --items 150 --users 40 --k 10 --lam 0.7 --seed 21

Outputs:
    /mnt/data/evaluation_results_small.csv
    /mnt/data/evaluation_summary_small.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Synthetic data generation
# -----------------------------

TOPICS = ["politics_left", "politics_right", "sports", "entertainment", "tech"]
TOPIC_TO_SLANT = {"politics_left": -1.0, "politics_right": +1.0, "sports": 0.0, "entertainment": 0.0, "tech": 0.0}

BASE_VOCAB = {
    "politics_left": ["equality","policy","rights","union","social","welfare","climate","labor"],
    "politics_right": ["market","freedom","security","tradition","tax","border","nation","family"],
    "sports": ["match","goal","tournament","league","coach","player","score","win"],
    "entertainment": ["movie","music","celebrity","drama","series","theater","album","festival"],
    "tech": ["ai","software","startup","chip","cloud","device","data","gadget"],
}

def make_doc(rng: np.random.Generator, topic: str, n_words: int = 20) -> str:
    words = rng.choice(BASE_VOCAB[topic], size=n_words, replace=True)
    return f"{topic} " + " ".join(words)

def base_click_prob(user_row: pd.Series, item_row: pd.Series) -> float:
    # Topic alignment
    if item_row.topic == user_row.like_main:
        p = 0.60
    elif item_row.topic == user_row.like_aux:
        p = 0.35
    else:
        p = 0.08

    # Ideology effect on political topics
    if "politics" in item_row.topic and user_row.ideology_strength > 0:
        aligned_left = (item_row.topic == "politics_left" and user_row.like_main == "politics_left")
        aligned_right = (item_row.topic == "politics_right" and user_row.like_main == "politics_right")
        if aligned_left or aligned_right:
            p = min(0.95, p + 0.20 * user_row.ideology_strength)
        else:
            p = max(0.01, p - 0.20 * user_row.ideology_strength)
    return p

# -----------------------------
# Feature construction
# -----------------------------

def user_profile_vec(uid: int, inter: pd.DataFrame, X_items, news: pd.DataFrame, rng: np.random.Generator):
    clicked_ids = inter[(inter.user_id==uid) & (inter.clicked==1)].news_id.values
    if len(clicked_ids)==0:
        ridx = rng.choice(news.index, size=min(6, len(news)), replace=False)
        return X_items[ridx].mean(axis=0)
    return X_items[clicked_ids].mean(axis=0)

def build_feature_row(urow: pd.Series, nrow: pd.Series, user_profiles, X_items, topic_to_idx):
    uvec = user_profiles[urow.user_id]
    ivec = X_items[nrow.news_id]
    sim = float(cosine_similarity(uvec, ivec).ravel()[0])
    topic_oh = np.zeros(len(TOPICS))
    topic_oh[topic_to_idx[nrow.topic]] = 1.0
    # features: [cosine_sim, one-hot topic (5), item slant, user ideology_strength]
    return np.hstack([sim, topic_oh, [nrow.slant, urow.ideology_strength]])

# -----------------------------
# Ranking policies
# -----------------------------

def predict_click_prob(uid: int, candidate_ids, users, news, user_profiles, X_items, clf, topic_to_idx):
    urow = users.loc[users.user_id==uid].iloc[0]
    feats = np.vstack([build_feature_row(urow, news.loc[news.news_id==nid].iloc[0],
                                         user_profiles, X_items, topic_to_idx) for nid in candidate_ids])
    return clf.predict_proba(feats)[:,1]

def greedy_rank(uid: int, candidate_ids, k: int, users, news, user_profiles, X_items, clf, topic_to_idx):
    probs = predict_click_prob(uid, candidate_ids, users, news, user_profiles, X_items, clf, topic_to_idx)
    order = np.argsort(-probs)[:k]
    return list(np.array(candidate_ids)[order])

def mmr_rank(uid: int, candidate_ids, k: int, lam: float, users, news, user_profiles, X_items, clf, topic_to_idx):
    rel = predict_click_prob(uid, candidate_ids, users, news, user_profiles, X_items, clf, topic_to_idx)
    selected = []
    remaining = list(candidate_ids)
    for _ in range(min(k, len(remaining))):
        if not selected:
            j = int(np.argmax(rel))
            selected.append(remaining[j])
            del remaining[j]
            rel = np.delete(rel, j)
        else:
            max_sims = []
            sel_vecs = X_items[selected]  # sparse CSR okay
            for nid in remaining:
                ivec = X_items[nid]
                max_sims.append(float(cosine_similarity(ivec, sel_vecs).max()))
            max_sims = np.array(max_sims)
            scores = lam * rel - (1 - lam) * max_sims
            j = int(np.argmax(scores))
            selected.append(remaining[j])
            del remaining[j]
            rel = np.delete(rel, j)
    return selected

# -----------------------------
# Metrics
# -----------------------------

def true_click_prob(uid: int, nid: int, users, news) -> float:
    urow = users.loc[users.user_id==uid].iloc[0]
    nrow = news.loc[news.news_id==nid].iloc[0]
    return base_click_prob(urow, nrow)

def precision_at_k(uid: int, ranked_ids, k: int, users, news, rng: np.random.Generator) -> float:
    picks = ranked_ids[:k]
    clicks = [int(rng.random() < true_click_prob(uid, nid, users, news)) for nid in picks]
    return float(np.mean(clicks)) if len(clicks)>0 else 0.0

def intra_list_similarity(ranked_ids, X_items) -> float:
    if len(ranked_ids) <= 1:
        return 1.0
    Xi = X_items[ranked_ids]
    S = cosine_similarity(Xi)
    m = S.shape[0]
    mask = ~np.eye(m, dtype=bool)
    return float(S[mask].mean())

def topic_coverage(ranked_ids, news) -> int:
    return int(news.loc[news.news_id.isin(ranked_ids), "topic"].nunique())

def mean_slant(ranked_ids, news) -> float:
    return float(news.loc[news.news_id.isin(ranked_ids), "slant"].mean()) if len(ranked_ids)>0 else 0.0

# -----------------------------
# Main experiment
# -----------------------------

def run(items: int, users_n: int, k: int, lam: float, seed: int):
    rng = np.random.default_rng(seed)

    # Items
    news = pd.DataFrame({
        "news_id": np.arange(items),
        "topic": rng.choice(TOPICS, size=items, replace=True)
    })
    news["slant"] = news["topic"].map(TOPIC_TO_SLANT).astype(float)
    news["text"] = news["topic"].apply(lambda t: make_doc(rng, t, n_words=20))

    # Users
    users = pd.DataFrame({"user_id": np.arange(users_n)})
    users["like_main"] = rng.choice(TOPICS, size=users_n, replace=True)
    users["like_aux"]  = rng.choice(TOPICS, size=users_n, replace=True)
    users["ideology_strength"] = rng.choice([0.0, 0.5, 1.0], size=users_n, p=[0.6, 0.3, 0.1])

    # Interactions
    rows = []
    for _, u in users.iterrows():
        seen_ids = rng.choice(news.news_id.values, size=min(60, items), replace=False)
        for nid in seen_ids:
            it = news.loc[news.news_id == nid].iloc[0]
            p = base_click_prob(u, it)
            clicked = int(rng.random() < p)
            rows.append((u.user_id, nid, clicked))
    inter = pd.DataFrame(rows, columns=["user_id","news_id","clicked"])

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=1500)
    X_items = tfidf.fit_transform(news["text"]).tocsr()

    # User profiles
    user_profiles = {uid: user_profile_vec(uid, inter, X_items, news, rng) for uid in users.user_id.values}
    topic_to_idx = {t:i for i,t in enumerate(TOPICS)}

    # Train logistic regression
    train_sample = inter.sample(frac=0.7, random_state=1)
    X_train = np.vstack([
        build_feature_row(users.loc[users.user_id==r.user_id].iloc[0],
                          news.loc[news.news_id==r.news_id].iloc[0],
                          user_profiles, X_items, topic_to_idx)
        for _, r in train_sample.iterrows()
    ])
    y_train = train_sample["clicked"].values
    clf = LogisticRegression(max_iter=600)
    clf.fit(X_train, y_train)

    # Evaluate
    def evaluate(n_users_eval=25):
        res = []
        eval_users = rng.choice(users.user_id.values, size=min(n_users_eval, users_n), replace=False)
        all_candidates = list(news.news_id.values)
        for uid in eval_users:
            cand = list(rng.choice(all_candidates, size=min(40, items), replace=False))
            ranked_g = greedy_rank(uid, cand, k, users, news, user_profiles, X_items, clf, topic_to_idx)
            ranked_m = mmr_rank(uid, cand, k, lam, users, news, user_profiles, X_items, clf, topic_to_idx)
            for name, ranked in [("greedy", ranked_g), ("mmr", ranked_m)]:
                res.append({
                    "user_id": uid,
                    "policy": name,
                    "precision@k": precision_at_k(uid, ranked, k, users, news, rng),
                    "ILS": intra_list_similarity(ranked, X_items),
                    "topic_coverage": topic_coverage(ranked, news),
                    "mean_slant": mean_slant(ranked, news)
                })
        return pd.DataFrame(res)

    df_res = evaluate(n_users_eval=25)
    summary = df_res.groupby("policy")[["precision@k","ILS","topic_coverage","mean_slant"]].mean().round(3)

    outdir = Path("/mnt/data"); outdir.mkdir(exist_ok=True, parents=True)
    df_res.to_csv(outdir / "evaluation_results_small.csv", index=False)
    summary.to_csv(outdir / "evaluation_summary_small.csv")

    print("== Summary (mean across users) ==")
    print(summary)
    print("\nSaved:")
    print(" - /mnt/data/evaluation_results_small.csv")
    print(" - /mnt/data/evaluation_summary_small.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=150)
    parser.add_argument("--users", type=int, default=40)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    run(items=args.items, users_n=args.users, k=args.k, lam=args.lam, seed=args.seed)
