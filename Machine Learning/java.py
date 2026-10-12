import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# ---- 1) Synthetic news ----
rng = np.random.default_rng(42)
topics = ["politics","entertainment","tech","sports"]
def make_doc(topic):
    base = {
        "politics": ["election","policy","debate","government","vote"],
        "entertainment": ["movie","celebrity","music","tv","drama"],
        "tech": ["ai","software","startup","chip","gadget"],
        "sports": ["match","goal","tournament","league","coach"]
    }[topic]
    words = rng.choice(base, size=20, replace=True)
    return f"{topic} " + " ".join(words)

n_items = 400
news = pd.DataFrame({
    "news_id": range(n_items),
    "topic": rng.choice(topics, size=n_items, replace=True)
})
news["text"] = news["topic"].apply(make_doc)

# ---- 2) Synthetic users & clicks ----
n_users = 60
users = pd.DataFrame({"user_id": range(n_users)})
# mỗi user thích 1 chủ đề chính + 1 phụ
users["like_main"] = rng.choice(topics, size=n_users)
users["like_aux"]  = rng.choice([t for t in topics], size=n_users)

rows = []
for _, u in users.iterrows():
    for _, it in news.sample(120, random_state=int(u.user_id)).iterrows():
        # xác suất click nếu trùng chủ đề
        p = 0.55 if it.topic==u.like_main else (0.30 if it.topic==u.like_aux else 0.10)
        click = rng.random() < p
        rows.append((u.user_id, it.news_id, click))
inter = pd.DataFrame(rows, columns=["user_id","news_id","clicked"])

data = inter.merge(news, on="news_id")

# ---- 3) Vectorize & train a simple click model (logistic) ----
tfidf = TfidfVectorizer(max_features=2000)
X_items = tfidf.fit_transform(news["text"])

# encode user preference as 4-d onehot (topic prior)
topic2idx = {t:i for i,t in enumerate(topics)}
def user_vec(u):
    v = np.zeros(len(topics))
    v[topic2idx[u.like_main]] += 1.0
    v[topic2idx[u.like_aux]]  += 0.5
    return v

# build training matrix by concatenating [item tfidf reduced with dot to topic centroids] + user onehot
# quick trick: approximate item→topic by cosine to each topic centroid
centroids = {}
for t in topics:
    idx = news.index[news.topic==t]
    centroids[t] = X_items[idx].mean(axis=0)

def item_topic_scores(irow):
    # 4-d similarity to topic centroids
    sims = []
    x = X_items[irow.news_id]
    for t in topics:
        sims.append(float(cosine_similarity(x, centroids[t]).ravel()[0]))
    return np.array(sims)

feat_rows, y = [], []
for _, r in data.iterrows():
    u = users.loc[users.user_id==r.user_id].iloc[0]
    feats = np.hstack([item_topic_scores(r), user_vec(u)])
    feat_rows.append(feats)
    y.append(int(r.clicked))
X = np.vstack(feat_rows); y = np.array(y)

Xtr, Xte, ytr, yte, dtr, dte = train_test_split(X, y, data[["user_id","news_id"]], test_size=0.25, random_state=0, stratify=y)

clf = LogisticRegression(max_iter=500)
clf.fit(Xtr, ytr)

# ---- 4) Ranking policies ----
def predict_click_prob(user_id, candidate_ids):
    u = users.loc[users.user_id==user_id].iloc[0]
    U = np.tile(user_vec(u), (len(candidate_ids),1))
    IT = np.vstack([item_topic_scores(pd.Series({"news_id": nid})) for nid in candidate_ids])
    Xp = np.hstack([IT, U])
    return clf.predict_proba(Xp)[:,1]

def rank_greedy(user_id, candidate_ids, k=10):
    p = predict_click_prob(user_id, candidate_ids)
    return [nid for nid in np.array(candidate_ids)[np.argsort(-p)][:k]]

def rank_mmr(user_id, candidate_ids, k=10, lam=0.7):
    # MMR: select by lambda*relevance - (1-lambda)*max_similarity_with_selected
    p = predict_click_prob(user_id, candidate_ids)
    selected, pool = [], list(candidate_ids)
    # item-item similarity from TF-IDF
    sims = cosine_similarity(X_items[pool], X_items[pool])
    for _ in range(min(k, len(pool))):
        if not selected:
            j = int(np.argmax(p))
            selected.append(pool[j]);
            # remove
            del pool[j]; p = np.delete(p, j); sims = np.delete(np.delete(sims, j, 0), j, 1)
        else:
            scores = []
            for j in range(len(pool)):
                # max sim to already selected
                max_sim = max(cosine_similarity(X_items[pool[j]], X_items[[s]]).ravel()[0] for s in selected)
                scores.append(lam*p[j] - (1-lam)*max_sim)
            j = int(np.argmax(scores))
            selected.append(pool[j])
            del pool[j]; p = np.delete(p, j); sims = np.delete(np.delete(sims, j, 0), j, 1)
    return selected

# ---- 5) Metrics: Precision@k (proxy CTR), Intra-List Similarity, Topic Coverage ----
def metrics_for_user(user_id, k=10, policy="greedy"):
    cands = list(news.news_id.values)
    if policy=="greedy":
        ranked = rank_greedy(user_id, cands, k)
    else:
        ranked = rank_mmr(user_id, cands, k)
    # precision@k using held-out labels (approx via true click prob generation)
    # here we approximate by our synthetic rule again for evaluation
    u = users.loc[users.user_id==user_id].iloc[0]
    def true_click_prob(nid):
        t = news.loc[news.news_id==nid, "topic"].iloc[0]
        return 0.55 if t==u.like_main else (0.30 if t==u.like_aux else 0.10)
    prec = np.mean([true_click_prob(nid) > 0.5 for nid in ranked])

    # Intra-List Similarity
    Xi = X_items[ranked]
    ils = cosine_similarity(Xi).mean()

    # Topic coverage
    cov = news.loc[news.news_id.isin(ranked), "topic"].nunique()
    return prec, ils, cov

def evaluate(k=10):
    out = []
    for uid in users.user_id.sample(30, random_state=7):
        for pol in ["greedy","mmr"]:
            prec, ils, cov = metrics_for_user(uid, k=k, policy=pol)
            out.append({"user_id":uid, "policy":pol, "precision@k":prec, "ILS":ils, "topic_coverage":cov})
    return pd.DataFrame(out)

res = evaluate(k=10)
print(res.groupby("policy")[["precision@k","ILS","topic_coverage"]].mean().round(3))
