import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Third‑party libraries used for the CF models
from surprise import Dataset, Reader, KNNBasic, NMF

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")

models = (
    "Course Similarity",  # 0
    "User Profile",       # 1  (content‑based, no training yet)
    "Clustering",         # 2  (content‑based, no training yet)
    "Clustering with PCA",# 3  (content‑based, no training yet)
    "KNN",                # 4  (collaborative filtering)
    "NMF",                # 5  (matrix factorisation)
    "Neural Network",     # 6  (future work)
    "Regression with Embedding Features",       # 7  (future work)
    "Classification with Embedding Features"    # 8  (future work)
)


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


def _model_path(model_name: str) -> str:
    """Create a consistent file name for persisting the fitted model."""
    safe_name = model_name.lower().replace(" ", "_")
    return os.path.join("trained_models", f"{safe_name}.pkl")


def _ensure_model_dir():
    os.makedirs("trained_models", exist_ok=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model_name: str, params: Optional[Dict] = None):
    """Fit *model_name* on the current ratings data and persist the model.

    Parameters
    ----------
    model_name : str
        One of the names in the *models* tuple.
    params : dict, optional
        Hyper‑parameters specific to the model.  Leave empty for sensible
        defaults.
    """
    if params is None:
        params = {}

    _ensure_model_dir()

    # -------------------------------------------------------
    # Content‑based models that rely only on pre‑computed files
    # -------------------------------------------------------
    if model_name == models[0]:  # "Course Similarity"
        # nothing to fit – similarity matrix already exists on disk
        return None

    # -------------------------------------------------------
    # Collaborative filtering models — using the *surprise* package
    # -------------------------------------------------------
    if model_name in (models[4], models[5]):  # KNN or NMF
        ratings_df = load_ratings()
        reader = Reader(rating_scale=(ratings_df.rating.min(), ratings_df.rating.max()))
        data = Dataset.load_from_df(ratings_df[["user", "item", "rating"]], reader)
        trainset = data.build_full_trainset()

        if model_name == models[4]:  # KNN item‑based CF
            sim_options = {
                "name": params.get("similarity", "cosine"),
                "user_based": params.get("user_based", False),  # False → item‑based
            }
            algo = KNNBasic(sim_options=sim_options, verbose=params.get("verbose", False))
        else:  # models[5] → NMF
            algo = NMF(
                n_factors=params.get("n_factors", 20),
                n_epochs=params.get("n_epochs", 50),
                random_state=params.get("random_state", 2024),
                reg_pu=params.get("reg_pu", 0.06),
                reg_qi=params.get("reg_qi", 0.06),
            )

        algo.fit(trainset)

        # Persist
        with open(_model_path(model_name), "wb") as f:
            pickle.dump(algo, f)
        return algo

    # -------------------------------------------------------
    # Placeholder for future models (User‑profile, Clustering, …)
    # -------------------------------------------------------
    raise NotImplementedError(
        f"Training routine for ‘{model_name}’ has not been implemented yet.")


# ---------------------------------------------------------------------------
# Prediction / inference
# ---------------------------------------------------------------------------

def _load_or_train(model_name: str, params: Dict):
    """Load a persisted model or (re)train if the pickle is missing."""
    path = _model_path(model_name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    # fall back to on‑the‑fly training so predict() never fails
    return train(model_name, params)


def predict(model_name: str, user_ids: List[int], params: Optional[Dict] = None) -> pd.DataFrame:
    """Return a dataframe with USER, COURSE_ID, SCORE columns.

    * For content‑based recommendations SCORE is a cosine similarity
      (0‑1).  The *sim_threshold* param (percentage) filters low scores.
    * For CF models SCORE is the estimated rating (higher ⇒ better).

    Parameters
    ----------
    model_name : str
        One of the names declared at the top of this file.
    user_ids : List[int]
        Target user ids we want recommendations for.
    params : dict, optional
        Prediction‑time switches (e.g. sim_threshold, top_n).
    """
    if params is None:
        params = {}

    # ------ Shared data used by several models ------
    ratings_df = load_ratings()
    idx_id_dict, id_idx_dict = get_doc_dicts()
    all_course_ids = set(idx_id_dict.values())

    users: List[int] = []
    courses: List[int] = []
    scores: List[float] = []

    # -------------------------------------------------------
    # 1️⃣  Pure content‑based similarity model (already half‑done)
    # -------------------------------------------------------
    if model_name == models[0]:  # "Course Similarity"
        sim_matrix = load_course_sims().to_numpy()
        sim_threshold = params.get("sim_threshold", 60) / 100.0  # default 0.60

        for uid in user_ids:
            enrolled_course_ids = ratings_df.loc[ratings_df.user == uid, "item"].tolist()
            if not enrolled_course_ids:
                continue

            recs = course_similarity_recommendations(
                idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix
            )
            for cid, score in recs.items():
                if score >= sim_threshold:
                    users.append(uid)
                    courses.append(cid)
                    scores.append(float(score))

        return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores})

    # -------------------------------------------------------
    # 2️⃣  Collaborative filtering models (KNN / NMF)
    # -------------------------------------------------------
    if model_name in (models[4], models[5]):
        algo = _load_or_train(model_name, params)

        top_n: Optional[int] = params.get("top_n")  # None → no cut‑off

        for uid in user_ids:
            seen = set(ratings_df.loc[ratings_df.user == uid, "item"].tolist())
            unseen = all_course_ids - seen
            if not unseen:
                continue

            # Estimate rating for each unseen course
            ests = [algo.predict(uid=uid, iid=iid, verbose=False) for iid in unseen]
            # sort descending by estimated rating
            ests.sort(key=lambda pred: pred.est, reverse=True)

            selected = ests if top_n is None else ests[:top_n]
            for pred in selected:
                users.append(uid)
                courses.append(int(pred.iid))
                scores.append(float(pred.est))

        return pd.DataFrame({"USER": users, "COURSE_ID": courses, "SCORE": scores})

    # -------------------------------------------------------
    # Placeholder for the remaining models
    # -------------------------------------------------------
    raise NotImplementedError(
        f"Prediction routine for ‘{model_name}’ has not been implemented yet.")
