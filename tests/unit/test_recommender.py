import numpy as np
import pandas as pd
import pytest

from src.core.models.hybrid import HybridRecommender


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(0)
    n_users, n_items = 20, 15
    items_df = pd.DataFrame(
        {
            "item_id": [f"i{i}" for i in range(n_items)],
            "title": [f"Product {i} category {i % 3}" for i in range(n_items)],
            "description": [f"Description text for product {i}" for i in range(n_items)],
        }
    )
    rows = []
    for u in range(n_users):
        preferred_cat = u % 3
        for _ in range(15):
            it = rng.integers(0, n_items)
            cat = it % 3
            rating = 1.0 if cat == preferred_cat else 0.1
            rows.append({"user_id": f"u{u}", "item_id": f"i{it}", "rating": rating})
    return pd.DataFrame(rows), items_df


def test_fit_actually_trains(small_dataset):
    interactions, items = small_dataset
    model = HybridRecommender({"epochs": 10, "embedding_dim": 8})
    history = model.fit(interactions, items, text_cols=["title", "description"])
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert model.is_fitted


def test_recommend_requires_fit(small_dataset):
    model = HybridRecommender({})
    with pytest.raises(RuntimeError):
        model.recommend("u0", n=5)


def test_recommend_unknown_user_returns_empty(small_dataset):
    interactions, items = small_dataset
    model = HybridRecommender({"epochs": 3})
    model.fit(interactions, items, text_cols=["title", "description"])
    assert model.recommend("nonexistent-user", n=5) == []


def test_cold_start_content_score_is_neutral_not_random(small_dataset):
    interactions, items = small_dataset
    model = HybridRecommender({"epochs": 3})
    model.fit(interactions, items, text_cols=["title", "description"])
    recs = model.recommend("u0", n=5, user_items=[])
    assert all(r["content_score"] == 0.5 for r in recs)


def test_recommend_returns_requested_count(small_dataset):
    interactions, items = small_dataset
    model = HybridRecommender({"epochs": 3})
    model.fit(interactions, items, text_cols=["title", "description"])
    recs = model.recommend("u0", n=5)
    assert len(recs) == 5
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_save_and_load_roundtrip(tmp_path, small_dataset):
    interactions, items = small_dataset
    model = HybridRecommender({"epochs": 3})
    model.fit(interactions, items, text_cols=["title", "description"])
    recs_before = model.recommend("u0", n=3)

    save_path = str(tmp_path / "model")
    model.save(save_path)

    loaded = HybridRecommender({})
    loaded.load(save_path)
    recs_after = loaded.recommend("u0", n=3)

    assert [r["item_id"] for r in recs_before] == [r["item_id"] for r in recs_after]