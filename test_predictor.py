
import numpy as np
import pytest

from src.predictor import BiomassPredictor


ARTIFACT_PATH = (
    "artifacts/"
    "biomass_dinov3_large_ridge.joblib"
)


@pytest.fixture
def predictor():
    return BiomassPredictor(
        ARTIFACT_PATH
    )


def test_prediction_output(predictor):

    embedding = np.zeros(
        (1, predictor.embedding_dim),
        dtype=np.float32
    )

    prediction = predictor.predict_embedding(
        embedding
    )

    assert set(prediction.keys()) == set(
        predictor.target_names
    )

    assert len(prediction) == 5

    assert all(
        np.isfinite(value)
        for value in prediction.values()
    )


def test_rejects_wrong_embedding_dimension(predictor):

    bad_embedding = np.zeros(
        (1, 10),
        dtype=np.float32
    )

    with pytest.raises(ValueError):
        predictor.predict_embedding(
            bad_embedding
        )


def test_rejects_non_2d_embedding(predictor):

    bad_embedding = np.zeros(
        predictor.embedding_dim,
        dtype=np.float32
    )

    with pytest.raises(ValueError):
        predictor.predict_embedding(
            bad_embedding
        )


def test_rejects_multiple_embeddings(predictor):

    bad_embedding = np.zeros(
        (2, predictor.embedding_dim),
        dtype=np.float32
    )

    with pytest.raises(ValueError):
        predictor.predict_embedding(
            bad_embedding
        )
