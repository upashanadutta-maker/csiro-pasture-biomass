
import joblib


class BiomassPredictor:
    """
    Load the trained regression artifact and convert
    one DINO embedding into biomass predictions.
    """

    def __init__(self, artifact_path):

        self.artifact = joblib.load(
            artifact_path
        )

        self.regressor = self.artifact[
            "regressor"
        ]

        self.target_names = self.artifact[
            "target_names"
        ]

        self.embedding_dim = self.artifact[
            "embedding_dim"
        ]

    def predict_embedding(self, embedding):
        """
        Predict biomass from one DINO embedding.

        Expected shape:
        (1, embedding_dim)
        """

        if embedding.ndim != 2:
            raise ValueError(
                "Embedding must be a 2D array."
            )

        if embedding.shape[0] != 1:
            raise ValueError(
                "Exactly one embedding is expected."
            )

        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"got {embedding.shape[1]}."
            )

        predictions = self.regressor.predict(
            embedding
        )[0]

        return {
            target: float(value)
            for target, value in zip(
                self.target_names,
                predictions
            )
        }
