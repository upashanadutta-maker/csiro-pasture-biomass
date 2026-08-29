
from src.predictor import BiomassPredictor

artifact_path = (
    "artifacts/"
    "biomass_dinov3_large_ridge.joblib"
)

predictor = BiomassPredictor(
    artifact_path
)

print("Fresh-process import successful.")
print("Embedding dimension:", predictor.embedding_dim)
print("Targets:", predictor.target_names)
