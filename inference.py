
from src.feature_extractor import DINOFeatureExtractor
from src.predictor import BiomassPredictor


class BiomassInferencePipeline:

    def __init__(self, artifact_path, device=None):

        self.predictor = BiomassPredictor(
            artifact_path
        )

        artifact = self.predictor.artifact

        self.feature_extractor = DINOFeatureExtractor(
            backbone_name=artifact["backbone_name"],
            image_height=artifact["image_height"],
            image_width=artifact["image_width"],
            device=device
        )

    def predict(self, image_path):

        embedding = self.feature_extractor.extract(
            image_path
        )

        return self.predictor.predict_embedding(
            embedding
        )
