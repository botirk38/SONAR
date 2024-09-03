import logging
from pathlib import Path
from typing import Any, Dict, Literal, Union

import yaml

from .audio import AudioToEmbeddingPipelineFactory
from .metric_analyzer import MetricAnalyzerPipelineFactory
from .pipeline import Pipeline
from .text import (
    EmbeddingToTextPipelineFactory,
    TextSegmentationPipelineFactory,
    TextToEmbeddingPipelineFactory,
)

logger = logging.getLogger(__name__)

# Define a custom type for supported operations
SupportedOperation = Literal[
    "text_to_embedding", "embedding_to_text", "text_segmentation", "analyze_metric"
]


class PipelineBuilder:
    def __init__(
        self, config_dir: Union[str, Path] = "huggingface_pipelines/datacards"
    ):
        self.config_dir = Path(config_dir)
        self.pipeline_factories: Dict[SupportedOperation, Any] = {
            "text_to_embedding": TextToEmbeddingPipelineFactory(),
            "embedding_to_text": EmbeddingToTextPipelineFactory(),
            "text_segmentation": TextSegmentationPipelineFactory(),
            "analyze_metric": MetricAnalyzerPipelineFactory(),
            "audio_to_embedding": AudioToEmbeddingPipelineFactory(),
        }

    def load_config(
        self, dataset_name: str, operation: SupportedOperation
    ) -> Dict[str, Any]:
        config_file = self.config_dir / f"{dataset_name}/{operation}.yaml"
        try:
            with open(config_file, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config File not found: {config_file}")
            raise FileNotFoundError(
                f"No configuration file found for dataset: {dataset_name} and operation: {operation}"
            )

    def create_pipeline(
        self, dataset_name: str, operation: SupportedOperation
    ) -> Pipeline:
        if operation not in self.pipeline_factories:
            raise ValueError(
                f"Unsupported operation: {operation}. Supported operations are: {', '.join(self.pipeline_factories.keys())}"
            )

        config = self.load_config(dataset_name, operation)
        return self.pipeline_factories[operation].create_pipeline(config)
