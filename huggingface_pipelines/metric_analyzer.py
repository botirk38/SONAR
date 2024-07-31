import logging
from typing import List, Dict, Any
from dataclasses import dataclass, replace
from datasets import load_metric as eval_load
from .pipeline import Pipeline, PipelineConfig, PipelineOverwrites

logger = logging.getLogger(__name__)


class MetricOverwrites(PipelineOverwrites, total=False):
    metric_name: str
    low_score_threshold: float


@dataclass
class MetricPipelineConfig(PipelineConfig):
    """
    Configuration class for metrics.

    Attributes:
        metric_name (str): The name of the metric to be used for evaluation.
        low_score_threshold (float): The threshold below which the score is considered low.
    """
    reconstructed_column: str = None
    metric_name: str = "bleu"
    low_score_threshold: float = 0.5

    def with_overwrites(self, overwrites: MetricOverwrites):
        return replace(self, **overwrites)


@dataclass
class MetricAnalyzerPipeline(Pipeline):
    """
    A pipeline to analyze metrics for different data types.
    """
    config: MetricPipelineConfig

    def __post_init__(self):
        logger.info(f"Loading metric: {self.config.metric_name}...")
        self.metric = eval_load(self.config.metric_name)
        logger.info(f"Metric {self.config.metric_name} loaded successfully.")

    def compute_metric(self, original_data: List[Any], reconstructed_data: List[Any]) -> Dict[str, Any]:
        """
        Computes the metric score between original and reconstructed data.

        Args:
            original_data (List[Any]): A list of original data.
            reconstructed_data (List[Any]): A list of reconstructed data.

        Returns:
            Dict[str, Any]: A dictionary containing the metric score.
        """
        logger.info(f"Computing {self.config.metric_name} score...")
        references = [text for text in original_data]
        predictions = reconstructed_data

        # Compute the metric
        metric_score = self.metric.compute(
            predictions=predictions, references=references)
        logger.info(
            f"{self.config.metric_name} score computed: {metric_score}")
        return metric_score

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single batch of data by computing the metric and updating the current batch.

        Args:
            batch (Dict[str, Any]): A batch of data.

        Returns:
            batch: The updated batch with the 'metric_score' column.

        """

        for column in self.config.columns:
            original_data = batch[column]
            reconstructed_data = batch[self.config.reconstructed_column]
            metric_score = self.compute_metric(
                original_data, reconstructed_data)
            batch[f"column_{self.config.output_column_suffix}"] = [metric_score] * \
                len(original_data)
        return batch
