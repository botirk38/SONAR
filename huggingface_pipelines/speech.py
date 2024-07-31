import torch
from typing import Dict, Any
from dataclasses import dataclass
import logging
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from .pipeline import PipelineOverwrites, PipelineConfig, Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechToEmbeddingOverwrites(PipelineOverwrites, total=False):
    encoder_model: str
    fbank_dtype: str
    n_parallel: int
    pad_idx: int


@dataclass
class SpeechToEmbeddingPipelineConfig(PipelineConfig):
    encoder_model: str = "sonar_speech_encoder_eng"
    fbank_dtype: torch.dtype = torch.float32
    n_parallel: int = 4
    pad_idx: int = 0


@dataclass
class SpeechToEmbeddingPipeline(Pipeline):

    config: SpeechToEmbeddingPipelineConfig

    def __post_init__(self):
        self.model = SpeechToEmbeddingModelPipeline(
            encoder=self.config.encoder_model,
            device=torch.device(self.config.device),
            fbank_dtype=self.config.fbank_dtype
        )

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        audio_inputs = []
        valid_columns = []

        for column in self.config.columns:
            if column in batch:

                audio_inputs.extend(batch[column])
                valid_columns.append(column)
            else:
                logger.warning(f"Column {column} not found in batch.")

        if not audio_inputs:
            logger.warning("No valid audio inputs found in batch.")
            return batch

        try:
            all_embeddings = self.model.predict(
                input=audio_inputs,
                batch_size=self.config.batch_size,
                n_parallel=self.config.n_parallel,
                pad_idx=self.config.pad_idx
            )

            start_idx = 0
            for column in valid_columns:
                end_idx = start_idx + len(batch[column])
                batch[f"{column}_{self.config.output_column_suffix}"] = all_embeddings[start_idx:end_idx]
                start_idx = end_idx

        except Exception as e:
            logger.error(f"Error processing batch: {e}")

        return batch
