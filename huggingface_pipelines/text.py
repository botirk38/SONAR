import json
import logging
from datasets import load_dataset, Dataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline, EmbeddingToTextModelPipeline
from dataclasses import dataclass
from typing import List, Dict, Any
from .pipeline import Pipeline
from .pipeline_config import TextPipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextToTextHFPipeline(Pipeline):
    """
    A pipeline for encoding text datasets from HuggingFace into embeddings, decoding embeddings back into texts,
    and evaluating the quality using metrics.
    """
    config: TextPipelineConfig

    def __post_init__(self):
        """
        Initializes the dataset, models, and metric after the instance is created.
        """
        logger.info(
            f"Loading dataset {self.config.dataset_name} with split {self.config.dataset_split}...")
        self.dataset = load_dataset(
            self.config.dataset_name, split=self.config.dataset_split)
        logger.info("Dataset loaded. Initializing models...")
        self.t2vec_model = TextToEmbeddingModelPipeline(
            encoder=self.config.encoder_model, tokenizer=self.config.encoder_model, device=self.config.device)
        self.t2t_model = EmbeddingToTextModelPipeline(
            decoder=self.config.decoder_model, tokenizer=self.config.encoder_model, device=self.config.device)
        logger.info("Models initialized.")

    def encode_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Encodes a list of texts into embeddings.

        Args:
            texts (List[str]): A list of texts to be encoded.

        Returns:
            List[Dict[str, Any]]: A list of encoded embeddings.
        """
        try:
            logger.info(f"Encoding {len(texts)} texts...")
            embeddings = self.t2vec_model.predict(
                texts, source_lang=self.config.source_lang, batch_size=self.config.batch_size)
            logger.info("Texts encoded successfully.")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

    def decode_embeddings(self, embeddings: List[Any]) -> List[str]:
        """
        Decodes a list of embeddings back into texts.

        Args:
            embeddings (List[Any]): A list of embeddings to be decoded.

        Returns:
            List[str]: A list of decoded texts.
        """
        try:
            logger.info(f"Decoding {len(embeddings)} embeddings...")
            decoded_texts = self.t2t_model.predict(
                embeddings, target_lang=self.config.target_lang, batch_size=self.config.batch_size)
            logger.info("Texts decoded successfully.")
            return decoded_texts
        except Exception as e:
            logger.error(f"Error decoding texts: {e}")
            raise

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single batch of data, returning original, reconstructed texts and metric score.

        Args:
            batch (Dict[str, Any]): A batch of data containing texts.

        Returns:
            Dict[str, Any]: A dictionary containing original texts, reconstructed texts, and column name.
        """
        results = {}
        for column in self.config.columns:
            texts = batch[column]
            embeddings = self.encode_texts(texts)
            reconstructed_texts = self.decode_embeddings(embeddings)
            results[column] = {'original': texts,
                               'reconstructed': reconstructed_texts}
        return results

    def cache_results(self):
        """
        Caches the results to a JSON file.

        The results are saved in a file named 'output_file_name_shard_{shard_id}.json'.
        """
        try:
            file_name = f'{self.config.output_file_name}_shard_{self.config.shard_id}.json'
            logger.info(f"Caching results to {file_name}...")
            with open(file_name, 'w') as f:
                json.dump(self.results, f)
            logger.info("Results cached successfully.")
        except Exception as e:
            logger.error(f"Error caching results: {e}")

    def cache_results_arrow(self):
        """
        Caches the results to an Arrow file.

        The results are saved in a file named 'output_file_name_shard_{self.config.shard_id}.arrow'.
        """
        try:
            file_name = f'{self.config.output_file_name}_shard_{self.config.shard_id}.arrow'
            logger.info(f"Caching results to {file_name}...")
            dataset = Dataset.from_dict({
                "original": [result['original'] for result in self.results],
                "reconstructed": [result['reconstructed'] for result in self.results]
            })
            dataset.save_to_disk(file_name)
            logger.info("Results cached successfully.")
        except Exception as e:
            logger.error(f"Error caching results: {e}")

