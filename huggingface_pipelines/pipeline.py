from abc import ABC, abstractmethod
from typing import List, TypedDict, Dict, Any
import logging
from dataclasses import dataclass, replace
from datasets import Dataset
import os
from contextlib import contextmanager
import torch
from .dataset import DatasetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineOverwrites(TypedDict):
    batch_size: int
    device: str
    cache_to_arrow: bool
    take: int
    output_dir: str
    output_file_name: str
    columns: List[str]


@dataclass
class PipelineConfig(ABC):
    """
    Abstract base class for pipeline configurations.

    Attributes:
        dataset_name (str): The name of the HuggingFace dataset to be used.
        dataset_split (str): The dataset split to be used (e.g., 'train', 'test', 'validation').
        batch_size (int): The batch size to be used for processing.
        device (str): The device to use for inference (e.g., 'cpu', 'cuda').
        cache_to_arrow (bool): Whether to cache results to Arrow format. Defaults to False.
        output_dir (str): The directory to save the output to. Defaults to 'results'.
        take (int): The number of batches to take for processing. Defaults to -1 (process all).
        dataset_uuid (str): The id for the dataset instance, this is used for caching. Defaults to None.

    """
    columns: List[str]
    dataset_config: DatasetConfig
    batch_size: int = 5
    device: str = "cpu"
    take: int = -1
    cache_to_arrow: bool = False

    def with_overwrites(self, overwrites: PipelineOverwrites):
        return replace(self, **overwrites)


class Pipeline(ABC):
    """
    Abstract base class for different pipelines.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = []

    @contextmanager
    def resource_manager(self):
        try:
            if torch.cuda.is_available() and self.config.device == 'cuda':
                torch.cuda.empty_cache()

            yield

        finally:
            if torch.cuda.is_available() and self.config.device == 'cuda':
                torch.cuda.empty_cache()

    @abstractmethod
    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single batch of data and returns the updated batch.

        Args:
            batch (Dict[str, Any]): A batch of data.

        Returns:
            Dict[str, Any]: The updated batch.
        """
        pass

    def __call__(self, dataset: Dataset) -> Dataset:
        """
        Processes the dataset and updates it.

        Args:
            dataset (Dataset): The dataset to process.

        Returns:
            Dataset: The updated dataset.
        """
        try:
            logger.info("Starting to process dataset...")
            os.makedirs(
                f"{self.config.dataset_config.output_dir}_{self.config.dataset_config.dataset_name}_{self.config.dataset_config.uuid}", exist_ok=True)

            if self.config.take > 0:
                dataset = dataset.select(
                    range(self.config.take * self.config.batch_size))

            cache_file_name = f"cache_{self.__class__.__name__}.arrow"
            cache_file_path = os.path.join(
                f"{self.config.dataset_config.output_dir}_{self.config.dataset_config.dataset_name}_{self.config.dataset_config.uuid}", cache_file_name)

            num_proc = 1

            updated_dataset = dataset.map(
                lambda batch: self.process_batch(batch),
                batched=True,
                batch_size=self.config.batch_size,
                load_from_cache_file=self.config.cache_to_arrow,
                cache_file_name=cache_file_path if self.config.cache_to_arrow else None,
                desc="Processing dataset",
                num_proc=num_proc


            )

            return updated_dataset
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise
