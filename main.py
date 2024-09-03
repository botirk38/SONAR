import logging
import os

from datasets import Dataset

from huggingface_pipelines.pipeline_builder import PipelineBuilder
from huggingface_pipelines.text import TextDatasetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configure and load the initial dataset
    dataset_config = TextDatasetConfig(
        dataset_name="ag_news",
        dataset_split="test",
        output_dir="results",
        trust_remote_code=True,
    )
    dataset = dataset_config.load_dataset()

    builder = PipelineBuilder()

    preprocessing_pipeline = builder.create_pipeline("ag_news", "text_segmentation")

    dataset = preprocessing_pipeline(dataset)

    text_to_embedding_pipeline = builder.create_pipeline("ag_news", "text_to_embedding")

    dataset = text_to_embedding_pipeline(dataset)

    # Build configuration for embedding to text pipeline
    embedding_to_text_pipeline = builder.create_pipeline("ag_news", "embedding_to_text")

    dataset = embedding_to_text_pipeline(dataset)

    # Initialize the metric pipeline config

    metrics_pipeline = builder.create_pipeline("ag_news", "analyze_metric")

    # Run metrics pipeline
    dataset = metrics_pipeline(dataset)

    # Save the dataset to disk

    cache_file_name = f"cache_{metrics_pipeline.__class__.__name__}.parquet"
    cache_file_path = os.path.join(
        f"{dataset_config.output_dir}_{dataset_config.dataset_name}", cache_file_name
    )

    if not dataset_config.streaming:
        dataset.save_to_disk(cache_file_path, num_proc=4)

    else:
        data_list = list(dataset)

        # Create a new Dataset from the collected data
        new_dataset = Dataset.from_list(data_list)
        new_dataset.save_to_disk(cache_file_path, num_proc=4)


if __name__ == "__main__":
    main()
