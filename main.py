from huggingface_pipelines.metric_analyzer import MetricAnalyzerPipeline, MetricOverwrites, MetricPipelineConfig
import logging
import os
from huggingface_pipelines.preprocessing import TextPreprocessingPipelineConfig, TextPreprocessingPipeline
from datasets import Dataset
from huggingface_pipelines.dataset import DatasetConfig
from huggingface_pipelines.text import (
    HFEmbeddingToTextPipeline, HFTextToEmbeddingPipeline,
    TextToEmbeddingPipelineConfig, EmbeddingToTextPipelineConfig,
    EmbeddingToTextOverwrites

)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configure and load the initial dataset
    dataset_config = DatasetConfig(
        dataset_name="ag_news",
        dataset_split="test",
        output_dir="results",
        streaming=True

    )
    dataset = dataset_config.load_dataset()

    preprocessing_config = TextPreprocessingPipelineConfig(
        columns=["text"],
        dataset_config=dataset_config,
        handle_missing='skip',
        source_lang="eng_Latn",
        batch_size=5,
        take=1,
        output_column_suffix="preprocessed"
    )

    preprocessing_pipeline = TextPreprocessingPipeline(preprocessing_config)
    dataset = preprocessing_pipeline(dataset)

    # Build configuration for text to embedding pipeline
    text_to_embedding_config = TextToEmbeddingPipelineConfig(
        columns=["text_preprocessed"],
        batch_size=5,
        device="cpu",
        take=1,
        encoder_model="text_sonar_basic_encoder",
        source_lang="eng_Latn",
        dataset_config=dataset_config,
        output_column_suffix="embeddings"
    )

    # Initialize and run the text to embedding pipeline
    text_to_embedding_pipeline = HFTextToEmbeddingPipeline(
        text_to_embedding_config)
    dataset = text_to_embedding_pipeline(dataset)

    # Build configuration for embedding to text pipeline
    embedding_to_text_config = EmbeddingToTextPipelineConfig(
        columns=["text_preprocessed_embeddings"],
        batch_size=5,
        device="cpu",
        take=1,
        decoder_model="text_sonar_basic_decoder",
        target_lang="eng_Latn",
        dataset_config=dataset_config,
        output_column_suffix="reconstructed"
    )
    # Initialize and run the embedding to text pipeline

    embedding_to_text_config = embedding_to_text_config
    embedding_to_text_pipeline = HFEmbeddingToTextPipeline(
        embedding_to_text_config)
    dataset = embedding_to_text_pipeline(dataset)

    # Initialize the metric pipeline config
    metric_config = MetricPipelineConfig(
        columns=["text"],
        reconstructed_column=["text_preprocessed_embeddings_reconstructed"],
        batch_size=5,
        device="cpu",
        take=1,
        metric_name="bleu",
        low_score_threshold=0.5,
        dataset_config=dataset_config,
        output_column_suffix="rating"
    )
    metric_overwrites = MetricOverwrites(
        dataset_config=dataset_config

    )
    metric_config = metric_config.with_overwrites(metric_overwrites)

    metrics_pipeline = MetricAnalyzerPipeline(metric_config)

    # Run metrics pipeline
    dataset = metrics_pipeline(dataset)

    # Save the dataset to disk

    cache_file_name = f"cache_{metrics_pipeline.__class__.__name__}.parquet"
    cache_file_path = os.path.join(
        f"{dataset_config.output_dir}_{dataset_config.dataset_name}_{dataset_config.uuid}", cache_file_name)

    if not dataset_config.streaming:
        dataset.save_to_disk(cache_file_path, num_proc=4)

    else:
        data_list = list(dataset)

        # Create a new Dataset from the collected data
        new_dataset = Dataset.from_list(data_list)
        new_dataset.save_to_disk(cache_file_path, num_proc=4)


if __name__ == "__main__":
    main()
