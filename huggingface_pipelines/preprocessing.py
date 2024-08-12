from transformers import WhisperProcessor, WhisperFeatureExtractor
import numpy as np
import spacy
from typing import List, Dict, Any, Optional
from .pipeline import Pipeline, PipelineConfig, PipelineFactory
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPACY_MODELS = {
    "eng_Latn": "en_core_web_sm",
    "fra_Latn": "fr_core_news_sm",
    "deu_Latn": "de_core_news_sm",
    "spa_Latn": "es_core_news_sm",
    "ita_Latn": "it_core_news_sm",
    "por_Latn": "pt_core_news_sm",
    "nld_Latn": "nl_core_news_sm",
}


@dataclass
class TextPreprocessingPipelineConfig(PipelineConfig):
    """
    Configuration class for text preprocessing pipelines.
    """
    handle_missing: str = 'skip'
    fill_value: Optional[str] = None
    source_lang: str = "eng_Latn"


class TextPreprocessingPipeline(Pipeline):
    """
    A pipeline for preprocessing text data.
    """

    def __init__(self, config: TextPreprocessingPipelineConfig):
        super().__init__(config)
        self.config = config
        self.nlp = self.load_spacy_model(self.config.source_lang)
        logger.info("Text preprocessing model initialized.")

    def load_spacy_model(self, lang_code: str):
        if lang_code not in SPACY_MODELS:
            raise ValueError(
                f"Unsupported language code: {lang_code}. Please add it to the SPACY_MODELS dictionary.")
        model_name = SPACY_MODELS[lang_code]
        try:
            return spacy.load(model_name)
        except OSError:
            logger.warning(
                f"SpaCy model {model_name} not found. Attempting to download...")
            spacy.cli.download(model_name)
            return spacy.load(model_name)

    def preprocess_text(self, text: Optional[str]) -> List[str]:
        if text is None or (isinstance(text, str) and text.strip() == ''):
            if self.config.handle_missing == 'skip':
                return []
            elif self.config.handle_missing == 'remove':
                return []
            elif self.config.handle_missing == 'fill':
                return [self.config.fill_value] if self.config.fill_value else []
            else:
                raise ValueError(
                    f"Invalid handle_missing option: {self.config.handle_missing}")
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for column in self.config.columns:
            if column in batch:
                batch[f"{column}_preprocessed"] = [
                    self.preprocess_text(text) for text in batch[column]]
        return batch


class TextPreprocessingPipelineFactory(PipelineFactory):
    def create_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        pipeline_config = TextPreprocessingPipelineConfig(**config)
        return TextPreprocessingPipeline(pipeline_config)


@dataclass
class AudioPreprocessingPipelineConfig(PipelineConfig):
    """
    Configuration class for audio preprocessing pipelines.
    """
    whisper_model: str = "openai/whisper-base"
    max_duration: Optional[float] = None


class AudioPreprocessingPipeline(Pipeline):
    """
    A pipeline for preprocessing audio data using Whisper's feature extractor.
    """

    def __init__(self, config: AudioPreprocessingPipelineConfig):
        super().__init__(config)
        self.config = config
        self.whisper_processor = WhisperProcessor.from_pretrained(
            self.config.whisper_model)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config.whisper_model)
        logger.info("Whisper processor and feature extractor initialized.")

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if self.config.max_duration is not None:
            max_samples = int(self.config.max_duration * sample_rate)
            audio = audio[:max_samples]
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="np"
        )
        return inputs.input_features.squeeze()

    def process_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for column in self.config.columns:
            if column not in batch:
                logger.warning(f"Column {column} not found in batch.")
                continue
            audio_data = batch[column]
            processed_audio = []
            for audio in audio_data:
                if isinstance(audio, dict) and 'array' in audio and 'sampling_rate' in audio:
                    audio_array, sample_rate = audio['array'], audio['sampling_rate']
                elif isinstance(audio, np.ndarray):
                    audio_array, sample_rate = audio, self.feature_extractor.sampling_rate
                else:
                    raise ValueError(
                        f"Unsupported audio data format: {type(audio)}")
                processed = self.preprocess_audio(audio_array, sample_rate)
                processed_audio.append(processed)
            batch[f"{column}_preprocessed"] = processed_audio
        return batch


class AudioPreprocessingPipelineFactory(PipelineFactory):
    def create_pipeline(self, config: Dict[str, Any]) -> Pipeline:
        pipeline_config = AudioPreprocessingPipelineConfig(**config)
        return AudioPreprocessingPipeline(pipeline_config)

