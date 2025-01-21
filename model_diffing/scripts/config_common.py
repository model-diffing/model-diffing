from pydantic import BaseModel


class LLMConfig(BaseModel):
    name: str
    revision: str | None


class LLMsConfig(BaseModel):
    llms: list[LLMConfig]
    dtype: str


class CommonCorpusTokenSequenceIteratorConfig(BaseModel):
    cache_dir: str
    sequence_length: int


class ConnorsTokenSequenceLoaderConfig(BaseModel):
    cache_dir: str
    sequence_length: int


class ActivationsIteratorConfig(BaseModel):
    layer_indices_to_harvest: list[int]
    harvest_batch_size: int
    sequence_iterator: CommonCorpusTokenSequenceIteratorConfig | ConnorsTokenSequenceLoaderConfig


class ShuffleConfig(BaseModel):
    shuffle_buffer_size: int


class DataConfig(BaseModel):
    activations_iterator: ActivationsIteratorConfig
    shuffle_config: ShuffleConfig
    activations_reshaper: str
    batch_size: int


class WandbConfig(BaseModel):
    name: str | None = None
    project: str
    entity: str


class BaseExperimentConfig(BaseModel):
    cache_dir: str
    dtype: str = "float32"  # put this somewhere else?
    data: DataConfig
    seed: int
    llms: LLMsConfig
    layer_indices_to_harvest: list[int]
    wandb: WandbConfig | None
