from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal


class ExperimentCfg(BaseModel):
    seed: int
    device: Literal["cpu", "cuda"] = "cpu"


class DataCfg(BaseModel):
    data_dir: str
    raw_dir: str
    processed_dir: str
    seq_len: int
    num_workers: int
    pin_memory: bool


class ModelCfg(BaseModel):
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    pre_ln: bool
    n_layers: int
    predict_variance: bool


class TrainingCfg(BaseModel):
    epochs: int
    batch_size: int
    use_amp: bool
    grad_clip: float

    @field_validator("epochs", "batch_size")
    def positive_int(cls, v):
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class OptimizerCfg(BaseModel):
    name: Literal["AdamW", "Adam", "SGD"]
    lr: float
    weight_decay: float

    @field_validator("lr")
    def lr_positive(cls, v):
        if v <= 0.0:
            raise ValueError("lr must be > 0")
        return v

    @field_validator("weight_decay")
    def wd_non_negative(cls, v):
        if v < 0.0:
            raise ValueError("weight_decay must be >= 0")
        return v


class SchedulerCfg(BaseModel):
    name: str
    warmup_frac: float
    min_lr: float

    @field_validator("warmup_frac")
    def warmup_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("warmup_frac must be between 0 and 1")
        return v

    @field_validator("min_lr")
    def min_lr_non_negative(cls, v):
        if v < 0.0:
            raise ValueError("min_lr must be >= 0")
        return v


class HeteroCfg(BaseModel):
    clamp_logvar: bool
    logvar_clamp_min: float
    logvar_clamp_max: float


class LossCfg(BaseModel):
    type: str
    hetero: Optional[HeteroCfg]


class MLflowInnerCfg(BaseModel):
    tracking_uri: str
    experiment_name: str
    run_name: str


class LoggingCfg(BaseModel):
    use_mlflow: bool
    mlflow: MLflowInnerCfg


class CheckpointCfg(BaseModel):
    checkpoint_dir: str
    resume_from: Optional[str] = None


class Config(BaseModel):
    experiment: ExperimentCfg
    data: DataCfg
    model: ModelCfg
    training: TrainingCfg
    optimizer: OptimizerCfg
    scheduler: SchedulerCfg
    loss: LossCfg
    logging: LoggingCfg
    checkpoint: CheckpointCfg
