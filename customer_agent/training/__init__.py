# -*- coding: utf-8 -*-
"""
训练模块

提供模型训练和打包功能。
"""

from customer_agent.training.trainer import Trainer, TrainerConfig
from customer_agent.training.model_storage import (
    ModelMetadata,
    create_model_package,
    extract_model_archive,
    get_latest_model,
    get_model_path,
    load_metadata_from_archive,
)
from customer_agent.training.finetune import (
    FinetuneDataGenerator,
    FinetuneConfig,
    FinetuneExample,
    Paraphraser,
    ParaphraserConfig,
)

__all__ = [
    "Trainer",
    "TrainerConfig",
    # Model Storage
    "ModelMetadata",
    "create_model_package",
    "extract_model_archive",
    "get_latest_model",
    "get_model_path",
    "load_metadata_from_archive",
    # Finetune
    "FinetuneDataGenerator",
    "FinetuneConfig",
    "FinetuneExample",
    "Paraphraser",
    "ParaphraserConfig",
]
