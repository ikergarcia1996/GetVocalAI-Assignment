from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The local path or huggingface hub name of the model and tokenizer to use."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the training dataset or the split to load from the Hub"
        },
    )
    validation_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the validation dataset or the split to load from the Hub"
        },
    )

    test_datasets: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "The path to the test dataset or the split to load from the Hub"
        },
    )
