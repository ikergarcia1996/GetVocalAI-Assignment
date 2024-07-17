from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from typing import List, Dict, Union
import logging
import json
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding
import os
import multiprocessing
from functools import partial


def prepare_input(
    example: Dict[str, Union[List[Dict[str, str]], List[Dict[str, str]]]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> BatchEncoding:
    """
    Prepare input for encoder-only model
    [CLS] Conversation history [SEP] Candidate response [SEP]
    Args:
        example: Dict[str, Union[List[Dict[str, str]], List[Dict[str, str]]]
        tokenizer: PreTrainedTokenizer
        max_length: int
    Returns:
        model_inputs: BatchEncoding
    """
    conversation = example["conversation"]
    candidates = example["candidates"]
    label = example["label"]

    # Build conversation
    text_conversation = []
    for turn in conversation:
        role = turn["role"]
        text = turn["content"]
        text_conversation.append(f"{role}: {text}")
    text_conversation = " ".join(text_conversation)

    # Build input examples
    model_inputs = []
    for candidate_no, candidate in enumerate(candidates):
        candidate_text = candidate[0]["content"]
        inputs = tokenizer(
            text_conversation,
            candidate_text,
            add_special_tokens=True,
            padding="do_not_pad",
            truncation=False,
        )
        if len(inputs["input_ids"]) > max_length:
            logging.warning(
                f"Input length {len(inputs['input_ids'])} exceeds max_length {max_length}. We will truncate this example."
            )
            # Truncate from the beginning, but keep fist token [CLS]
            first_token = inputs["input_ids"][0]
            inputs["input_ids"] = inputs["input_ids"][-max_length + 1 :]
            inputs["input_ids"] = [first_token] + inputs["input_ids"]
            first_token = inputs["attention_mask"][0]
            inputs["attention_mask"] = inputs["attention_mask"][-max_length + 1 :]
            inputs["attention_mask"] = [first_token] + inputs["attention_mask"]

        inputs["label"] = 1 if candidate_no == label else 0

        model_inputs.append(inputs)

    return model_inputs


class ConversationDataset(Dataset):
    def __init__(
        self, dataset_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 512
    ):
        """
        Initialize ConversationDataset
        Args:
            dataset_path: str
            tokenizer: PreTrainedTokenizer
            max_length: int

        Returns:
            None
        """

        self.dataset_path = dataset_path

        examples = []
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf8") as f:
            for line in f:
                example = json.loads(line)
                examples.append(example)

        print(f"Loaded {len(examples)} lines from {dataset_path}")

        self.dataset = []

        for example in tqdm(examples, desc="Preparing input"):
            self.dataset.extend(
                prepare_input(
                    example=example, tokenizer=tokenizer, max_length=max_length
                )
            )

        print(f"Prepared {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_dataloader(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
):
    """
    Get DataLoader for ConversationDataset
    Args:
        dataset_path: str
        tokenizer: PreTrainedTokenizer
        batch_size: int
        max_length: int
    Returns:
        DataLoader
    """

    dataset = ConversationDataset(
        dataset_path=dataset_path, tokenizer=tokenizer, max_length=max_length
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,  # Better performance for mixed precision training
        padding=True,
        return_tensors="pt",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator,
    )

    return dataloader
