import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from copy import deepcopy
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.utils import PaddingStrategy


def altenate_roles(conversation: List[Dict[str, str]]) -> List[str]:
    current_role = "user"
    new_conversation = deepcopy(conversation)
    for i, turn in enumerate(new_conversation):
        new_conversation[i]["role"] = current_role
        if current_role == "user":
            current_role = "assistant"
        else:
            current_role = "user"
    return new_conversation


def prepare_input(
    example: Dict[str, Union[List[Dict[str, str]], List[Dict[str, str]]]],
    tokenizer: PreTrainedTokenizer,
) -> BatchEncoding:
    """
    Prepare input for instruction models
    Args:
        example: Dict[str, Union[List[Dict[str, str]], List[Dict[str, str]]]
        tokenizer: PreTrainedTokenizer
        max_length: int
    Returns:
        model_inputs: BatchEncoding
    """
    conversation = example["conversation"]
    candidates = example["candidates"]
    if len(conversation) % 2 == 0:
        conversation = conversation[
            1:
        ]  # Ensure that assistant is always the last one to speak

    examples = []
    for candidate in candidates:
        # print(altenate_roles(conversation + candidate))
        labels = tokenizer.apply_chat_template(
            altenate_roles(conversation + candidate),
            tokenize=False,
        )

        prompt = tokenizer.apply_chat_template(
            altenate_roles(conversation),
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer(
            text=labels,
            max_length=2048,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        prompt = tokenizer(
            text=prompt,
            max_length=2048,
            truncation=True,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]

        loss_weight_mask = np.ones(len(model_inputs["input_ids"]), dtype=np.float32)
        for i in range(len(prompt)):
            loss_weight_mask[i] = 0.0

        model_inputs["loss_weight_mask"] = loss_weight_mask
        model_inputs["labels"] = model_inputs["input_ids"].copy()

        examples.append(model_inputs)

    return examples


class ConversationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizer,
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
                    example=example,
                    tokenizer=tokenizer,
                )
            )

        print(f"Prepared {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        inputs_ids = (
            [feature["input_ids"] for feature in features]
            if "input_ids" in features[0].keys()
            else None
        )
        max_input_len = max(len(l) for l in inputs_ids)

        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        orig_labels = (
            [feature["labels"].copy() for feature in features].copy()
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)

        words_ids = (
            [feature["words_ids"] for feature in features]
            if "words_ids" in features[0].keys()
            else None
        )
        if words_ids is not None:
            max_words_ids_length = max(len(l) for l in words_ids)
            for feature in features:
                remainder = [-1] * (max_words_ids_length - len(feature["words_ids"]))
                feature["words_ids"] = feature["words_ids"] + remainder

        original_sentence_ids = (
            [feature["original_sentence_ids"] for feature in features]
            if "original_sentence_ids" in features[0].keys()
            else None
        )
        if original_sentence_ids is not None:
            max_original_sentence_ids_length = max(
                len(l) for l in original_sentence_ids
            )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_original_sentence_ids_length
                    - len(feature["original_sentence_ids"])
                )
                feature["original_sentence_ids"] = (
                    feature["original_sentence_ids"] + remainder
                )

        labeled_sentence_ids = (
            [feature["labeled_sentence_ids"] for feature in features]
            if "labeled_sentence_ids" in features[0].keys()
            else None
        )
        if labeled_sentence_ids is not None:
            max_labeled_sentence_ids_length = max(len(l) for l in labeled_sentence_ids)
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_labeled_sentence_ids_length
                    - len(feature["labeled_sentence_ids"])
                )
                feature["labeled_sentence_ids"] = (
                    feature["labeled_sentence_ids"] + remainder
                )

        loss_weight_mask = (
            [feature["loss_weight_mask"] for feature in features]
            if "loss_weight_mask" in features[0].keys()
            else None
        )

        if loss_weight_mask is not None:
            max_loss_weight_mask_length = max(len(l) for l in loss_weight_mask)
            if self.pad_to_multiple_of is not None:
                max_loss_weight_mask_length = (
                    (max_loss_weight_mask_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [0.0 if self.label_pad_token_id == -100 else 1.0] * (
                    max_loss_weight_mask_length - len(feature["loss_weight_mask"])
                )
                if isinstance(feature["loss_weight_mask"], list):
                    feature["loss_weight_mask"] = (
                        feature["loss_weight_mask"] + remainder
                        if padding_side == "right"
                        else remainder + feature["loss_weight_mask"]
                    )
                elif padding_side == "right":
                    feature["loss_weight_mask"] = np.concatenate(
                        [feature["loss_weight_mask"], remainder]
                    ).astype(np.float32)
                else:
                    feature["loss_weight_mask"] = np.concatenate(
                        [remainder, feature["loss_weight_mask"]]
                    ).astype(np.float32)

        # print(self.tokenizer.padding_side)
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        """
        if features["input_ids"].size() != features["labels"].size():
            raise ValueError(
                f"Input and label sizes do not match\n"
                f"Input size: {features['input_ids'].size()}\n"
                f"Label size: {features['labels'].size()}\n"
                f"max_input_len: {max_input_len}\n"
                f"max_label_length: {max_label_length}\n"
                f""
                f"Input: {features['input_ids']}\n"
                f"Label: {features['labels']}\n"
                f"Input: {self.tokenizer.batch_decode(inputs_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)}\n"
                f"Label: {self.tokenizer.batch_decode(orig_labels,skip_special_tokens=False,clean_up_tokenization_spaces=False)}\n"
            )
        """
        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features


def get_dataloader(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
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

    dataset = ConversationDataset(dataset_path=dataset_path, tokenizer=tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,  # = 8 May be faster on some hardware
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    return dataloader
