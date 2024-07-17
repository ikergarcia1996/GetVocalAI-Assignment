import argparse
import os

import torch
from accelerate import Accelerator
from dataset import get_dataloader
from load_model import load_model
from scorer import ConversationAccuracyScorer
from tqdm.auto import tqdm


@torch.no_grad()
def main(
    model_name_or_path, dataset_path, batch_size: int = 4, quantization: int = None
):
    model, tokenizer = load_model(
        inference=True,
        model_weights_name_or_path=model_name_or_path,
        quantization=quantization,
    )
    test_dataloader = get_dataloader(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    accelerator = Accelerator()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    computed_losses = []
    model.eval()
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        labels = batch.pop("labels")
        loss_weight_mask = batch.pop("loss_weight_mask")
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        batch_size, sequence_length, vocab_size = logits.size()

        # Flatten the logits and labels for cross_entropy computation
        logits = logits.view(batch_size * sequence_length, vocab_size)
        labels = labels.view(batch_size * sequence_length)
        loss_weight_mask = loss_weight_mask.view(batch_size * sequence_length)

        # Compute the loss for each token and reshape back to (batch_size, sequence_length)
        losses = torch.nn.functional.cross_entropy(
            logits, labels, reduction="none"
        ).view(batch_size, sequence_length)

        # Apply the loss weight mask
        losses = losses * loss_weight_mask.view(batch_size, sequence_length)

        # Average the loss over the sequence length for each example, considering the mask
        example_losses = losses.sum(dim=1) / loss_weight_mask.view(
            batch_size, sequence_length
        ).sum(dim=1)

        # Convert the tensor to a list of loss values for each example
        example_losses = example_losses.tolist()
        computed_losses.extend(example_losses)

    conversation_scorer = ConversationAccuracyScorer(
        gold_data_path=test_dataloader.dataset.dataset_path,
        predicted_labels=computed_losses,
    )
    acc = conversation_scorer.compute()
    print(f"Accuracy: {acc}")
    conversation_scorer.print_predictions(
        os.path.join(model_name_or_path.replace("/", "_"), "predictions.json")
    )

    with open(
        os.path.join(model_name_or_path.replace("/", "_"), "accuracy.txt"), "w"
    ) as f:
        print(f"Accuracy: {acc}", file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The model name or path to the model weights",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="The path to the dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The batch size for evaluation",
    )
    parser.add_argument(
        "--quantization",
        type=int,
        default=None,
        help="The quantization level to use",
    )

    args = parser.parse_args()

    main(
        model_name_or_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        quantization=args.quantization,
    )
