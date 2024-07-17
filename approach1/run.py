import math
import os
import sys

import torch
import wandb
from accelerate import Accelerator
from config import DataTrainingArguments, ModelArguments
from dataset import get_dataloader
from scorer import ConversationAccuracyScorer, SentenceScorer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    get_scheduler,
)


def get_dtype(accelerator: Accelerator):
    if accelerator.state.mixed_precision == "bf16":
        dtype = "bfloat16"
    elif accelerator.state.mixed_precision == "fp16":
        dtype = "float16"
    else:
        dtype = None
    return dtype


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"\n---> Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}\n"
    )

    return trainable_params, all_param, 100 * trainable_params / all_param


@torch.no_grad()
def evaluate(
    dataloader: DataLoader,
    accelerator: Accelerator,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    epoch: int = -1,
    train_step: int = -1,
):
    """
    Evaluate model on the given dataset

    Args:
        dataloader (DataLoader): The DataLoader to use
        accelerator (Accelerator): The Accelerator object to use
        model (PreTrainedModel): The model to evaluate
        tokenizer (PreTrainedTokenizerBase): The tokenizer of the model
        output_dir (str): The output directory to save the evaluation results
        stage (str): The stage of evaluation
        epoch (int): The epoch number
        train_step (int): The training step number

    Return:
        None
    """

    if accelerator.is_local_main_process:
        print(f"***** Evaluating {dataloader.dataset.dataset_path} *****")
        if epoch != -1:
            print(f"  Epoch = {epoch}")
            print(f"  Train step = {train_step}")
        print(f"  Num examples = {len(dataloader.dataset)}")
        print()

        os.makedirs(output_dir, exist_ok=True)

    model.eval()

    predicted_labels_scores = []

    dtype = get_dtype(accelerator)
    if dtype is not None:
        dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    if accelerator.is_local_main_process:
        sent_scorer = SentenceScorer()
    else:
        sent_scorer = None

    samples_seen: int = 0

    f1 = -1

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        with torch.cuda.amp.autocast(enabled=dtype is not None, dtype=dtype):
            outputs = model(**batch)
            logits = outputs.logits
            logits = torch.nn.functional.softmax(logits, dim=-1)

            input_tokens = (
                accelerator.gather(
                    accelerator.pad_across_processes(
                        batch.input_ids,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                        pad_first=tokenizer.padding_side == "left",
                    )
                )
                .cpu()
                .tolist()
            )

            logits = accelerator.gather(logits)
            gold_labels = accelerator.gather(batch.labels).cpu().tolist()

            if accelerator.is_local_main_process:
                if accelerator.num_processes > 1:
                    # Remove duplicated in last batch if we are in a distributed setting
                    if step == len(dataloader) - 1:
                        logits = logits[: (len(dataloader.dataset) - samples_seen)]
                        input_tokens = input_tokens[
                            : (len(dataloader.dataset) - samples_seen)
                        ]

                predicted_labels_scores.extend(logits[:, 1].tolist())
                predicted_labels_class = torch.argmax(
                    torch.tensor(logits), dim=-1
                ).tolist()

                input_sentences = tokenizer.batch_decode(
                    input_tokens, skip_special_tokens=True
                )

                sent_scorer.add_batch(
                    input_texts=input_sentences,
                    predicted_labels=predicted_labels_class,
                    gold_labels=gold_labels,
                )

    if accelerator.is_local_main_process:
        precision, recall, f1 = sent_scorer.compute()
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        sent_scorer.print_summary(output_dir=output_dir)

        conversation_scorer = ConversationAccuracyScorer(
            gold_data_path=dataloader.dataset.dataset_path,
            predicted_labels=predicted_labels_scores,
        )
        acc = conversation_scorer.compute()
        print(f"Conversation accuracy: {acc:.4f}")
        conversation_scorer.print_predictions(
            os.path.join(output_dir, "predictions.json")
        )
        with open(os.path.join(output_dir, "accuracy.txt"), "w", encoding="utf8") as f:
            print(f"Conversation accuracy: {acc}", file=f)

    model.train()

    return f1


def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
):
    accelerator = Accelerator()

    print(f"Using {accelerator.device} for training")

    if accelerator.is_local_main_process:
        print(f"Loading model from {model_args.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.update({"num_labels": 2})
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )
    if accelerator.is_local_main_process:
        print("Model loaded!")
        print_trainable_parameters(model)

    if training_args.do_train:
        if accelerator.is_local_main_process:
            wandb.init(
                project="IkerGetVocal",
                name=training_args.output_dir,
                config=training_args,
            )

        train_dataloader = get_dataloader(
            dataset_path=data_args.train_dataset,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,
            batch_size=training_args.per_device_train_batch_size,
            shuffle=True,
        )

        dev_dataloader = get_dataloader(
            dataset_path=data_args.validation_dataset,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,
            batch_size=training_args.per_device_eval_batch_size,
            shuffle=False,
        )

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader)
            / training_args.gradient_accumulation_steps
            / accelerator.num_processes
        )
        max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

        total_batch_size = (
            training_args.per_device_train_batch_size
            * accelerator.num_processes
            * training_args.gradient_accumulation_steps
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=float(training_args.learning_rate),
            eps=1e-7,
        )

        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=max_train_steps,
        )

        completed_steps = 0
        best_epoch_metric: float = -1
        running_loss = 0
        num_batches = 0
        first = True

        if accelerator.is_local_main_process:
            print("***** Running training *****")
            print(f"  Num examples = {len(train_dataloader.dataset)}")
            print(f"  Num Epochs = {training_args.num_train_epochs}")
            print(
                f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
            )
            print(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
            )
            print(
                f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}"
            )
            print(f"  Total optimization steps = {max_train_steps}")
            print(f"  Logging steps = {training_args.logging_steps}")
            print(f"  Save steps = {training_args.save_steps}")
            print(f"  Device = {accelerator.device}")
            print()

        progress_bar = tqdm(
            range(max_train_steps),
            disable=not accelerator.is_local_main_process,
            ascii=True,
            desc="Training",
        )

        for epoch in range(training_args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                ### DEBUG ###
                if first and accelerator.is_local_main_process:
                    decodeable_inputs = batch.input_ids.clone()
                    decodeable_inputs[decodeable_inputs == -100] = (
                        tokenizer.pad_token_id
                    )
                    model_inputs = "\n".join(
                        tokenizer.batch_decode(
                            decodeable_inputs[:4],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                    labels = batch.labels[:4]

                    print("*** Sample of batch 0 ***")
                    print(f"Model inputs:\n{model_inputs}")
                    print(f"Labels:\n{labels}")
                    print("*** End of sample ***")
                    first = False

                outputs = model(**batch)
                loss = outputs.loss
                running_loss += loss.item()
                loss = loss / training_args.gradient_accumulation_steps
                accelerator.backward(loss)
                num_batches += 1

                if (
                    step % training_args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                    if (
                        accelerator.is_local_main_process
                        and training_args.logging_steps > 0
                        and (completed_steps % training_args.logging_steps == 0)
                    ):
                        wandb.log(
                            {
                                "Train/Loss": loss.item(),
                                "Train/Running Loss": loss.item() / num_batches,
                                "Train/Learning Rate": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                                "step": completed_steps,
                            }
                        )

            dev_dataloader = accelerator.prepare(dev_dataloader)

            f1 = evaluate(
                dataloader=dev_dataloader,
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                output_dir=os.path.join(training_args.output_dir, f"epoch_{epoch}"),
                epoch=epoch,
                train_step=completed_steps,
            )

            if accelerator.is_local_main_process:
                wandb.log({"Dev/F1": f1, "epoch": epoch})
                if f1 > best_epoch_metric:
                    best_epoch_metric = f1
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        training_args.output_dir, save_function=accelerator.save
                    )
                    tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_predict:
        if training_args.do_train:
            model_path = training_args.output_dir
        else:
            model_path = model_args.model_name_or_path

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.update({"num_labels": 2})
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, config=config
        )
        model = accelerator.prepare(model)

        for dataset in data_args.test_datasets:
            test_dataloader = get_dataloader(
                dataset_path=dataset,
                tokenizer=tokenizer,
                max_length=tokenizer.model_max_length,
                batch_size=training_args.per_device_eval_batch_size,
                shuffle=False,
            )

            test_dataloader = accelerator.prepare(test_dataloader)

            evaluate(
                dataloader=test_dataloader,
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                output_dir=os.path.join(
                    training_args.output_dir,
                    os.path.splitext(os.path.basename(dataset))[0],
                ),
            )


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_yaml_file(
        yaml_file=os.path.abspath(sys.argv[-1])
    )
    main(model_args, data_args, training_args)
