from load_model import load_model
from accelerate import Accelerator
from dataset import get_dataloader
from tqdm.auto import tqdm
import torch


@torch.no_grad()
def main(model_name_or_path, dataset_path, batch_size: int = 4):
    model, tokenizer = load_model(
        inference=True, model_weights_name_or_path=model_name_or_path, quantization=4
    )
    test_dataloader = get_dataloader(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    accelerator = Accelerator()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model.eval()

    if accelerator.is_local_main_process:
        print("***** Running Evaluation *****")
        print("  Num examples = %d", len(test_dataloader.dataset))
        print("  Batch size = %d", batch_size)

    losses = []
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        labels = batch.pop("labels")
        loss_weight_mask = batch.pop("loss_weight_mask")
        outputs = model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        loss_weight_mask = loss_weight_mask[..., 1:].contiguous()


        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss_weight_mask = loss_weight_mask.view(-1)
        
        