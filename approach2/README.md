# GetVocalAI-Assignment

# Requeriments

```
transformers - pip install transformers
accelerate - pip install accelerate
torch - pip install torch 
wandb - pip install wandb 
flash attention - pip install flash-attn --no-build-isolation
bitsandbytes - pip install bitsandbytes
peft - pip install peft
```

# Run 

```
accelerate launch run.py --model_name_or_path google/gemma-2b-it --dataset_path ../data/daily_dialog_test.jsonl --batch_size 8
```
