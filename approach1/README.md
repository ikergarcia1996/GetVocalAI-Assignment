# GetVocalAI-Assignment

# Requeriments

```
transformers - pip install transformers
accelerate - pip install accelerate
torch - pip install torch 
wandb - pip install wandb 
```

# Run 

```
accelerate launch --mixed_precision fp16 run.py configs/xlm-roberta.yaml
```
