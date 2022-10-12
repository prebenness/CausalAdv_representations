# Train

```bash
python causaladv.py --train=true --dataset={cifar10, cifar100}
```

# Compute representations
```bash
python causaladv.py --store_repr=true --dataset={cifar10, cifar100}
```

Requires pre-trained model. Searches for model in: 
```python
model_path = os.path.join(
    args.output_dir, f'{args.dataset}-{args.model_name}-best.pth'
)
```