# Train

```bash
python causaladv.py --train --dataset={cifar10, cifar100}
```

# Compute representations
```bash
python causaladv.py --store_repr --dataset={cifar10, cifar100}
```

Requires pre-trained model. Searches for model in: 
```python
model_path = os.path.join(
    args.output_dir, f'{args.dataset}-{args.model_name}-best.pth'
)
```