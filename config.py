import torch

save_dir = None
mean, std, num_classes, img_shape = None, None, None, None

# Dataset stats
dataset_stats = {
    'mnist': {
        'mean': torch.tensor([0.1306604762738431,]).view(1, -1, 1, 1),
        'std': torch.tensor([0.3081078038564622,]).view(1, -1, 1, 1),
        'num_classes': 10,
        'img_shape': (1, 28, 28),
    },
    'emnist_balanced': {
        'mean': torch.tensor([0.17501088976860046,]).view(1, -1, 1, 1),
        'std': torch.tensor([0.33314022421836853,]).view(1, -1, 1, 1),
        'num_classes': 47,
        'img_shape': (1, 28, 28),
    },
    'fashion_mnist': {
        'mean': torch.tensor([0.285908579826355,]).view(1, -1, 1, 1),
        'std': torch.tensor([0.35312530398368835,]).view(1, -1, 1, 1),
        'num_classes': 10,
        'img_shape': (1, 28, 28),
    },
    'cifar10': {
        'mean': torch.tensor([0.49139968, 0.48215827 ,0.44653124]).view(1, -1, 1, 1),
        'std': torch.tensor([0.24703233, 0.24348505, 0.26158768]).view(1, -1, 1, 1),
        'num_classes': 10,
        'img_shape': (3, 32, 32),
    },
    'cifar100': {
        'mean': torch.tensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343]).view(1, -1, 1, 1),
        'std': torch.tensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404]).view(1, -1, 1, 1),
        'num_classes': 100,
        'img_shape': (3, 32, 32),
    },
    'pcam': {
        'mean': torch.tensor([0.7001121044158936, 0.5378641486167908, 0.6908648610115051]).view(1, -1, 1, 1),
        'std': torch.tensor([0.23504479229450226, 0.2774609923362732, 0.21302694082260132]).view(1, -1, 1, 1),
        'num_classes': 2,
        'img_shape': (3, 96, 96),
    }
}