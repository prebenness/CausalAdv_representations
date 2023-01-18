import os
import torch
import numpy as np
from progress.bar import Bar

from causaladv_utils import Basis, PredYWithS, get_dataset, get_s_pre, init_anchor
from resnet import ResNet18


def store_representations(args):

    num_classes = 100 if args.dataset == 'cifar100' else 10

    # Get dataset
    train_loader, test_loader = get_dataset(args)

    # Load pre-trained models
    model_path = os.path.join(args.model_path)
    loaded_dict = torch.load(model_path)

    ## Net: Model h()
    net = ResNet18(num_classes=num_classes).to(args.device)
    net.load_state_dict(loaded_dict['net'])
    net.cuda()      # Set model to use CUDA device
    net.eval()      # Set model to not be in train mode

    ## Basis
    basis = loaded_dict['b']
    
    ## Model g()
    model_g = PredYWithS(feat_dim=basis.size(1), num_classes=num_classes).to(args.device)
    model_g.load_state_dict(loaded_dict['g'])

    # Forward pass over dataset and store representations
    split_dict = { 'train': train_loader, 'test': test_loader }
    for split_name, loader in split_dict.items():
        
        print(f'Storing representations for {split_name} dataset')
        bar = Bar('Processing', max=len(loader), index=0)

        images, style, content = [], [], []
        for idx, (x, y) in enumerate(loader):
            x, y = x.to(args.device), y.to(args.device)     # Load data to GPU

            with torch.no_grad():
                net.is_train(True)                          # Set train_state to True to get z and W_c*z
                z, y_c = net(x)                             # ResNet forward pass

                # Construct orhogonal projection from final ResNet layer
                anchor = init_anchor(weight=net.fc[0].weight.data.detach())
                v_space = Basis(anchor=anchor)
                basis = v_space.get_basis(f=z, weight=net.fc[0].weight.data)

                # Estimate of mean of style content
                mu_s = get_s_pre(z, basis)
                y_s = model_g(mu_s.detach().clone(), y, ratio=args.adv_ratio)

                # Store image, style, content
                images.append(x)
                style.append(y_s)
                content.append(y_c)

            bar.next()
        bar.finish()

        ## Save images, contents and styles for test and train sets
        images = torch.cat(images, 0).detach().cpu().numpy()
        style = torch.cat(style, 0).detach().cpu().numpy()
        content = torch.cat(content, 0).detach().cpu().numpy()

        outpath = os.path.join(args.output_dir, 'representations', f'{args.dataset}-{args.model_name}')
        os.makedirs(outpath, exist_ok=True)
        np.savez(os.path.join(outpath, f'content_{split_name}.npz'), content)
        np.savez(os.path.join(outpath, f'images_{split_name}.npz'), images)
        np.savez(os.path.join(outpath, f'style_{split_name}.npz'), style)

