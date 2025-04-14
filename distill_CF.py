import datetime
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm, trange
# from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, TensorDataset, epoch, get_loops, match_loss, ParamDiffAug, Conv3DNet
import wandb
import copy
import random
from reparam_module import ReparamModule
import warnings

import importlib.util
import sys

# from utils.utils import update_feature_extractor

utils_file_path = "./utils.py"
spec = importlib.util.spec_from_file_location("utils_file", utils_file_path)
utils_file = importlib.util.module_from_spec(spec)
sys.modules["utils_file"] = utils_file
spec.loader.exec_module(utils_file)

from utils.utils import update_feature_extractor

get_dataset = utils_file.get_dataset
get_network = utils_file.get_network
get_eval_pool = utils_file.get_eval_pool
evaluate_synset = utils_file.evaluate_synset
get_time = utils_file.get_time
DiffAugment = utils_file.DiffAugment
TensorDataset = utils_file.TensorDataset
epoch = utils_file.epoch
get_loops = utils_file.get_loops
# match_loss = utils_file.match_loss
ParamDiffAug = utils_file.ParamDiffAug
Conv3DNet = utils_file.Conv3DNet




from NCFM.NCFM import match_loss, cailb_loss, mutil_layer_match_loss, CFLossFunc

from utils.diffaug import diffaug


warnings.filterwarnings("ignore", category=DeprecationWarning)

import distill_utils

def main(args):
    if args.outer_loop is None and args.inner_loop is None:
        args.outer_loop, args.inner_loop = get_loops(args.ipc)
    elif args.outer_loop is None or args.inner_loop is None:
        raise ValueError(f"Please set neither or both outer/inner_loop: {args.outer_loop}, {args.inner_loop}")
    print('outer_loop = %d, inner_loop = %d'%(args.outer_loop, args.inner_loop))

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    print('Evaluation iterations: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader= get_dataset(args.dataset, args.data_path)
    if args.preload:
        print("Preloading dataset")
        video_all = []
        label_all = []
        for i in trange(len(dst_train)):
            _ = dst_train[i]
            video_all.append(_[0])
            label_all.append(_[1])
        video_all = torch.stack(video_all)
        label_all = torch.tensor(label_all)
        dst_train = torch.utils.data.TensorDataset(video_all, label_all)
    
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    project_name = "Baseline_{}".format(args.method)

    wandb.init(sync_tensorboard=False,
               project=project_name,
               job_type="CleanRepo",
               config=args,
               name = f'{args.dataset}_ipc{args.ipc}_{args.lr_img}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
               )
    
    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    labels_all = label_all if args.preload else dst_train.labels
    indices_class = [[] for c in range(num_classes)]

    print("BUILDING DATASET")
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        if n == 1:
            imgs = dst_train[idx_shuffle[0]][0].unsqueeze(0)
        else:
            imgs = torch.cat([dst_train[i][0].unsqueeze(0) for i in idx_shuffle], 0)
        return imgs.to(args.device)

    image_syn = torch.randn(size=(num_classes*args.ipc, args.frames, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor(np.stack([np.ones(args.ipc)*i for i in range(0, num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    syn_lr = torch.tensor(args.lr_teacher).to(args.device) if args.method == 'MTT' else None

    if args.init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(0, num_classes):
            i = c 
            image_syn.data[i*args.ipc:(i+1)*args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')

    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(args.train_lr) if args.method == 'MTT' else None
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5) if args.train_lr else None
    optimizer_img.zero_grad()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_img, mode="min", factor=0.5, patience=500, verbose=False
    )

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    

    # aug_fn, _ = diffaug(args)
    
    if args.method == "CF":
        print("CF")
        # for it in trange(0, args.Iteration+1, ncols=60):
        for it in range(0, args.Iteration+1):
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                # print(it)
                save_this_best_ckpt = False
                for model_eval in model_eval_pool:
                    print('Evaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    accs_test = []
                    accs_train = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                        image_syn_eval, label_syn_eval = image_syn.detach().clone(), label_syn.detach().clone() # avoid any unaware modification
                        _, acc_train, acc_test, acc_per_cls = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, mode='none',test_freq=250)

                        accs_test.append(acc_test)
                        accs_train.append(acc_train)
                        print("acc_per_cls:",acc_per_cls)
                    accs_test = np.array(accs_test)
                    accs_train = np.array(accs_train)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_best_ckpt = True
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                        len(accs_test), model_eval, acc_test_mean, acc_test_std))
                    wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                    wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                    wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                    wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

                    print(f"Max_Accuracy{best_acc[model_eval]}")
            
            if it in eval_it_pool and (save_this_best_ckpt or it % 1000 == 0):
                image_save = image_syn.detach()
                save_dir = os.path.join(args.save_path, project_name, wandb.run.name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                if save_this_best_ckpt:
                    save_this_best_ckpt = False
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))


            model_init = get_network(args.model, channel, num_classes, im_size).to(args.device)
            model_interval = get_network(args.model, channel, num_classes, im_size).to(args.device)
            model_final = get_network(args.model, channel, num_classes, im_size).to(args.device)

            model_init, model_final, model_interval = update_feature_extractor(
                args, model_init, model_final, model_interval, a=0, b=1
            )

            # print("model loaded!")

            cf_loss_func = CFLossFunc(0.5, 0.5)

            match_loss_total = 0

            loss = torch.tensor(0.0).to(args.device)
            for c in range(0,num_classes):

                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, args.frames, channel, im_size[0], im_size[1]))

                # img_aug = aug_fn(torch.cat([img_real, img_syn]))
                # n = img_real.shape[0]

                

                loss = match_loss(img_real, img_syn, model_interval, cf_loss_func)

                # loss = match_loss(img_aug[:n], img_aug[n:], model_interval, cf_loss_func)
                
                match_loss_total += loss.item()

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()

            # if args.sampling_net:
            #     sampling_net = SampleNet(feature_dim=2048)
            #     optim_sampling_net =
            # scheduler = optim.lr_scheduler.MultiStepLR(
            #     optimizer_img, milestones=[1 * args.niter // 4,2 * args.niter // 4,  3* args.niter // 4], gamma=0.8
            # )


            current_loss = (
                (match_loss_total) / args.nclass
            )
            scheduler.step(current_loss)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='dataset')

    parser.add_argument('--method', type=str, default='DC', help='MTT or DM')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='use top5 to eval top5 accuracy, use S to eval single accuracy')
    
    parser.add_argument('--outer_loop', type=int, default=None, help='')
    parser.add_argument('--inner_loop', type=int, default=None, help='')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=50, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='how many distillation steps to perform')

    parser.add_argument('--lr_net', type=float, default=0.001, help='learning rate for network')
    parser.add_argument('--lr_img', type=float, default=1, help='learning rate for synthetic data')
    parser.add_argument('--lr_lr', type=float, default=1e-5, help='learning rate for synthetic data')
    parser.add_argument('--lr_teacher', type=float, default=0.001, help='learning rate for teacher')
    parser.add_argument('--train_lr', action='store_true', help='train synthetic lr')
    
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn')

    parser.add_argument('--init', type=str, default='real', choices=['noise', 'real', 'real-all'], help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--data_path', type=str, default='distill_utils/data', help='dataset path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=64, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    parser.add_argument('--buffer_path', type=str, default=None, help='buffer path')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--preload', action='store_true', help='preload dataset')
    parser.add_argument('--save_path',type=str, default='./logged_files', help='path to save')
    parser.add_argument('--frames', type=int, default=16, help='')


    parser.add_argument('--pretrain_dir', type=str, default='./pretrain', help='pretrain model path')

    parser.add_argument('--nclass', type=int, default=50, help='number of class')

    parser.add_argument('--aug_type', type=str, default="color_crop_cutout", help='augmentation type')
    parser.add_argument('--mixup', type=str, default="cut", help='mixup')



    args = parser.parse_args()

    main(args)

