import datetime
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from collections import defaultdict
from tqdm import tqdm, trange


import importlib.util
import sys

# 动态导入 utils.py 文件
utils_file_path = "./utils.py"  # 指定 utils.py 文件的路径
spec = importlib.util.spec_from_file_location("utils_file", utils_file_path)
utils_file = importlib.util.module_from_spec(spec)
sys.modules["utils_file"] = utils_file
spec.loader.exec_module(utils_file)

# # 正常从 utils 文件夹导入模块
# from utils.utils import update_feature_extractor

# 使用 utils.py 文件中的内容
get_dataset = utils_file.get_dataset
get_network = utils_file.get_network
get_eval_pool = utils_file.get_eval_pool
evaluate_synset = utils_file.evaluate_synset
get_time = utils_file.get_time
DiffAugment = utils_file.DiffAugment
TensorDataset = utils_file.TensorDataset
epoch = utils_file.epoch
get_loops = utils_file.get_loops
match_loss = utils_file.match_loss
ParamDiffAug = utils_file.ParamDiffAug
Conv3DNet = utils_file.Conv3DNet



import wandb
import copy
import random
from reparam_module import ReparamModule
import warnings
import time
# from utils.utils import update_feature_extractor


from NCFM.NCFM import match_loss, cailb_loss, mutil_layer_match_loss, CFLossFunc




warnings.filterwarnings("ignore", category=DeprecationWarning)

import distill_utils





def train_model(net, trainloader, testloader, args, test_freq=None):
    lr = float(args.lr_net)
    Epoch = int(args.epoch_train)
    lr_schedule = [Epoch//2+1]#, 3*Epoch//4+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    start = time.time()


    for ep in tqdm(range(Epoch + 1)):
        loss_train, acc_train, _= epoch('train', trainloader, net, optimizer, criterion, args)
        if (test_freq is None and ep == Epoch) or (test_freq is not None and ep % test_freq == 0 and ep != 0):
            with torch.no_grad():
                loss_test, acc_test, acc_per= epoch('test', testloader, net, optimizer, criterion, args)
                if args.eval_mode != 'top5':
                    print('%s : Ep %d time = %ds loss = %.6f train acc = %.2f, test acc = %.2f' % (get_time(), ep, int(time.time() - start), loss_train, acc_train*100, acc_test*100))
                    #print('acc_per', acc_per)
        if ep in lr_schedule:
            lr *= 0.1
            print('lr = %.6f'%lr)
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        if ep % 10 == 0 and args.eval_mode == 'test':
            print("Epoch: %d, loss: %.6f, acc_train: %.2f" % (ep, loss_train, acc_train*100))
    time_train = time.time() - start
    if args.eval_mode != 'top5':
        print('%s : Ep %d time = %ds loss = %.6f train acc = %.2f, test acc = %.2f' % (get_time(), Epoch, int(time_train), loss_train, acc_train*100, acc_test*100))

    return net, acc_train, acc_test, acc_per



def epoch(mode, dataloader, net, optimizer, criterion, args):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    top5_acc_avg, top3_acc_avg, top1_acc_avg= 0.0, 0.0, 0.0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    correct_per_class = defaultdict(list)

    if mode == 'train':
        for i_batch, datum in enumerate(dataloader):
            img = datum[0].float().to(args.device)
            if 'Video' in args.model:
                img = img[:,:, :, 24:-24,24:-24]
            img = (img - img.mean()) / img.std()
            lab = datum[1].long().to(args.device)
            n_b = lab.shape[0]

            output = net(img)
            loss = criterion(output, lab)
            matched = np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy())
            acc = np.sum(matched)
            top5_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -5:]
            top5_matched = np.array([lab.cpu().data.numpy()[i] in top5_preds[i] for i in range(len(lab))])
            top5_acc = np.sum(top5_matched)

            for y, c in zip(lab.cpu().tolist(), matched.tolist()):
                correct_per_class[y].append(c)
                
            loss_avg += loss.item()*n_b
            acc_avg += acc
            num_exp += n_b
            top5_acc_avg += top5_acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else :
        for j_ in range(3):
            for i_batch, datum in enumerate(dataloader):
                img = datum[0].float().to(args.device)
                if 'Video' in args.model:
                    img = img[:,:, :, 24:-24,24:-24]
                img = (img - img.mean()) / img.std()
                lab = datum[1].long().to(args.device)
                n_b = lab.shape[0]

                output = net(img)
                loss = criterion(output, lab)
                acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
                top1_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -1:]
                top1_matched = np.array([lab.cpu().data.numpy()[i] in top1_preds[i] for i in range(len(lab))])
                top1_acc = np.sum(top1_matched)
                top3_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -3:]
                top3_matched = np.array([lab.cpu().data.numpy()[i] in top3_preds[i] for i in range(len(lab))])
                top3_acc = np.sum(top3_matched)
                top5_preds = np.argsort(output.cpu().data.numpy(), axis=-1)[:, -5:]
                top5_matched = np.array([lab.cpu().data.numpy()[i] in top5_preds[i] for i in range(len(lab))])
                top5_acc = np.sum(top5_matched)

                for y, c in zip(lab.cpu().tolist(), np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()).tolist()):
                    correct_per_class[y].append(c)

                loss_avg += loss.item()*n_b
                acc_avg += acc
                top5_acc_avg += top5_acc
                top3_acc_avg += top3_acc
                top1_acc_avg += top1_acc
                num_exp += n_b

    loss_avg /= num_exp
    acc_avg /= num_exp
    top5_acc_avg /= num_exp
    top3_acc_avg /= num_exp
    top1_acc_avg /= num_exp

    top_acc_avg = [acc_avg, top1_acc_avg, top3_acc_avg, top5_acc_avg]

    correct_per_class = dict(correct_per_class)
    correct_per_class = [
        np.mean(correct_per_class[i]) 
            if i in correct_per_class 
            else None 
        for i in range(len(correct_per_class))]

    if args.eval_mode == 'top5':
        return loss_avg, top_acc_avg, correct_per_class
    else:
        return loss_avg, acc_avg, correct_per_class
















def main(args):


    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)


    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    save_this_best_ckpt = False
    for model_eval in model_eval_pool:
        # print('Evaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        best_acc = {m: 0 for m in model_eval_pool}
        best_std = {m: 0 for m in model_eval_pool}

        accs_test = []
        accs_train = []

        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model

        _, acc_train, acc_test, acc_per_cls = train_model(net_eval, trainloader, testloader, args, test_freq=100)



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





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='dataset')

    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='use top5 to eval top5 accuracy, use S to eval single accuracy')

    parser.add_argument('--epoch_train', type=int, default=1000,
                        help='epochs to train a model')

    parser.add_argument('--lr_net', type=float, default=0.001, help='learning rate for network')
    
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--data_path', type=str, default='distill_utils/data', help='dataset path')

    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--preload', action='store_true', help='preload dataset')
    parser.add_argument('--save_path',type=str, default='./logged_files', help='path to save')


    args = parser.parse_args()

    main(args)

