import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import importlib.util
import sys
import time
from tqdm import trange

# 动态加载 utils 文件（重名文件）
utils_file_path = "./utils.py"
spec = importlib.util.spec_from_file_location("utils_file", utils_file_path)
utils_file = importlib.util.module_from_spec(spec)
sys.modules["utils_file"] = utils_file
spec.loader.exec_module(utils_file)

get_dataset = utils_file.get_dataset
get_network = utils_file.get_network
epoch = utils_file.epoch

def save_model_checkpoint(model, save_path, filename):
    """保存模型到指定路径"""
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, filename))

def load_last_model_checkpoint(save_path, model, model_number):
    """加载指定编号的模型"""
    trained_model_path = os.path.join(save_path, f"premodel{model_number}_trained.pth")
    if os.path.exists(trained_model_path):
        model.load_state_dict(torch.load(trained_model_path))
        print(f"Loaded model: {trained_model_path}")
    else:
        print(f"No trained model found for model number {model_number}. Starting from scratch.")
    return model

def get_completed_model_count(save_path):
    """获取已经完成训练的模型数量"""
    if not os.path.exists(save_path):
        return 0
    completed_models = [f for f in os.listdir(save_path) if f.startswith("premodel") and f.endswith("_trained.pth")]
    if not completed_models:
        return 0
    completed_numbers = [int(f.split("premodel")[1].split("_")[0]) for f in completed_models]
    return max(completed_numbers)

def log_test_accuracy(log_path, model_number, accuracy):
    """记录模型的测试准确率"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"Model {model_number}: Test Accuracy = {accuracy:.4f}\n")

def train_model(net, trainloader, testloader, args, model_number):
    """训练单个模型"""
    lr = float(args.lr_net)
    Epoch = int(args.epoch_train)
    lr_schedule = [Epoch // 2 + 1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    save_path = os.path.join(os.getcwd(), "pretrain")
    save_model_checkpoint(net, save_path, f"premodel{model_number}_init.pth")

    for ep in tqdm(range(Epoch + 1)):
        loss_train, acc_train, _ = epoch('train', trainloader, net, optimizer, criterion, args)
        if ep % 100 == 0 or ep == Epoch:
            with torch.no_grad():
                loss_test, acc_test, _ = epoch('test', testloader, net, optimizer, criterion, args)
                save_model_checkpoint(net, save_path, f"premodel{model_number}_trained.pth")
                print(f"Epoch {ep}: Train Acc = {acc_train:.2f}, Test Acc = {acc_test:.2f}")

        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    return acc_train, acc_test

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, _, _, _, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)


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

    save_path = os.path.join(os.getcwd(), "pretrain")
    log_path = os.path.join(save_path, "accuracy_log.txt")

    completed_model_count = get_completed_model_count(save_path)
    remaining_model_count = max(0, args.num_model - completed_model_count)

    if remaining_model_count == 0:
        print(f"All {args.num_model} models are already trained.")
        return

    print(f"Starting training for {remaining_model_count} models...")
    start_model_number = completed_model_count + 1

    for model_idx in range(remaining_model_count):
        model_number = start_model_number + model_idx
        print(f"\nTraining Model {model_number}...")

        net = get_network(args.model, channel, num_classes, im_size).to(args.device)

        # 加载断点
        if args.resume and model_idx == 0 and completed_model_count > 0:
            net = load_last_model_checkpoint(save_path, net, completed_model_count)

        _, acc_test = train_model(net, trainloader, testloader, args, model_number)

        log_test_accuracy(log_path, model_number, acc_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='Dataset name')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='Model name')
    parser.add_argument('--epoch_train', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='use top5 to eval top5 accuracy, use S to eval single accuracy')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--lr_net', type=float, default=0.001, help='Learning rate for network')
    parser.add_argument('--batch_train', type=int, default=256, help='Batch size for training')
    parser.add_argument('--data_path', type=str, default='distill_utils/data', help='Path to dataset')
    parser.add_argument('--num_model', type=int, default=15, help='Number of models to train')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from last checkpoint')
    
    parser.add_argument('--preload', action='store_true', help='preload dataset')

    args = parser.parse_args()
    main(args)