import torch

checkpoint = torch.load("./pretrain/premodel1_trained.pth", map_location="cpu")

if isinstance(checkpoint, dict):
    print("The file contains a state_dict.")
    print(checkpoint.keys())  # 查看 state_dict 的内容
else:
    print("The file contains an entire model.")
    print(type(checkpoint))  # 查看模型的类型