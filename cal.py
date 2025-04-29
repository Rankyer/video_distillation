# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# size = 1000
# a = torch.randn(size, size, device=device)
# b = torch.randn(size, size, device=device)

# while True:
#     a = torch.matmul(a, b)
#     a = torch.sin(a)
#     a = torch.exp(a)
#     a = a / (a + 1)
#     a = torch.tanh(a)
#     result = a.sum().item()
#     # print("result:", result)













# import torch

# # 检查是否有可用 GPU
# num_gpus = torch.cuda.device_count()
# if num_gpus < 1:
#     raise RuntimeError("No GPU devices found. This code requires at least one GPU.")

# print(f"Detected {num_gpus} GPU(s).")

# # 定义矩阵大小：行数为 350 * GPU数量，列数固定为 1000
# rows = 750 * num_gpus
# cols = 1000

# # 创建矩阵 a 和 b
# a = torch.randn(rows, cols).to('cuda:0')  # 初始矩阵 `a` 在主 GPU 上
# b = torch.randn(cols, cols).to('cuda:0')  # 矩阵 `b` 在主 GPU 上

# # 将矩阵 `a` 按行分为多个子矩阵，分配到多个 GPU
# chunks_a = torch.chunk(a, num_gpus, dim=0)  # 按行分块
# devices = [f'cuda:{i}' for i in range(num_gpus)]  # 获取所有 GPU 的设备名

# # 将每一块子矩阵分配到对应的 GPU
# chunks_a_on_gpus = [chunk.to(devices[i]) for i, chunk in enumerate(chunks_a)]
# b_on_gpus = [b.to(devices[i]) for i in range(num_gpus)]  # 将 `b` 拷贝到每个 GPU

# # 计算循环
# while True:
#     results = []
#     for i, chunk in enumerate(chunks_a_on_gpus):
#         # 在每块 GPU 上进行计算
#         chunk = torch.matmul(chunk, b_on_gpus[i])  # 矩阵乘法
#         chunk = torch.sin(chunk)                  # 元素逐项求 sin
#         chunk = torch.exp(chunk)                  # 元素逐项求 exp
#         chunk = chunk / (chunk + 1)               # 元素逐项计算
#         chunk = torch.tanh(chunk)                 # 元素逐项求 tanh
#         results.append(chunk)                     # 保存每块计算结果

#     # 将所有 GPU 的计算结果汇总到主 GPU 上
#     final_result = torch.cat([result.to('cuda:0') for result in results], dim=0)

#     # 计算最终结果（求和）
#     result_sum = final_result.sum().item()
#     # print("Final result:", result_sum)










import torch
import time

# 检查是否有可用的 GPU
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No GPU available!")

# print(f"Detected {num_gpus} GPU(s).")

# 超参数：通过调整矩阵大小控制计算负载
power_level = 8192  # 矩阵大小，例如 1024 表示 1024x1024 的矩阵

# 为每个 GPU 创建张量
device_tensors = []
for i in range(num_gpus):
    device = torch.device(f"cuda:{i}")
    # 每个 GPU 上创建一个矩阵大小为 power_level 的张量
    dummy_tensor = torch.ones((power_level, power_level), device=device)
    device_tensors.append(dummy_tensor)

# print(f"Set GPU computation power to matrix size: {power_level}x{power_level}")

# 无限循环，在每个 GPU 上模拟高负载
# print("Running high-power computation on all GPUs...")
try:
    while True:
        for i in range(num_gpus):
            # 执行简单的矩阵乘法，增加计算功率
            device_tensors[i] = torch.matmul(device_tensors[i], device_tensors[i].T)

            # 保持显存占用低
            device_tensors[i] = device_tensors[i] / 1.0001

        # 偶尔让出 CPU（可选）
        time.sleep(0.075)

except KeyboardInterrupt:
    # print("Stopped high-power computation.")
    pass