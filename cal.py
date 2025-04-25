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













import torch

# 检查是否有可用 GPU
num_gpus = torch.cuda.device_count()
if num_gpus < 1:
    raise RuntimeError("No GPU devices found. This code requires at least one GPU.")

print(f"Detected {num_gpus} GPU(s).")

# 定义矩阵大小：行数为 350 * GPU数量，列数固定为 1000
rows = 350 * num_gpus
cols = 1000

# 创建矩阵 a 和 b
a = torch.randn(rows, cols).to('cuda:0')  # 初始矩阵 `a` 在主 GPU 上
b = torch.randn(cols, cols).to('cuda:0')  # 矩阵 `b` 在主 GPU 上

# 将矩阵 `a` 按行分为多个子矩阵，分配到多个 GPU
chunks_a = torch.chunk(a, num_gpus, dim=0)  # 按行分块
devices = [f'cuda:{i}' for i in range(num_gpus)]  # 获取所有 GPU 的设备名

# 将每一块子矩阵分配到对应的 GPU
chunks_a_on_gpus = [chunk.to(devices[i]) for i, chunk in enumerate(chunks_a)]
b_on_gpus = [b.to(devices[i]) for i in range(num_gpus)]  # 将 `b` 拷贝到每个 GPU

# 计算循环
while True:
    results = []
    for i, chunk in enumerate(chunks_a_on_gpus):
        # 在每块 GPU 上进行计算
        chunk = torch.matmul(chunk, b_on_gpus[i])  # 矩阵乘法
        chunk = torch.sin(chunk)                  # 元素逐项求 sin
        chunk = torch.exp(chunk)                  # 元素逐项求 exp
        chunk = chunk / (chunk + 1)               # 元素逐项计算
        chunk = torch.tanh(chunk)                 # 元素逐项求 tanh
        results.append(chunk)                     # 保存每块计算结果

    # 将所有 GPU 的计算结果汇总到主 GPU 上
    final_result = torch.cat([result.to('cuda:0') for result in results], dim=0)

    # 计算最终结果（求和）
    result_sum = final_result.sum().item()
    # print("Final result:", result_sum)