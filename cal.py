import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size = 10000
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

while True:
    a = torch.matmul(a, b)
    a = torch.sin(a)
    a = torch.exp(a)
    a = a / (a + 1)
    a = torch.tanh(a)
    result = a.sum().item()
    # print("result:", result)