import torch
from torch import nn
import matplotlib.pyplot as plt


# 构造数据
T= 1000
time = torch.arange(1, T + 1, dtype=torch.float16)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,), dtype=torch.float16) # 正弦波叠加噪声
plt.figure(figsize=(6, 3))
plt.plot(time, x)
plt.show()
# print(time)

# 制作时间序列回归预测数据集
tau = 4
def create_dataset(x, tau):
    n = x.shape[0] - tau
    y = x[tau:].reshape((-1, 1))
    X = torch.zeros((n, tau), dtype=torch.float16)
    for i in range(tau):
        X[:, i] = x[i: i + n]
    return X, y