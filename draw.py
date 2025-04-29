import os
import matplotlib.pyplot as plt
from datetime import datetime

# 读取 Loss 数据
# loss_file_path = './Loss/loss_log_20250415_090138.txt'
# loss_file_path = './Loss/loss_log_20250416_203218.txt'
# loss_file_path = './Loss/loss_log_20250416_203243.txt'
loss_file_path = './Loss/loss_log_20250416_203247.txt'

steps_loss = []
loss_values = []

with open(loss_file_path, 'r') as f:
    for line in f:
        step, loss = line.strip().split(',')
        steps_loss.append(int(step))
        loss_values.append(float(loss))

# 读取 Accuracy 数据
# accuracy_file_path = './sh/cf/CF_loss300_eval50_focal_1.0_2.log'
# accuracy_file_path = './sh/cf/CF_loss200_sqrt_loss_1.log'
# accuracy_file_path = './sh/cf/CF_loss200_sqrt_loss_6.log'
accuracy_file_path = './sh/cf/CF_loss200_sqrt_loss_7.log'

steps_accuracy = []
accuracy_values = []

with open(accuracy_file_path, 'r') as f:
    for line in f:
        if 'iteration =' in line:
            step = int(line.split('iteration = ')[1])
        if 'mean =' in line:
            accuracy = float(line.split('mean = ')[1].split(' ')[0])
            steps_accuracy.append(step)
            accuracy_values.append(accuracy)

# 创建图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制 Loss 曲线（左侧纵坐标）
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss', color='tab:blue')
ax1.plot(steps_loss, loss_values, color='tab:blue', label='Loss', linestyle='-', marker=None)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# 绘制 Accuracy 曲线（右侧纵坐标）
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy', color='tab:red')
ax2.plot(steps_accuracy, accuracy_values, color='tab:red', label='Accuracy', linestyle='--', marker='x')
ax2.tick_params(axis='y', labelcolor='tab:red')

# 添加标题和图例
plt.title('Loss and Accuracy vs. Step')
fig.tight_layout()  # 调整布局防止重叠
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)

# 创建保存路径
output_dir = './draw'
os.makedirs(output_dir, exist_ok=True)  # 如果文件夹不存在则创建

# 获取当前时间并生成文件名
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = os.path.join(output_dir, f'{current_time}.png')

# 保存图片
plt.savefig(output_path)
print(f'图表已保存到: {output_path}')

# 显示图表（可选）
plt.show()