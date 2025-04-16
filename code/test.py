import pandas as pd
import matplotlib.pyplot as plt

# 修改为实际CSV文件的路径
csv_file = "/data/qin2/chein/NeutraPro/code/PR-AUC.csv"

# 读取 CSV 文件
df = pd.read_csv(csv_file)

# 假设 CSV 文件中第二列为 Epoch，第三列为 Loss
epochs = df.iloc[:, 1]
loss = df.iloc[:, 2]

# 绘制 loss 曲线图
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, label="Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.title("PR-AUC Curve")
plt.legend()
plt.grid(True)

# 保存图片到指定路径，这里保存为 loss_curve.png
plt.savefig("/data/qin2/chein/NeutraPro/result/PR-AUC.png", dpi=300)

plt.show()