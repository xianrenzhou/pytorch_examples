import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取CSV文件
df = pd.read_csv('./data/model_quadratic_function.csv')

# 假设CSV文件有两列：'x'和'y'
X = df[['x']].values
Y = df[['y']].values

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y)


# 转换为PyTorch的张量
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
X_tensor = X_tensor.to(device)
Y_tensor = Y_tensor.to(device)

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1,70),
            nn.Sigmoid(),
            nn.Linear(70,1)
        )

    def forward(self, x):
        y_pred = self.linear_relu_stack(x)
        return y_pred

# 创建模型实例
model = LinearRegressionModel().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 10000
plt.figure("test")
plt.ion()
plt.show()
for epoch in range(num_epochs):
    model.train()
    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        plt.cla()
        plt.plot(X_tensor.cpu().detach().numpy(),Y_tensor.cpu().detach().numpy())
        plt.plot(X_tensor.cpu().detach().numpy(),outputs.cpu().detach().numpy())
        plt.pause(0.01)
model.eval()
with torch.no_grad():
    predictions = model(X_tensor)
    print(predictions)
    predictions = scaler_Y.inverse_transform(predictions.cpu().numpy())  # 转移到CPU进行逆标准化

# 打印一些预测值和实际值
for i in range(50):
    print(f'Predicted: {predictions[i][0]:.4f}, Actual: {df["y"].values[i]}')