import numpy as np

from net import net


def main():
    # 1. 创建模拟数据集
    X_train = np.random.randn(4, 2)
    y_train = np.array([[1,0,0], [0,1,0], [0,0,1], [0,1,0]])
    
    # 2. 网络初始化
    nn = net(layer_size=[2, 3, 3])

    # 3. 设置超参数
    learning_rate = 0.01
    epochs = 1000

    # 4. 训练
    for epoch in range(epochs):
        grad = nn.gradient(x=X_train, t=y_train)

        # 更新参数
        for key, val in grad.items():
            if key[0] == 'w':
                nn.w_list[int(key[1:])] -= learning_rate * val
            elif key[0] == 'b':
                nn.b_list[int(key[1:])] -= learning_rate * val
            else:
                print(f"grad 出现问题")
                return
            
        if epoch % 100 == 0:
            loss = nn.loss(X_train, y_train)
            print(f"Epoch {epoch}, loss = {loss:.6f}")

if __name__ == "__main__":
    main()
    pass