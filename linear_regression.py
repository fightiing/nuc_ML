#1.定义数据集
#定义数据特征
x_data = [1,2,3]
#定义数据标签
y_data = [2,4,6]
#初始化w参数
w = 4

#定义线性回归模型
def forward(x):
    return x * w

#定义损失函数
def cost(xs,ys):
    costvalue = 0
    for x,y in zip(xs,ys):
        y_pred = forward(x)
        costvalue += (y_pred - y)**2
    return costvalue / len(xs)
#利用平均的损失来进行一个梯度更新

#定义计算梯度的函数
def gradient(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2 * x * (x * w - y)  #更新梯度
    return grad/len(xs)

#更新 设置轮次
for epoch in range(100):
    #通过损失函数的函数来计算误差损失:
    cost_val = cost(x_data,y_data) #传入我们定义好的特征数据和标签数据
    #计算梯度
    grad_val = gradient(x_data,y_data)
    #利用梯度进行参数更新
    w = w - 0.01 * grad_val
    #挨个打印出来
    print('训练轮次：',epoch,"w=",w,"loss",cost_val)

print("100轮后w已经训练好了，此时我们用训练好的w进行推理，学习时间为四个小时的时候最终的得分为：",forward(4))


