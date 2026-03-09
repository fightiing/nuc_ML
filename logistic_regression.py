#一般会导入pandas和numpy包 此处不用画图就没导入画图包
import numpy as np
import pandas as pd
#导入数据集划分的函数
from sklearn.model_selection import train_test_split
#导入数据集预处理的函数
from sklearn.preprocessing import MinMaxScaler
#导入逻辑回归模型
from sklearn.linear_model import LogisticRegression
#帮助我们进行分类的一个计算，比如我们的精确度和召回率F1
from sklearn.metrics import classification_report

#第一步读取数据
dataset = pd.read_csv('breast_cancer_data.csv')
#可以打印运行一下看看
#print(dataset)

#提取x 约定俗成用大X表示特征
X = dataset.iloc[:, :-1]
#把除了最后一列的数据给提取出来了
#print(X)

#提取数据中的标签Y
Y = dataset['target']
#print(Y)

#划分数据集和测试集
#八份是我们的训练集，两份是我们的测试集
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)

#数据的归一化
#先让他实例化一下
#实例化一个对象，使值规划到0-1之间
sc = MinMaxScaler(feature_range=(0,1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#print(x_train)

#利用sklearn搭建逻辑回归模型
lr = LogisticRegression()
lr.fit(x_train,y_train)

#打印模型的参数
# print('w:',lr.coef_)
# print('b:',lr.intercept_)
#训练集测试集是随机划分的，所以结果不一样会很正常

#利用训练好的模型进行推理测试
pre_result = lr.predict(x_test)
#print(pre_result)

#每个预测结果对应的概率
pre_result_proba = lr.predict_proba(x_test)
#print(pre_result_proba)

#获取恶性肿瘤的概率，也就是刚才结果的第二列
pre_list = pre_result_proba[:,1]
#print(pre_list)

#默认阈值是50%，大于是恶性
#我们也可以调整阈值
thresholds = 0.3

#设置保存结果的列表
result = []
result_name = []

for i in range(len(pre_list)):
    if pre_list[i] >= thresholds:
        result.append(1) #1就是恶性
        result_name.append('恶性')
    else:
        result.append(0)
        result_name.append('良性')

#打印调整阈值后的结果
# print(result)
# print(result_name)

#检测模型好不好
#精确率召回f1值 要计算精度，就要有预测值和真实值的对比
report = classification_report(y_test, result, labels=[0,1],target_names=['良性肿瘤','恶性肿瘤'])
print(report)