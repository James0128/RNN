# -*- coding: utf-8 -*- 2

import  numpy as np
import  tensorflow as tf
print (tf.__version__)
#加载matplotlib工具包，对预测的sin函数曲线进行绘图
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

learn = tf.contrib.learn

HIDDEN_SIZE = 30 #LSTM中隐藏节点的个数
NUM_LAYERS =2 #LSTM的层数
TIMESTEPS =10 #循环神经网络的截断长度
TRAINING_STEP = 10000 #训练轮数

BATCH_SIZE = 32 #batch的大小
TRAINING_EXAMPLES =10000 #训练数据个数
TESTING_EXAMPLES =1000 #测试数据个数
SAMPLE_GAP =0.01 #采样间隔

def generate_data(seq):
    X=[]
    y=[]
    for i in range(len(seq)-TIMESTEPS-1):
        X.append([seq[i:i+TIMESTEPS]])
        y.append(seq[i+TIMESTEPS])
    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(X,y):
    #使用多层lstm结构
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)
    x_ = tf.unpack(X,axis =1)

    #计算前向传播结果
    output,_ = tf.nn.rnn(cell,x_,dtype=tf.float32)
    #将结果作为下一时刻的预测值
    output = output[-1]

    #对LSTM网络的输出再做加一层全链接层并计算损失。默认的损失为平均平方差损失函数
    prediction , loss = learn.models.linear_regression(output,y)
    #创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(loss,tf.contrib.framework.get_gloabal_step(),optimizer="Adagrad",learning_rate=0.1)

    return prediction,loss,train_op

#建立深层循环网络模型
regressor =learn.Estimator(model_fn = lstm_model())
#用正弦函数生成训练和测试数据合集
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end= (TRAINING_EXAMPLES + TESTING_EXAMPLES) *SAMPLE_GAP
train_X,train_y = generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES,dtype=np.float32)))
test_X,test_y = generate_data(np.sin(np.linspace(test_start,test_end,TRAINING_EXAMPLES,dtype=np.float32)))

#调用fit函数训练模型
regressor.fit(train_X,train_y,batch_size = BATCH_SIZE,steps= TRAINING_STEP)

#调用训练好的模型对测试数据进行预测
predicted=[[pred] for pred in regressor.predict(test_X)]
#计算rmse作为评价标准
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print ("Mean Square Error is : %f" %rmse[0])



fig = plt.figure()
plot_predicted = plt.plot(predicted, label = 'predicted')
plot_test = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
#fig.savefig('sin.png')
plt.show();