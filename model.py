import handle_data
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout
from keras.layers import GRU
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
import random
from scipy import stats
import os

class LossHistory(keras.callbacks.Callback):
    '''
    绘制损失函数图像
    '''
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def loss_plot(self):
        dict_data = {
            'loss': self.losses
        }
        data_pd = pd.DataFrame(dict_data)
        data_pd.plot()
        plt.plot(data_pd[['loss']])
        plt.show()

def Strategy(c_price, p_price, err):
    '''
    筛选股票的策略
    :param c_price:今日的收盘价
    :param p_price: 未来预测的价格
    :param err:测试集的平均误差值
    :return:percent
    '''
    print(c_price)
    up_price = p_price + err
    low_price = p_price - err
    if c_price<low_price:
        print("预测的价格高于今日收盘价，可做多")
        return 1
    if c_price>up_price:
        print("预测的价格低于今日收盘价，可做空")
        return 0
    precent = (up_price - c_price)/(2*err)
    print("预测的价格比今日收盘价高可能性为")
    print(precent)
    return precent

def Model_Create(input_dim, input_length):
    '''
    :param input_dim: 单个样本的特征值维度
    :param input_length: 输入的时间点长度
    :return:
    '''
    model = Sequential()
    model.add(Conv1D(filters=64, input_shape=(input_length, input_dim), kernel_size=1, activation='relu'))
    model.add(Dropout(0.2))
    model.add(GRU(units=128))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    print(model.summary())
    return model

def Multi_stocktrain(num, file_name, days, time_stamp, model, epochs, batch_sizes):
    '''
    采用随机的方法来选取不同的股票来对同一模型进行训练
    :param num:     #选取额外的股票数据作为训练集
    :param file_name:
    :param days:预测未来几天内的收盘价的平均值
    :param timestamps:
    :param model:模型
    :param epochs:
    :param batch_sizes:
    :return:  model 返回训练好的模型
    '''
    files = os.listdir('stock_data')
    files.remove(file_name)
    stock_list = random.sample(range(1, len(files)), num)
    for i in stock_list:
        print(files[i])
        file_path = "stock_data\\" + files[i]
        data = handle_data.load_data(file_path)
        data = handle_data.K_mean(data, days)

        scaler = MinMaxScaler(feature_range=(0, 1))
        train = scaler.fit_transform(data)

        x_train, y_train = [], []
        for i in range(time_stamp, len(train) - days + 1):
            x_train.append(train[i - time_stamp:i])
            y_train.append(train[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_sizes, verbose=1)
        return model

def Modle_preday(file_name, days, time_stamp, division):
    '''
    :param file_name: 对应的是股票数据的文件名
    :param days: 预测未来几天内的收盘价的平均值
    :param time_stamp: 一共使用前多少天的数据进行预测
    :param division: 把数据分成测试集和训练集的比例
    :return:
    '''
    #提取已经储存好的股票的历史数据
    file_path = "stock_data\\" + file_name
    data = handle_data.load_data(file_path)
    data = handle_data.K_mean(data, days)

    # 划分训练集以及测试集
    divide = division * data.shape[0]
    train = data[data.index <= divide]
    test = data[data.index > divide]

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    s_train = scaler.fit_transform(train)
    s_test = scaler.fit_transform(test)

    # 测试集与训练集
    x_train, y_train = [], []
    for i in range(time_stamp, len(train)-days+1):
        x_train.append(s_train[i - time_stamp:i])
        y_train.append(s_train[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_test, y_test = [], []
    for i in range(time_stamp, len(s_test)-days+1):
        x_test.append(s_test[i - time_stamp: i])
        y_test.append(test.iloc[i, 0])
    x_test = np.array(x_test)

    history = LossHistory()

    # 创建模型
    epochs = 10
    batch_size = 16
    #model = Model_Lstm(x_train.shape[-1], x_train.shape[1])
    model = Model_Create(x_train.shape[-1], x_train.shape[1])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[history])  # 训练网络

    #Multi_stocktrain(2, file_name, days, time_stamp, model, 3, batch_size)

    #保存模型
    model_file = "model_save\\" + file_name.strip('.csv') + ".h5"
    model.save(model_file)

    #使用测试集进行预测
    predict_price = model.predict(x_test)
    scaler.fit_transform(pd.DataFrame(test['close'].values))
    predict_price = scaler.inverse_transform(predict_price)

    # 模型效果指标
    pre = predict_price.reshape(1, -1)[0]
    pre = pre.tolist()
    handle_data.calPerformance(y_test, pre)

    #计算预测误差绝对值的平均数
    sum = 0
    for i in range(0, len(y_test)):
        sum += abs(y_test[i]-pre[i])
    err = sum/len(y_test)
    print(err)

    # 图像展示
    dict_data = {
        'pre': pre,
        'close': y_test
    }
    data_pd = pd.DataFrame(dict_data)
    data_pd.plot()
    plt.plot(data_pd[['pre', 'close']])
    plt.show()

    #损失值的图像展示
    history.loss_plot()

   #未来三天的预测值
    Strategy(test.iloc[-1, 4], pre[-1], err)
    #print(pre[-1])

    #把预测数据和目标值存入文件
    predata_file = 'pre_data\\' + file_name
    data_pd.to_csv(predata_file)

#handle_data.stock_info(20)
#Modle_preday('000001.SZ.csv', 3, 50, 2/3)
# for root, dirs, files in os.walk('stock_data'):
# print(files)