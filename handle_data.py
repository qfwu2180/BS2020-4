import tushare as ts
import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


tushare_token = '8cde7eff54a79bc4259d46a909a0605343e4bc454faddf0e75e6b74c'

def stock_catch():
    '''
    查询当前所有正常上市交易的股票列表
    :return:
    '''
    pro = ts.pro_api(tushare_token)
    data = pro.stock_basic(exchange='', list_status='L')
    data.to_csv('stocks_info.csv')


def stock_info(num):
    '''
    按顺序获取在07年之前上市的num支股票的历史数据
    :param num:
    :return:
    '''
    pro = ts.pro_api(tushare_token)
    info = pd.read_csv('stocks_info.csv')
    data = info[info.list_date < 20170101]
    stock_num = data.ts_code
    flag = 1  #只获取一只股票
    for i in stock_num:
        if flag <= num:
            wd = pro.daily(ts_code=i)
            wd1 = pro.daily_basic(ts_code=i)
            wd.drop(['ts_code', 'pre_close', 'change', 'pct_chg'], axis=1, inplace=True)
            wd1.drop(['ts_code', 'close','turnover_rate_f', 'pe_ttm', 'ps_ttm','dv_ratio','dv_ttm','total_share', 'float_share', 'free_share','total_mv','circ_mv'], axis=1, inplace=True)
            df = pd.merge(wd, wd1, how='right', on='trade_date')
            df = df.sort_values(by='trade_date')
            df.to_csv('stock_data\\' + i + '.csv', index=False)
            flag = flag + 1

def calPerformance(y_true, y_pred):
    '''
    模型效果指标评估
    y_true：真实的数据值
    y_pred：回归模型预测的数据值
    explained_variance_score：解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
    的方差变化，值越小则说明效果越差。
    mean_absolute_error：平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度
    ，其其值越小说明拟合效果越好。
    mean_squared_error：均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的
    平方和的均值，其值越小说明拟合效果越好。
    r2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因
    变量的方差变化，值越小则说明效果越差。
    '''
    model_metrics_name=[explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
    tmp_list=[]
    for one in model_metrics_name:
        tmp_score=one(y_true,y_pred)
        tmp_list.append(tmp_score)
    print(['explained_variance_score','mean_absolute_error','mean_squared_error','r2_score'])
    print(tmp_list)
    return tmp_list

def load_data(file_path):
    '''
    加载数据并删除交易日期
    :param file_path:数据路径
    :return:data
    '''
    df = pd.read_csv(file_path)
    data = df.drop(['trade_date'], axis=1)
    #print(data.isna().sum())#判断数据中是否有nan
    return data

def K_mean(data, k): #包括今天未来一共三天的平均值
    '''
    计算未来K天的均值
    :param data:数据
    :param k: 天数
    :return: 添加后的数据
    '''
    meanprice = []
    for i in range(0, data.shape[0]):
        dayprice = data.iloc[i:i + k, 3]
        meanprice.append(dayprice.mean())
    data['k_means'] = meanprice
    df = data['k_means']
    data = data.drop('k_means', axis=1)
    data.insert(0, 'k_means', df)
    return data

def pstock_info(stock):
    '''
    获取特定的股票数据
    :param stock:
    :return:
    '''
    pro = ts.pro_api(tushare_token)
    wd = pro.daily(ts_code=stock)
    wd1 = pro.daily_basic(ts_code=stock)
    wd.drop(['ts_code', 'pre_close', 'change', 'pct_chg'], axis=1, inplace=True)
    wd1.drop(
        ['ts_code', 'close', 'turnover_rate_f', 'pe_ttm', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share',
         'free_share', 'total_mv', 'circ_mv'], axis=1, inplace=True)
    df = pd.merge(wd, wd1, how='right', on='trade_date')
    df = df.sort_values(by='trade_date')
    df.to_csv('stock_data\\' + stock + '.csv', index=False)
