import csv
import pickle
import time

import ewtpy
import numpy as np
import pywt
from neurokit2 import ppg_process
from scipy.signal import decimate, butter, filtfilt, lfilter, medfilt, firwin
from utils.plot.lineChart import *
from filepaths import Folder
from utils import *

from process.Signal import Signal
from utils.preprocess.HRextraction import PPG2HR


class preprocess:
    def __init__(self,pkl=False):

        self.signal = None

        if pkl:
            self.readPkl()
        else:
            self.readCsv()
        linePlot(x=self.signal.timestamp,y=self.signal.data)

    def readCsv(self):
        # 从给定的文件中读取数据并处理
        timestamp = []
        data = []
        with open(csv_filepath,'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                timestamp.append(float(row['Timestamp']))
                data.append(float(row[item_name]))
        time_begin = timestamp[0]
        self.signal = Signal(item_name=item_name,
                             timestamp=np.array(timestamp) - time_begin,
                             data=np.array(data))

        WriteFile.pkl(self.signal,pkl_filepath)
    def readPkl(self):
        with open(pkl_filepath,'rb') as file:
            self.signal = pickle.load(file)


    def filter(self,param:dict):
        """
        type:bandpass bandstop FIR medianFilter 带通 带阻 FIR　中值
        """
        #time = None
        data = None

        btype = param['type']
        if btype == 'bandpass' or btype == 'bandstop':
            nyq = 0.5 * self.signal.sample_rate
            low = param['lowcut'] / nyq
            high = param['highcut'] / nyq
            order = param['order'] if 'order' in param else 5
            b, a = butter(order, [low, high], btype=btype)

            data = filtfilt(b, a, self.signal.data)
            #self.time_align()
        elif btype == 'low':
            # 设计Butterworth低通滤波器
            high = param['highcut']  # 截止频率，单位：Hz
            nyq = 0.5 * self.signal.sample_rate
            norm_cutoff = high / nyq  # 归一化截止频率
            order = param['order'] if 'order' in param else 4 # 滤波器阶数
            b, a = butter(order, norm_cutoff, btype='low', analog=False)
            data = filtfilt(b, a, self.signal.data)

        elif btype == 'FIR':
            nyq = 0.5 * self.signal.sample_rate
            cut = param['cut'] if 'cut' in param else 4 #默认是１０？
            order = param['order'] if 'order' in param else 45 # 滤波器阶数
            fir_coeff = firwin(order, cut / nyq)  # 设计FIR滤波器

            data = lfilter(fir_coeff, 1.0, self.signal.data)  # 应用FIR滤波器
        elif btype == 'med':
            # 中值滤波
            kernel_size = param['kernel'] if 'kernel' in param else 25  # 中值滤波器的大小

            data = medfilt(self.signal.data, kernel_size=kernel_size)  # 应用中值滤波


        print("======滤波：完成======")
        linePlot(x=self.signal.timestamp,y=data)
        self.signal.data = data
        assert len(data) == len(self.signal.timestamp)

    def least_squares_fitting(self,order=1):
        # 拟合基线漂移
        # order拟合多项式的阶数
        p = np.polyfit(self.signal.timestamp, self.signal.data, order)  # 最小二乘拟合
        baseline_estimated = np.polyval(p, self.signal.timestamp)  # 拟合的基线漂移信号

        # 去除基线漂移
        data = self.signal.data - baseline_estimated
        linePlot(x=self.signal.timestamp, y=data)
        self.signal.data = data

        print("======消除基线漂移：完成======")
    def range_outliers(self,normal_range):
        # 筛选异常值，对异常值进行线性插值补全

        condition = np.logical_or(self.signal.data < normal_range[0], self.signal.data > normal_range[1])
        outliers_idx = np.where(condition)[0]
        return outliers_idx

    def grad_outliers(self,threshold):
        # 计算梯度/差分
        gradient = np.diff(self.signal.data)

        # 定义异常阈值，这里我们使用均值加上2倍的标准差
        #threshold = np.mean(gradient) + 2 * np.std(gradient)

        # 找到异常梯度的索引
        outliers_idx = np.where(abs(gradient) > threshold)[0] + 1  # 加1是因为差分后索引向前移动了一位

        return outliers_idx
    def linear_interpolate(self, outliers_idx, normal_range):
        # 线性插值补全
        """
        找到最近的两个正常值锚点，以此为端点获得直线，使用直线上的点替换原数据
        """
        data = self.signal.data.copy()

        # 边界情况处理
        if 0 in outliers_idx:
            data[0] = normal_range[0] if abs(data[0] - normal_range[0]) < abs(data[0] - normal_range[1]) else normal_range[1]
        if len(data) - 1 in outliers_idx:
            data[-1] = normal_range[0] if abs(data[-1] - normal_range[0]) < abs(data[-1] - normal_range[1]) else normal_range[1]

        i = 0
        while i < len(outliers_idx):
            start = i
            # 查找连续异常值的范围
            while i + 1 < len(outliers_idx) and outliers_idx[i + 1] == outliers_idx[i] + 1:
                i += 1
            end = i

            # 找到连续异常值序列的起始和终点索引
            start_idx = outliers_idx[start] - 1
            end_idx = outliers_idx[end] + 1

            # 处理边界条件
            if start_idx < 0:
                start_idx = 0
            if end_idx >= len(data):
                end_idx = len(data) - 1

            # 确保前后索引有效
            if start_idx < end_idx:
                # 计算斜率
                slope = (data[end_idx] - data[start_idx]) / (end_idx - start_idx)
                # 为连续的异常值赋予线性变化的值
                for j in range(start, end + 1):
                    idx = outliers_idx[j]
                    data[idx] = data[start_idx] + slope * (idx - start_idx)

            i += 1
        linePlot(x=self.signal.timestamp, y=data)
        self.signal.data = data
    def remove_eye_noise(self):
        # create an ewt parameter class instance
        """params = ewtpy.utilities.ewt_params()

        # modify the parameters as you wish
        params.N = 5 # number of EMFs
        params.log = 1 # use log scale
        params.detect = 'locmax' # use local maxima method for boundary detection
        """
        # perform EWT on EEG signal
        start_time = time.time()
        end_time = time.time()
        emfs, mfb, boundaries = ewtpy.EWT1D(self.signal.data, N=12) # N取值 todo
        end_time = time.time()
        print('----cost1:',end_time - start_time)
        start_time = end_time
        # choose sym7 wavelet as the best wavelet function
        wavelet = 'sym7'

        # decompose each EMF into approximation and detail coefficients using wavelet transform
        coeffs = [pywt.wavedec(emf, wavelet, level=1) for emf in emfs] #参数
        end_time = time.time()
        print('----cost2:', end_time - start_time)
        start_time = end_time

        # keep only 7 levels of approximation coefficients and 5 levels of detail coefficients for each EMF
        coeffs_new = [c[:8] + [None] * (len(c) - 8) for c in coeffs]
        coeffs_new = [c[:1] + c[-6:] for c in coeffs_new]
        end_time = time.time()
        print('----cost3:', end_time - start_time)
        start_time = end_time

        # reconstruct each EMF from the modified coefficients using inverse wavelet transform
        emfs_new = [pywt.waverec(c, wavelet) for c in coeffs_new]
        end_time = time.time()
        print('----cost4:', end_time - start_time)
        start_time = end_time

        # set the default parameters for ApEn: m=2, r=0.2*std
        m = 2  # embedding dimension
        r = 0.2 * np.std(self.signal.data)  # tolerance (fraction of std)

        # compute ApEn for each EMF
        apens = [app_entropy(emf,m=3) for emf in emfs_new]
        end_time = time.time()
        print('----cost5:', end_time - start_time)
        start_time = end_time

        # set the threshold for EOG artifact detection: 0.45
        threshold = 0.45  # EOG近似熵判定阈值

        # identify and remove the EMFs with ApEn higher than the threshold
        emfs_clean = [emf for emf, apen in zip(emfs_new, apens) if apen < threshold]
        end_time = time.time()
        print('----cost6:', end_time - start_time)
        start_time = end_time

        # reconstruct the EEG signal from the clean EMFs
        data = np.sum(emfs_clean, axis=1)
        end_time = time.time()
        print('----cost7:', end_time - start_time)
        #start_time = end_time

        self.signal.data = data
        self.signal.time_align()
        linePlot(x=self.signal.timestamp, y=self.signal.data)
        print("======消除眼电伪迹：完成======")
i = 0
def app_entropy(U, m=2):
    global  i
    """Compute the approximate entropy of a given time series.

    Parameters
    ----------
    U : list
        Time series
    m : int, optional 嵌入维度，用于构建相空间的数据子集的长度
        Length of compared run of data (default: 2) 
    r : float, optional 相似度阈值，用于定义两个序列段是否相似
        Filtering level (default: 0.2 * std(U))

    Returns
    -------
    float
        Approximate entropy
    """
    """
     def _phi(m):
        x = np.array([U[i:i + m] for i in range(len(U) - m + 1)]) # list of compared run of data
        C = np.sum(np.max(np.abs(x[:, np.newaxis] - x[np.newaxis, :]), axis=2) <= r, axis=0) / (len(U) - m + 1)
        return np.sum(np.log(C)) / (len(U) - m + 1)

    return _phi(m) - _phi(m + 1)

    
    """
    begin_time = time.time()
    r = 0.2 * np.std(U)

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    result = abs(_phi(m + 1) - _phi(m))
    end_time = time.time()
    i += 1
    print('---|||cost: ',end_time - begin_time,' ---',i)
    return result


csv_filepath = 'GSR_Shimmer_28.csv'
pkl_filepath = 'example.pkl'

item_name = 'PPG_mv'

if __name__ == '__main__':
    if item_name == 'GSR_ohm':
        agent = preprocess()

        agent.signal.data = 1000 / agent.signal.data
        #agent.signal = agent.signal.slice_from_tail(window=500)
        agent.signal.timestamp -= agent.signal.timestamp[0]

        linePlot(x=agent.signal.timestamp, y=agent.signal.data)

        agent.filter(param={
            'type':'low',
            'highcut' : 0.2,
            'order':4
        })
        idx = agent.range_outliers(normal_range=(0.05,60))
        agent.linear_interpolate(idx,normal_range=(0.05,60))
        idx2 = agent.grad_outliers(threshold=8)
        agent.linear_interpolate(idx2,normal_range=(0.05,60))
        agent.least_squares_fitting()
    elif item_name == 'rawValue':
        agent = preprocess()
        agent.signal = agent.signal.slice(window=60) #default : startAt0=False
        agent.signal.timestamp -= agent.signal.timestamp[0]
        linePlot(x=agent.signal.timestamp, y=agent.signal.data)
        agent.filter(param={
            'type':'bandpass',
            'lowcut' : 0.5, 'highcut' : 49.75
        })
        agent.filter(param={
            'type': 'low',
            'highcut': 49.75
        })
        Q=30 # 带通滤波器的质量因子
        f1 = 50 / (1.0 + 1.0 / (2.0 * Q))
        f2 = 50 * (1.0 + 1.0 / (2.0 * Q))
        agent.filter(param={
            'type':'bandstop',
            'lowcut': f1, 'highcut': f2
        })
        agent.remove_eye_noise()

        agent.least_squares_fitting()
        idx = agent.range_outliers(normal_range=(-100,100))
        agent.linear_interpolate(idx,normal_range=(-100,100))

    elif item_name == 'PPG_mv':
        agent = preprocess()
        agent.signal.data = agent.signal.data / 1000 # mv -> V
        #linePlot(x=agent.signal.timestamp, y=agent.signal.data)
        #agent.signal = agent.signal.slice_from_head(window=400)
        agent.signal = agent.signal.slice(begin=300,end=400)
        agent.signal.timestamp -= agent.signal.timestamp[0]
        linePlot(x=agent.signal.timestamp, y=agent.signal.data)
        #idx = agent.range_outliers(normal_range=(-100, 100))
        #agent.linear_interpolate(idx, normal_range=(-100, 100))

        agent.filter(param={
            'type': 'med',
            'order':15
        })
        """
         agent.filter(param={
            'type': 'FIR',
            'kernel': 1
        }) # 消除基线偏移
        
        """



        agent.least_squares_fitting()
        """
          Q = 30  # 带通滤波器的质量因子
        f1 = 60 / (1.0 + 1.0 / (2.0 * Q))
        f2 = 60 * (1.0 + 1.0 / (2.0 * Q))
        agent.filter(param={
            'type': 'bandstop',
            'lowcut': f1, 'highcut': f2
        })
        """

        RRI,HR = PPG2HR(ppg=agent.signal)
        linePlot(x=RRI.timestamp, y=RRI.data)
        linePlot(x=HR.timestamp, y=HR.data)

        agent.signal = HR

        #idx = agent.grad_outliers(threshold=1) #参数需谨慎
        #agent.linear_interpolate(idx, normal_range=(70, 150))

        idx = agent.range_outliers(normal_range=(70, 150))
        agent.linear_interpolate(idx, normal_range=(70, 150))

        linePlot(x=agent.signal.timestamp, y=agent.signal.data,color='r')
        """
        agent.filter(param={
            'type': 'med',
            'kernel': 5
        })
        """







