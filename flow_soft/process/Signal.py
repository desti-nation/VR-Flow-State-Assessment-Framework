import time
from collections import namedtuple

#from  EMD import EMD
from sklearn.decomposition import FastICA
from sklearn.neighbors import LocalOutlierFactor

from utils import *
import ewtpy
import numpy as np
import pywt
from scipy.signal import decimate, butter, filtfilt, lfilter, firwin, medfilt, wiener

from utils.plot.lineChart import *

PROCESS_PARA = namedtuple('Process_Para', [
    'down_sample_rate',
    'lowcut',
    'highcut',
    'remove_outlier_method',
    'standarize_method'
])


class Signal:
    def __init__(self, item_name, data: np.ndarray, timestamp: np.ndarray):
        self.item_name = item_name

        if len(data) == 0:
            self.data = np.array([0])
            self.timestamp = np.array([0])
            return

        self.data = data
        self.timestamp = timestamp
        self.sample_rate = self.calc_sample_rate()

        self.debug = True

    def calc_sample_rate(self):
        intervals = np.diff(self.timestamp)
        mean = np.mean(intervals)
        return 1 / mean

    def update(self):
        self.sample_rate = self.calc_sample_rate()

    def len(self):
        return len(self.timestamp)

    def slice(self, begin=None, window=None, head=False, tail=True, startAt0=False):
        assert window is not None
        if begin is not None:
            time_begin, time_end = min(begin, self.timestamp[-1]), min(begin + window, self.timestamp[-1])
        else:
            assert head != tail
            if head:
                time_begin, time_end = self.timestamp[0], min(self.timestamp[0] + window, self.timestamp[-1])
            else:
                time_begin, time_end = max(self.timestamp[-1] - window, self.timestamp[0]), self.timestamp[-1]

        condition = np.where((self.timestamp >= time_begin) & (self.timestamp <= time_end))
        new_time, new_data = self.timestamp[condition], self.data[condition]
        if startAt0:
            new_time -= new_time[0]
        return Signal(item_name=self.item_name, timestamp=new_time, data=new_data)

    def resample(self, new_sample_rate):
        # time_span = time_chip[1] - time_chip[0]
        # assert abs(time_span - (self.timestamp[-1]-self.timestamp[0])) < 2
        # if abs(self.sample_rate - new_sample_rate) < 1.5:
        #     self.sample_rate = new_sample_rate
        #     return
        
        # if self.sample_rate < new_sample_rate:
        #     self.data = Signal.interpolate(self.data, new_sample_rate * time_span)
        # else:
        #     from scipy.signal import resample as sci_resample
        #     # 降采样
        #     self.data = sci_resample(self.data, int(time_span * new_sample_rate))
        
        # self.data = Signal.resample(self.data, new_sample_rate * time_span)
        # self.timestamp = np.linspace(time_chip[0], time_chip[1], len(self.data))
        
        
        cur_time = self.timestamp[0]
        new_data,new_time = [],[]
        while cur_time + 1 < self.timestamp[-1]:
            condition = np.logical_and(cur_time <= self.timestamp, self.timestamp < cur_time + 1)

            real_begin =  self.timestamp[np.where(condition)][0]
            real_end = self.timestamp[np.where(condition)][-1]
            try:
                assert abs(real_end - real_begin - 1) < 0.2
            except:
                print(real_end - real_begin - 1)
            new_data.append( Signal._resample(data=self.data[np.where(condition)],target_length=new_sample_rate,blk=('openness' in self.item_name) ))
            new_time.append( np.linspace(real_begin, real_end , len(new_data)) )

            assert len(new_data[-1]) == new_sample_rate,str(len(new_data[-1])) + ' ' + str(new_sample_rate)
            cur_time += 1

        self.data,self.timestamp = np.concatenate(new_data),np.concatenate(new_time)
        self.sample_rate = new_sample_rate

        assert len(self.data) % new_sample_rate == 0, "data length must be multiple of sample rate," + str(len(self.data))+ " " + str(new_sample_rate)

    
    # @staticmethod
    # def blink_expand(data, fs):
    #     # 创建结果数组
    #     interpolated_array = np.zeros(fs).astype(int)
    #
    #     # 计算每个原始点应出现在新数组中的位置
    #     indices = np.round(np.linspace(0, fs - 1, num=len(data))).astype(int)
    #
    #     last = 0
    #     status = 1
    #     # 将原始数据点放入计算出的位置
    #     for orig_idx, new_idx in enumerate(indices):
    #         interpolated_array[last:new_idx] = status | data[orig_idx]
    #         last = new_idx + 1
    #         status = data[orig_idx]
    #         interpolated_array[new_idx] = data[orig_idx]
    #     if last < fs:
    #         interpolated_array[last:] = status
    #
    #     return interpolated_array
    @staticmethod
    def _resample(data,target_length,blk=False):
        target_length = int(target_length)
        if len(data) < target_length:
            if blk:
                return Signal.blk_interpolate(data, target_length)
            else:
                return Signal.interpolate(data, target_length)
        elif len(data) > target_length:
            # from scipy.signal import resample as sci_resample
            # # 降采样

            # return sci_resample(data, target_length)
            return Signal.downsample(data, target_length)
        else:
            return data

    @staticmethod
    def interpolate(data, target_length):
        target_length = int(target_length)
        def linear_interpolate(orig_array, target_length):
            x = len(orig_array)  # 原始数组的长度
            y = target_length  # 目标数组的长度

            # 创建结果数组
            interpolated_array = np.zeros(y)


            # 计算每个原始点应出现在新数组中的位置
            indices = np.round(np.linspace(0, y - 1, num=x)).astype(int)

            # 将原始数据点放入计算出的位置
            for orig_idx, new_idx in enumerate(indices):
                interpolated_array[new_idx] = orig_array[orig_idx]

            # 对于新数组中的每个位置，如果它还没有被赋值，则进行线性插值
            for i in range(1, y):
                if interpolated_array[i] == 0:
                    # 找到插值的左右边界
                    left = i - 1
                    while interpolated_array[left] == 0 and left > 0:
                        left -= 1
                    right = i + 1
                    while right < y and interpolated_array[right] == 0:
                        right += 1

                    # 执行线性插值
                    if right < y and interpolated_array[left] != 0 and interpolated_array[right] != 0:
                        interpolated_array[i] = (interpolated_array[left] +
                                                 ((interpolated_array[right] - interpolated_array[left]) *
                                                  ((i - left) / (right - left))))

            return interpolated_array

        assert len(data.shape) <= 2
        if len(data.shape) == 1:
            return linear_interpolate(data, target_length)
        else:
            return np.array([Signal.interpolate(data[i], target_length) for i in range(data.shape[0])])

    # @staticmethod
    # def downsample(data,target_length):
    #     target_length = int(target_length)
    #     old_length = len(data)

    #     step = int(np.ceil(old_length / target_length))
    #     indices = np.arange(0, old_length, step)
    #     return data[indices]

    @staticmethod
    def downsample(data,target_length):
        old_length = len(data)
        step = old_length / target_length
        new_x = np.zeros(target_length)
        for i in range(target_length):
            index = i * step
            weight = np.cos((np.arange(old_length) - index) * np.pi / old_length)
            new_x[i] = np.sum(data * weight) / np.sum(weight)
        return new_x

    @staticmethod
    def blk_interpolate(data, target_length):
        orig_array = data

        x = len(orig_array)  # 原始数组的长度
        y = int(target_length)  # 目标数组的长度

        # 创建结果数组
        interpolated_array = np.zeros(y)

        # 计算每个原始点应出现在新数组中的位置
        indices = np.round(np.linspace(0, y - 1, num=x)).astype(int)

        # 将原始数据点放入计算出的位置
        last_index = None # 上一个为1的点
        for orig_idx, new_idx in enumerate(indices):
            # if orig_array[orig_idx] == 1 and last_index is not None and interpolated_array[last_index] == 1: 
            #     # 形成闭眼区间
            #     for i in range(last_index + 1, new_idx):
            #         interpolated_array[i] = 1
            interpolated_array[new_idx] = orig_array[orig_idx]
            last_index = new_idx

        # 两个1之间设置成0 


        return interpolated_array

    # 带通滤波

    """
    def filter(self, lowcut, highcut, btype, order=5):
        ""#"
        带通阻波器
        :param order:
        :param lowcut: 最低频率
        :param highcut: 最高频率
        :return:
        ""#"
        # 滤波
        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype=btype)
        self.data = filtfilt(b, a, self.data)
        self.time_align()

        self.update()
        self.print("======滤波：完成======")
    """

    def filter(self, btype, lowcut=None, highcut=None, cutoff=None, order=5, kernel=25):
        """
        type:bandpass bandstop FIR medianFilter 带通 带阻 FIR　中值
        """
        data = None
        nyq = 0.5 * self.sample_rate
        if btype == 'bandpass' or btype == 'bandstop':
            b, a = butter(order, [lowcut / nyq, highcut / nyq], btype=btype)
            data = filtfilt(b, a, self.data)
        elif btype == 'lowpass' or btype == 'highpass':
            # 设计Butterworth低通滤波器
            b, a = butter(order, cutoff / nyq, btype=btype, analog=False)
            data = filtfilt(b, a, self.data)
        elif btype == 'lowstop':
            b, a = butter(order, cutoff / nyq, btype='lowpass', analog=False)
            data = self.data - filtfilt(b, a, self.data)
        elif btype == 'FIR':
            fir_coeff = firwin(order, cutoff / nyq)  # 设计FIR滤波器
            data = lfilter(fir_coeff, 1.0, self.data)  # 应用FIR滤波器
        elif btype == 'med':
            # 中值滤波
            data = medfilt(self.data, kernel_size=kernel)  # 应用中值滤波

        self.data = data
        self.time_align()

    def range_outliers(self, normal_range):
        # 筛选异常值，对异常值进行线性插值补全
        condition = np.logical_or(self.data < normal_range[0], self.data > normal_range[1])
        outliers_idx = np.where(condition)[0]
        self.__linear_interpolate__(outliers_idx, normal_range=normal_range)

    #def IQR_outliers(self):
    def remove_outliers(self,
                        n_neighbors=200,
                        fix_window=100, remove_outliers_method='LocalOutlierFactor'):
        assert remove_outliers_method in ['IQR','LocalOutlierFactor']
        outliers_idx = None

        if remove_outliers_method == 'IQR':
            q25 = np.percentile(self.data, 25)
            q75 = np.percentile(self.data, 75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outliers_idx = np.where((self.data < lower_bound) | (self.data > upper_bound))[0]
        if remove_outliers_method == 'LocalOutlierFactor':
            data_reshape = self.data.reshape(-1, 1)
            # 创建Local Outlier Factor模型

            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            # 训练模型并预测离群点
            outliers_score = lof.fit_predict(data_reshape)
            outliers_idx = np.where(outliers_score == -1)[0]
        if remove_outliers_method == '':
            outliers_idx = np.array([])

        # 对异常值取中位数进行填补
        delete_idx = []
        for idx in outliers_idx:
            valid_idx = []
            for _ in range(max(0, idx - 2 * fix_window), idx):
                # 向前查找两倍的修补窗口
                if _ not in outliers_idx:
                    valid_idx.append(_)
            valid_idx = np.array(valid_idx)

            if len(valid_idx) == 0:
                # 如果前面没有可用的有效数据，直接删除当前这个元素
                # delete
                # np.delete(self.timestamp, idx)
                # np.delete(self.data, idx)
                delete_idx.append(idx)
                continue

            valid_data = self.data[valid_idx]
            if len(valid_data) == 1:
                self.data[idx] = valid_data[0]
            else:
                self.data[idx] = np.median(valid_data)

        keep_indices = np.ones_like(self.data, dtype=bool)
        keep_indices[delete_idx] = False

        # 使用布尔索引选择要保留的元素
        self.timestamp = self.timestamp[keep_indices]
        self.data = self.data[keep_indices]

        self.time_align()
        self.update()

    def IQR_outliers(self,ratio=1.5):
        q25 = np.percentile(self.data, 25)
        q75 = np.percentile(self.data, 75)
        iqr = q75 - q25
        lower_bound = q25 - ratio * iqr
        upper_bound = q75 + ratio * iqr
        outliers_idx = np.where((self.data < lower_bound) | (self.data > upper_bound))[0]

        normal_range = (lower_bound,upper_bound)
        self.__linear_interpolate__(outliers_idx, normal_range=normal_range)


    def grad_outliers(self, threshold, normal_range):
        # 计算梯度/差分
        gradient = np.diff(self.data)

        # 定义异常阈值，这里我们使用均值加上2倍的标准差
        # threshold = np.mean(gradient) + 2 * np.std(gradient)

        # 找到异常梯度的索引
        outliers_idx = np.where(abs(gradient) > threshold)[0] + 1  # 加1是因为差分后索引向前移动了一位
        print('异常值数量：',len(outliers_idx))
        self.__linear_interpolate__(outliers_idx, normal_range=normal_range)

    def __linear_interpolate__(self, outliers_idx, normal_range):
        # 线性插值补全
        """
        找到最近的两个正常值锚点，以此为端点获得直线，使用直线上的点替换原数据
        """

        # 边界情况处理
        if 0 in outliers_idx:
            self.data[0] = normal_range[0] \
                if abs(self.data[0] - normal_range[0]) < abs(self.data[0] - normal_range[1]) else normal_range[1]
        if len(self.data) - 1 in outliers_idx:
            self.data[-1] = normal_range[0] \
                if abs(self.data[-1] - normal_range[0]) < abs(self.data[-1] - normal_range[1]) else normal_range[1]

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
            if end_idx >= len(self.data):
                end_idx = len(self.data) - 1

            # 确保前后索引有效
            if start_idx < end_idx:
                # 计算斜率
                slope = (self.data[end_idx] - self.data[start_idx]) / (end_idx - start_idx)
                # 为连续的异常值赋予线性变化的值
                for j in range(start, end + 1):
                    idx = outliers_idx[j]
                    self.data[idx] = self.data[start_idx] + slope * (idx - start_idx)

            i += 1

    def time_align(self):
        # 时间戳对齐
        if len(self.timestamp) < len(self.data):
            self.timestamp = np.linspace(self.timestamp[0], self.timestamp[-1], len(self.data))
        elif len(self.timestamp) > len(self.data):
            self.timestamp = self.timestamp[:len(self.data)]

    def eeg_preprocess(self):
        print('=====EEG 处理开始=====')
        linePlot(self.timestamp, self.data)


        step5 = '[EEG][detrend]'
        _ = Record(step5)
        self.least_squares_fitting()
        _.end()

        step1 = '[EEG][bandpass]'
        _ = Record(step1)
        self.filter(btype='bandpass', lowcut=0.5, highcut=49.75, order=5)
        _.end()


        step3 = '[EEG][bandstop]'
        _ = Record(step3)
        Q = 30  # 带通滤波器的质量因子
        f1 = 50 / (1.0 + 1.0 / (2.0 * Q))
        f2 = 50 * (1.0 + 1.0 / (2.0 * Q))
        self.filter(btype='bandstop', lowcut=f1, highcut=f2, order=6)
        linePlot(self.timestamp, self.data)
        _.end()

        step6 = '[EEG][remove outliers]'
        _ = Record(step6)
        self.range_outliers(normal_range=(-300, 300))
        linePlot(self.timestamp, self.data)
        # self.remove_outliers(n_neighbors=20,fix_window=5,remove_outliers_method='IQR')
        _.end()

        step4 = '[EEG][EOG remove]'
        _ = Record(step4)
        self.EOG_denoise()
        _.end()



        print('=====EEG 处理结束=====')
        linePlot(self.timestamp, self.data)

    def gsr_preprocess(self):
        print('=====GSR 处理开始=====')
        linePlot(self.timestamp, self.data)

        step1 = '[GSR][lowpass]'
        _ = Record(step1)
        self.filter(btype='lowpass', cutoff=0.2, order=4) #  todo 
        _.end()

        step2 = '[GSR][remove outliers]'
        _ = Record(step2)
        self.grad_outliers(threshold=8, normal_range=(0.05, 60))
        self.range_outliers(normal_range=(0.05, 40))
        _.end()

        step3 = '[GSR][detrend]'
        _ = Record(step3)
        #self.least_squares_fitting()
        _.end()

        print('=====GSR 处理结束=====')
        linePlot(self.timestamp, self.data)

    def ppg_preprocess(self):
        print('=====PPG 处理开始=====')
        linePlot(self.timestamp, self.data)

        step1 = '[PPG][MEDfilter]'
        _ = Record(step1)
        self.filter(btype='med', kernel=25)
        linePlot(self.timestamp, self.data)
        _.end()

        step2 = '[PPG][lowstop]'
        _ = Record(step2)
        self.filter(btype='lowstop', cutoff=2, order=2)
        linePlot(self.timestamp, self.data)
        _.end()

        step3 = '[PPG][detrend]'
        _ = Record(step3)
        self.least_squares_fitting()
        _.end()

        step4 = '[PPG][remove_outliers]'
        _ = Record(step4)
        self.IQR_outliers(ratio=10)
        #self.range_outliers(normal_range=(-0.5,0.5))
        _.end()

        print('=====PPG 处理结束=====')
        linePlot(self.timestamp, self.data)

    def rri_preprocess(self):
        print('=====RRI 处理开始=====')
        linePlot(self.timestamp, self.data)

        step1 = '[RRI][remove outliers]'
        _ = Record(step1)
        self.filter(btype='med', kernel=25)
        _.end()

        step2 = '[RRI][remove outliers]'
        _ = Record(step2)
        self.IQR_outliers()
        #self.remove_outliers(n_neighbors=50,
        #                fix_window=10,remove_outliers_method='IQR') # todo 2
        self.grad_outliers(threshold=10 / 60, normal_range=(60 / 150, 60 / 50))

        linePlot(self.timestamp, self.data)
        #self.range_outliers(normal_range=(60 / 150,60 / 50))
        _.end()

        print('=====RRI 处理结束=====')
        linePlot(self.timestamp, self.data)

    def least_squares_fitting(self, order=1):
        # 拟合基线漂移
        p = np.polyfit(self.timestamp, self.data, order)  # 最小二乘拟合
        baseline_estimated = np.polyval(p, self.timestamp)  # 拟合的基线漂移信号

        # 去除基线漂移
        self.data -= baseline_estimated

    def remove_eye_noise(self):
        # create an ewt parameter class instance
        """params = ewtpy.utilities.ewt_params()

        # modify the parameters as you wish
        params.N = 5 # number of EMFs
        params.log = 1 # use log scale
        params.detect = 'locmax' # use local maxima method for boundary detection
        """

        # perform EWT on EEG signal
        emfs, mfb, boundaries = ewtpy.EWT1D(self.data, N=12)

        # choose sym7 wavelet as the best wavelet function
        wavelet = 'sym7'

        # decompose each EMF into approximation and detail coefficients using wavelet transform
        coeffs = [pywt.wavedec(emf, wavelet, level=3) for emf in emfs]

        # keep only 7 levels of approximation coefficients and 5 levels of detail coefficients for each EMF
        coeffs_new = [c[:8] + [None] * (len(c) - 8) for c in coeffs]
        coeffs_new = [c[:1] + c[-6:] for c in coeffs_new]

        # reconstruct each EMF from the modified coefficients using inverse wavelet transform
        emfs_new = [pywt.waverec(c, wavelet) for c in coeffs_new]

        # set the default parameters for ApEn: m=2, r=0.2*std
        m = 2  # embedding dimension
        r = 0.2 * np.std(self.data)  # tolerance (fraction of std)

        # compute ApEn for each EMF
        apens = [app_entropy(emf) for emf in emfs_new]

        # set the threshold for EOG artifact detection: 0.45
        threshold = 0.45  # EOG近似熵判定阈值

        # identify and remove the EMFs with ApEn higher than the threshold
        emfs_clean = [emf for emf, apen in zip(emfs_new, apens) if apen < threshold]

        # reconstruct the EEG signal from the clean EMFs
        self.data = np.sum(emfs_clean, axis=1)
        self.time_align()


    def remove_eye_noise2(self,wavelet='db4',noise_sigma=1):
        # 进行小波分解
        coeffs = pywt.wavedec(self.data, wavelet, mode='per')

        # 高斯噪声的阈值
        threshold = noise_sigma * np.sqrt(2 * np.log(len(self.data)))

        # 小波系数去噪
        new_coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))

        # 重构信号
        denoised = pywt.waverec(new_coeffs, wavelet, mode='per')
        self.data = denoised
        self.time_align()

    def EOG_denoise(self):
        #if method == 'wavelet':

        #if method == 'EMD':
        #    # 经验模态分解
        #    imfs = EMD(self.data)
        """
                if method == 'ICA':
            #
            ica = FastICA(n_components=self.data.shape[1],random_state=0)
            ica_component = ica.fit_transform(self.data)
            noise_indices = []
            denoised_components = np.delete(ica_component,noise_indices,axis=1)
            self.data = ica.inverse_transform(denoised_components)
        """

        #if method == 'wiener':

        self.data = wiener(self.data,30)

        # 小波变换
        wavelet = 'db4'
        coeffs = pywt.wavedec(self.data, wavelet=wavelet, level=5)
        thresold = np.median(np.abs(coeffs[1])) * 0.67
        new_coffs = list(coeffs)
        for i in range(1, len(coeffs)):
            new_coffs[i] = pywt.threshold(new_coffs[i], thresold, mode='soft')

        # 重构
        self.data = pywt.waverec(new_coffs, wavelet=wavelet)
        self.time_align()

def app_entropy(U, m=2):
    """Compute the approximate entropy of a given time series.

    Parameters
    ----------
    U : list
        Time series
    m : int, optional
        Length of compared run of data (default: 2)
    r : float, optional
        Filtering level (default: 0.2 * std(U))

    Returns
    -------
    float
        Approximate entropy
    """
    r = 0.2 * np.std(U)

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))
