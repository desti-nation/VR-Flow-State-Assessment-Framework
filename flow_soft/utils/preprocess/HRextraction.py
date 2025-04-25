import numpy as np
import scipy.signal as signal
from scipy.signal import decimate

from process.Signal import Signal
from utils.plot.lineChart import linePlot


def down_sample(factor,signal):
    data = decimate(signal.data, factor)  # 自动应用一个低通滤波器防止混叠
    timestamp = signal.timestamp[::factor]  # 按步长切片即可
    return Signal(item_name=signal.item_name,timestamp=timestamp,data=data)

def PPG2HR(ppg):
    fs = ppg.sample_rate
    nyq = 0.5 * fs
    cutoff = 1 / nyq
    # 1. 信号预处理：简单低通滤波
    b, a = signal.butter(2, cutoff, 'low')
    filtered_signal = signal.filtfilt(b, a, ppg.data)
    x = ppg.data - filtered_signal

    linePlot(x=ppg.timestamp, y=x)
    # 2. 峰值检测
    peaks, _ = signal.find_peaks(x, distance=fs * 0.6)
    #peaks, _ = signal.find_peaks(ppg.data, distance=fs * 0.6)

    # 3. 计算RRI
    RRI = np.diff(peaks) / fs
    HR = 60 / RRI  # 每分钟心跳次数

    # 4. 时间戳提取：使用每个RRI间隔的起始峰值作为时间戳
    timestamp = ppg.timestamp[peaks[:-1]]  # 排除最后一个峰值，因为它没有后续峰值来计算RRI

    return (Signal(item_name='RRI',timestamp=timestamp,data=RRI),
            Signal(item_name='HR',timestamp=timestamp,data=HR))


