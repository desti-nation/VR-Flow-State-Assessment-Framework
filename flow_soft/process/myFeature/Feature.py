import csv
import os
from collections import namedtuple

from process.Signal import Signal
import numpy as np


class Feature:
    """
    目录格式
    为拼接特征做准备
    window必须统一大小

    player
        epoch
            phase
                每一个特征一个文件夹 (统计特征也是
    """
    window = 60  # todo
    step = 0.5

    def __init__(self, signal,
                 feature_items
                 ):
        # 数据初始化
        self.signal = signal
        self.item_name = signal.item_name

        # 切分
        self.window = Feature.window
        self.step = Feature.step
        self.rolling_signals = self.rolling()

        # 获取特征
        self.feature_items = feature_items
        # self.basic_feature_items = [self.item_name + '-' + _ for _ in BasicFeature]
        self.features = None
        self.timestamp = None

        # self.calc()

    def all_feature_names(self):
        return self.feature_items  # + self.basic_feature_items

    def __calc__(self, signal, base_feature):
        pass

    def calc(self, cut, base_feature=None):
        """
                if base_feature is None:
            base_feature = {_: 0 for _ in list(self.feature_items) + list(self.basic_feature_items)}

        """

        self.features = {feature_item: [] for feature_item in self.feature_items}
        self.timestamp = []
        signal_list = self.rolling_signals if cut else [self.signal]
        """
        basic_feature = {
            self.item_name + '-' + 'mean': list(map(lambda x: np.mean(x.data), signal_list)),
            self.item_name + '-' + 'median': list(map(lambda x: np.median(x.data), signal_list)),
            self.item_name + '-' + 'std': list(map(lambda x: np.std(x.data), signal_list)),
            self.item_name + '-' + 'max': list(map(lambda x: np.max(x.data), signal_list)),
            self.item_name + '-' + 'min': list(map(lambda x: np.min(x.data), signal_list)),
            self.item_name + '-' + 'range': list(map(lambda x: np.max(x.data) - np.min(x.data), signal_list)),
        }
                for basic_item in basic_feature:
            basic_feature[basic_item] = [_ - base_feature[basic_item] for _ in basic_feature[basic_item]]
        self.features.update(basic_feature)
        """

        for signal in signal_list:
            self.timestamp.append(signal.timestamp[0])
            self.__calc__(signal, base_feature)
            """
            
            try:
                
            except Exception:
                
                for feature_item in self.feature_items:
                    self.features[feature_item].append(0)  # 无效数据
            """

        if not cut:
            for feature_item in self.features:
                self.features[feature_item] = self.features[feature_item][0]

    """
    no rolling + base 
    """

    @staticmethod
    def rolling_slice(window, step, signal, item_name):
        signal_list = []

        data = signal.data
        timestamp = signal.timestamp

        cur_time = timestamp[0]

        while cur_time < timestamp[-1]:
            # 切片
            # if timestamp[-1] - cur_time < window -1:
            #    break
            condition = np.logical_and(cur_time <= timestamp, timestamp <= min(cur_time + window, timestamp[-1]))
            if np.sum(condition) > 1:
                signal_list.append(Signal(item_name=item_name,
                                          data=data[np.where(condition)],
                                          timestamp=timestamp[np.where(condition)]))
            else:
                signal_list.append(Signal(item_name=item_name,
                                          data=np.array([0]),
                                          timestamp=np.array([cur_time])
                                          ))

            cur_time = cur_time + step

        return signal_list

    @staticmethod
    def rolling_slice2(window,step,signal):
        signal_list = []
        data = signal.data
        idx = 0

        while idx + window * signal.sample_rate < len(data) :
            slice = data[idx:idx+int(window*signal.sample_rate)]
            signal_list.append(slice)

            idx += int(step * window * signal.sample_rate)

        return signal_list

    def rolling(self):
        return Feature.rolling_slice(window=self.window, step=self.step * self.window, signal=self.signal,
                                     item_name=self.item_name)

    def print(self, content):
        return
        print("【特征提取】", content)

    def write(self, folder):
        if not os.path.isdir(folder):
            os.makedirs(folder)

        filepath = os.path.join(folder, self.item_name + '.csv')
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp'] + list(self.features.keys()))
            for i in range(len(self.timestamp)):
                row = [self.timestamp[i]] + [self.features[feature_item][i] for feature_item in self.features]
                writer.writerow(row)
