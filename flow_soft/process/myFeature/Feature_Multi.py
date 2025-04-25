import csv
import os

import numpy as np

from process.Signal import Signal
from process.myFeature.Feature import Feature


class Feature_Multi(Feature):
    def __init__(self,signal_list,feature_items):
        self.signal_list = signal_list # dict
        self.item_names = signal_list.keys()
        self.timestamp = []

        # 切分
        self.window = Feature.window
        self.step = Feature.step
        rolling_result= self.rolling()
        self.rolling_signals = []
        for i in range(len(list(rolling_result.values())[0])):
            self.rolling_signals.append({
                item_name:rolling_result[item_name][i] for item_name in self.item_names
            })
            rand_item = list(self.item_names)[0]
            self.timestamp.append(rolling_result[rand_item][i].timestamp[0])

        # 获取特征
        self.feature_items = feature_items
        # self.basic_feature_items = [self.item_name + '-' + _ for _ in BasicFeature]
        self.features = None


    def rolling(self):
        signal_list ={ item_name: [] for item_name in self.item_names }

        window = self.window
        step = self.step * window

        for item_name in self.item_names:
            signal = self.signal_list[item_name]
            data = signal.data
            timestamp = signal.timestamp

            cur_time = timestamp[0]

            while cur_time < timestamp[-1]:
                # 切片

                condition = np.logical_and(cur_time <= timestamp, timestamp <= min(cur_time + window, timestamp[-1]))
                if np.sum(condition) > 1:
                    signal_list[item_name].append(Signal(item_name=item_name,
                                              data=data[np.where(condition)],
                                              timestamp=timestamp[np.where(condition)]))
                else:
                    signal_list[item_name].append(Signal(item_name=item_name,
                                              data=np.array([0]),
                                              timestamp=np.array([cur_time])
                                              ))

                cur_time = cur_time + step

            self.print("=====切割完毕=====")
        return signal_list
    def calc(self, cut, base_feature=None):
        """
                if base_feature is None:
            base_feature = {_: 0 for _ in list(self.feature_items) + list(self.basic_feature_items)}

        """


        self.features = {feature_item: [] for feature_item in self.feature_items}

        signal_list = self.rolling_signals if cut else [self.signal_list]

        for signal in signal_list:
            #self.timestamp.append(signal.timestamp[0])
            self.__calc__(signal, base_feature)
            """
            try:
                self.__calc__(signal, base_feature)
            except Exception:
                for feature_item in self.feature_items:
                    self.features[feature_item].append(0)  # 无效数据
            """


        if not cut:
            for feature_item in self.features:
                self.features[feature_item] = self.features[feature_item][0]

    def write(self, folder):
        if not os.path.isdir(folder):
            os.makedirs(folder)

        filepath = os.path.join(folder, self.__class__.__name__ + '.csv')
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp'] + list(self.features.keys()))
            for i in range(len(self.timestamp)):
                row = [self.timestamp[i]] + [self.features[feature_item][i] for feature_item in self.features]
                writer.writerow(row)