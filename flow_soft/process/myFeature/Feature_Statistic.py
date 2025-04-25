from collections import namedtuple
from datetime import datetime

import numpy as np
from process.myFeature.Feature import *
from scipy.stats import entropy, kurtosis, skew

from utils import Record

StatisticFeature = [
    'mean', 'median', 'std',
    'Q1', 'Q3', 'min', 'max',
    'diff1_mean', 'diff1_median', 'diff1_std',
    'diff2_mean', 'diff2_median', 'diff2_std',
    'entropy', 'kurtosis', 'skewness'
]


# StatisticFeature

class Feature_Statistic(Feature):
    def __init__(self, signal):
        super().__init__(signal=signal,
                         feature_items=[signal.item_name + _ for _ in StatisticFeature])

    @staticmethod
    def feature_items(prefix):
        return [prefix + _ for _ in StatisticFeature]

    def __calc__(self, signal, base_feature):
        record = Record('[statistic][feature extraction]')

        def basic3(x: np.ndarray):
            return [np.mean(x), np.median(x), np.std(x)]

        data = signal.data
        diff1 = np.diff(data)
        diff2 = np.diff(diff1)

        if len(data) <= 1:
            raise Exception('data length too short')

        value = (basic3(data) + [np.percentile(data, 25),
                                 np.percentile(data, 75),
                                 np.min(data), np.max(data)]
                 + basic3(diff1) + basic3(diff2) + self.distribution_feature(signal))

        result = {}
        assert len(value) == len(self.feature_items)
        for i in range(len(self.feature_items)):
            result[self.feature_items[i]] = value[i]

        for key in result:
            offset = base_feature[key] if base_feature is not None else 0
            self.features[key].append(result[key] - offset)

        record.end()

    def distribution_feature(self, signal):
        X = signal.data
        # 获取概率分布
        if np.max(X) == np.min(X):
            print('!!![warning] value_std=0 , item_name = ', signal.item_name, ' time: ', signal.timestamp[0], ' to ',
                  signal.timestamp[-1])
            return [0, 0, 0]
            # raise Exception('!!!!!!')

        X_normalized = (X - np.min(X)) / (np.max(X) - np.min(X))

        # 定义分箱的数量
        num_bins = 10

        # 使用numpy的digitize函数将归一化后的数据分箱
        bin_edges = np.histogram_bin_edges(X_normalized, bins=num_bins)
        binned_data = np.digitize(X_normalized, bin_edges)

        # 计算每个箱中的频率
        frequency, _ = np.histogram(binned_data, bins=num_bins)

        # 将频率转换为概率
        probability = frequency / len(X)

        return [entropy(probability), kurtosis(signal.data), skew(signal.data)]
