import pyhrv
from process.myFeature.Feature_Statistic import *
from utils import Record

time_domain = [
    'SDNN', 'RMSSD',
    'NN50', 'NN20', 'pNN50', 'pNN20',
    'tri_index',
    'TINN', 'TINN_N', 'TINN_M',
]
freq_domain = [
    'VLFP', 'LFP', 'HFP', 'ratio',
]
nonlinear = [
    'SD1', 'SD2', 'SD_ratio',
    'Ellipse_Area'
]

HRFeature = time_domain + freq_domain + nonlinear
HRFeature = ['HRV_' + _ for _ in HRFeature]


class Feature_HR(Feature):
    """
    心率原始数据：hr
    最终会输出一个:
    按滑动窗口分割若干个时间窗，对每个时间窗计算这个窗口内的特征
    """

    def __init__(self, ppg):
        """
        :param window: 时间窗口
        :param fs: 采样率
        """
        super().__init__(signal=ppg, feature_items=HRFeature)

    @staticmethod
    def feature_items():
        return HRFeature

    def __calc__(self, signal, base_feature):
        record = Record('[HR][feature extraction]')

        # 输入到pyhrv中

        result = pyhrv.hrv(nni=signal.data, show=False)

        hrv_result = {}

        for feature_item in time_domain + nonlinear:
            hrv_result['HRV_' + feature_item] = result[feature_item.lower()]

        freq_method = 'fft_abs'
        (hrv_result['HRV_VLFP'],
         hrv_result['HRV_LFP'],
         hrv_result['HRV_HFP']) = (result[freq_method][0],
                                   result[freq_method][1],
                                   result[freq_method][2])
        hrv_result['HRV_ratio'] = hrv_result['HRV_LFP'] / hrv_result['HRV_HFP']



        for item_name in hrv_result:
            offset = base_feature[item_name] if base_feature is not None else 0
            self.features[item_name].append(hrv_result[item_name] - offset)

        record.end()
