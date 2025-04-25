from process.myFeature.Feature import *
from process.myFeature.Feature_Multi import Feature_Multi
from utils import Record

EEGFeature = ['fractal_dim', 'NSI']
bands = [
    'delta',
    'theta',
    'lowAlpha','highAlpha',
    'lowBeta','highBeta',
    'lowGamma','midGamma'
]

class Feature_EEG(Feature):
    def __init__(self, eeg):
        super().__init__(signal=eeg, feature_items=EEGFeature)

    @staticmethod
    def feature_items():
        return EEGFeature

    def __calc__(self, signal, base_feature):
        record = Record('[EEG][feature extraction]')
        result = {
            'fractal_dim': self.higuchi_fd(signal.data),
            'NSI': self.NSI(signal.data)
        }
        for key in result:
            offset = base_feature[key] if base_feature is not None else 0
            self.features[key].append(result[key] - offset)

        record.end()
    def higuchi_fd(self, x, k_max=10):  # todo  参数
        """
        from gpt 使用higuichi方法计算分形维度
        """
        n = len(x)
        lk = np.zeros(k_max)

        # 计算每个分割段的长度
        for k in range(1, k_max + 1):
            Lmk = 0
            for m in range(k):
                Lmkn = 0
                for i in range(1, int(np.floor((n - m) / k)) + 1):
                    Lmkn += abs(x[m + i * k - 1] - x[m + (i - 1) * k - 1])
                Lmkn *= (n - 1) / (((n - m) / k) * k)
                Lmk += Lmkn
            Lmk /= k
            Lmk /= (n - 1) / k
            lk[k - 1] = Lmk

        # 计算分型维度
        h_fd = np.log(lk[0]) / np.log(1.0 / np.arange(1, k_max + 1)).mean()

        return h_fd

    def NSI(self, data):
        """
        非平稳指数 Non-stationart Index
        """
        nsi_value = np.std(np.diff(np.unwrap(np.angle(np.fft.fft(data)))))
        return nsi_value


class Feature_EEG_Power(Feature_Multi):
    def __init__(self, eeg_power):
        super().__init__(signal_list=eeg_power, feature_items=[band + '_relativeMag' for band in bands])

    @staticmethod
    def feature_items():
        return [band + '_relativeMag' for band in bands]
    def __calc__(self, signal, base_feature):
        record = Record('[Feature_EEG_Power]')
        total = []
        for i in range(len(signal['delta'].data)):
            total.append(sum([signal[item_name].data[i] for item_name in signal]))
        if 0 in total:
            raise Exception('!!![ERROR] eeg_power total=0, time: ',signal['delta'].timestamp[0],' to ',signal['delta'].timestamp[-1])
        result = {}
        for item_name in signal:
            _ = []
            for i in range(len(signal[item_name].data)):
                _.append(signal[item_name].data[i] / total[i])

            result[item_name + '_relativeMag'] = np.mean(np.array(_))

        for key in result:
            offset = base_feature[key] if base_feature is not None else 0
            self.features[key].append(result[key] - offset)

        record.end()