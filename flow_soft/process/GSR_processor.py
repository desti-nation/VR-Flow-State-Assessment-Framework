from process.Signal import Signal

import scipy.signal as signal
from scipy.signal import decimate
from process.Top_processor import *
from process.myFeature.GSR_feature import *
from signal_capture.utils.dataType import *


class GSR_processor(Top_processor):

    def __init__(self, controller, player: int):
        super().__init__(controller,
                         player=player,
                         device='GSR',
                         filelist=['GSR_Shimmer.csv'])
        #self.data[item_name][epoch_i][1].slice(window=150)
        _ = self.data['GSR_uS'][0][1]
        linePlot(x=_.timestamp,y=_.data)
        _ = self.data['GSR_uS'][1][1]
        linePlot(x=_.timestamp, y=_.data)
        _ = self.data['GSR_uS'][2][1]
        linePlot(x=_.timestamp, y=_.data)
    def preprocess(self):
        """
        gsr

        :return:
        """
        self.transUnit()
        self.data['GSR_uS'].gsr_preprocess()
        self.data['PPG_v'].ppg_preprocess()
        self.PPG2HR()

    def transUnit(self):
        """
        ohm转微西门子
        """
        gsr_ohm = self.data['GSR_ohm']
        self.data['GSR_uS'] = Signal(
            item_name='GSR_uS',
            data=1000 / gsr_ohm.data,
            timestamp=gsr_ohm.timestamp)

        self.data['PPG_v'] = Signal(
            item_name='PPG_v',
            data=self.data['PPG_mv'].data / 1000,
            timestamp=self.data['PPG_mv'].timestamp)


    def PPG2HR(self):
        ppg = self.data['PPG_v']
        peaks, _ = signal.find_peaks(ppg.data, distance=ppg.sample_rate * 0.6)

        # 3. 计算RRI
        RRI = np.diff(peaks) / ppg.sample_rate
        timestamp = ppg.timestamp[peaks[:-1]]

        self.data['RRI'] = Signal(item_name='RRI',timestamp=timestamp-timestamp[0],data=RRI)
        self.data['RRI'].rri_preprocess()

        timestamp = self.data['RRI'].timestamp

        HR = 60 / self.data['RRI'].data  # 每分钟心跳次数
        self.data['HR'] = Signal(item_name='HR', timestamp=timestamp - timestamp[0], data=HR)

        linePlot(self.data['HR'].timestamp, self.data['HR'].data,color='red')

    def feature_extraction(self, stat_item_list=None):
        super().feature_extraction(stat_item_list=['GSR_uS','HR'])
    def calc_feature1(self, Feature_cls_name=None, item_name=None):
        if Feature_cls_name is None and item_name is None:
            super().calc_feature1(Feature_cls_name='Feature_GSR', item_name='GSR_uS')
            super().calc_feature1(Feature_cls_name='Feature_HR',item_name='RRI')
        else:
            super().calc_feature1(Feature_cls_name=Feature_cls_name, item_name=item_name)

