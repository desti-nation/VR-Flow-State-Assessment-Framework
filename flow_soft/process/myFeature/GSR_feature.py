import numpy
from biosppy.signals.eda import *

from process.Top_processor import *
from process.myFeature.Feature import Feature
from utils import Record
from utils.plot.lineChart import *

GSRFeature = [
    'SCR_n',
    'SCR_meanInterval',
    'SCR_meanAmp', 'SCR_sumAmp',
    'SCR_meanRiseTime', 'SCR_sumRiseTime',
    'SC_maxFFT', 'SC_minFFT', 'SC_meanFFT', 'SC_stdFFT'
]



class Feature_GSR(Feature):
    def __init__(self, gsr):
        super().__init__(signal=gsr, feature_items=GSRFeature)

    @staticmethod
    def feature_items():
        return GSRFeature

    def __calc__(self, signal, base_feature, plot=False):
        record = Record('[GSR][feature extraction]')

        fft_data = np.fft.fft(signal.data)
        freq = np.fft.fftfreq(len(signal.data), d=1 / signal.sample_rate)
        fft_mag = np.abs(fft_data)
        # 绘制频率和对应的FFT幅度
        if plot:
            linePlot(freq, fft_mag)

        try:
            edaResult = eda(signal=signal.data, sampling_rate=signal.sample_rate, show=False)
        # m, wd, eda_clean = process_statistical(signal.data, use_scipy=True, sample_rate=128, new_sample_rate=40,
        #                                      segment_width=600, segment_overlap=0)
            amplitudes = edaResult['amplitudes']
            onsets = signal.timestamp[edaResult['onsets']]
            peaks = signal.timestamp[edaResult['peaks']]

            if len(amplitudes) == 0:
                result = {
                    'SCR_n': 0,
                    'SCR_meanInterval': 0,
                    'SCR_meanAmp': 0,
                    'SCR_sumAmp': 0,
                    'SCR_meanRiseTime': 0,
                    'SCR_sumRiseTime': 0,
                    'SC_maxFFT': np.max(fft_mag),
                    'SC_minFFT': np.min(fft_mag),
                    'SC_meanFFT': np.mean(fft_mag),
                    'SC_stdFFT': np.std(fft_mag)
                }
            else:
                result = {
                    'SCR_n': len(amplitudes),
                    'SCR_meanInterval': np.mean(np.diff(peaks)),
                    'SCR_meanAmp': np.mean(amplitudes),
                    'SCR_sumAmp': np.sum(amplitudes),
                    'SCR_meanRiseTime': np.mean(peaks - onsets),
                    'SCR_sumRiseTime': np.sum(peaks - onsets),
                    'SC_maxFFT': np.max(fft_mag),
                    'SC_minFFT': np.min(fft_mag),
                    'SC_meanFFT': np.mean(fft_mag),
                    'SC_stdFFT': np.std(fft_mag)
                }
            if len(amplitudes) == 1:
                print('!!![SCR]: detect scr_n = 1, time:' ,signal.timestamp[0],' to ',signal.timestamp[-1])
                result['SCR_meanInterval'] = signal.timestamp[-1] - signal.timestamp[0]
        except Exception as e:
            if str(e) == 'index 0 is out of bounds for axis 0 with size 0':
                print('!!![SCR]: detect scr_n = 0, time:' ,signal.timestamp[0],' to ',signal.timestamp[-1])
                result = {
                    'SCR_n': 0,
                    'SCR_meanInterval': 0,
                    'SCR_meanAmp': 0,
                    'SCR_sumAmp': 0,
                    'SCR_meanRiseTime': 0,
                    'SCR_sumRiseTime': 0,
                    'SC_maxFFT': np.max(fft_mag),
                    'SC_minFFT': np.min(fft_mag),
                    'SC_meanFFT': np.mean(fft_mag),
                    'SC_stdFFT': np.std(fft_mag)
                }
            else:
                raise e

        for item_name in result:
            offset = base_feature[item_name] if base_feature is not None else 0
            self.features[item_name].append(result[item_name] - offset)
        record.end()

