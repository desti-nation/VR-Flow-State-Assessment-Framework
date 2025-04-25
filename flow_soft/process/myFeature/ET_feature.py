import math
from collections import namedtuple

import numpy as np
from datetime import datetime

from process.myFeature.Feature import Feature
from process.myFeature.Feature_Multi import *
from utils import Record
from .PyGazeAnalyser.pygazeanalyser.detectors import *

ETFeature = [
    'Fix_n', 'Sac_n', 'Blink_n',
    'Fix_meanDuration', 'Fix_sumDuration', 'Fix_maxDuration',

    'Sac_meanDuration', 'Sac_sumDuration', 'Sac_maxDuration',
    'Sac_meanDis', 'Sac_sumDis',
    'Sac_meanVel',

    'Blink_meanDuration', 'Blink_sumDuration'
]


class Feature_ET(Feature_Multi):
    def __init__(self, et):
        super().__init__(signal_list=et, feature_items=ETFeature)

    @staticmethod
    def feature_items():
        return ETFeature

    def __calc__(self, signal, base_feature):
        record = Record('[ET][feature extraction]')

        x, y = signal['screen_gaze_point_x'], signal['screen_gaze_point_y']
        x, y, time = x.data * 1000 , y.data * 1000 , x.timestamp * 1000

        left, right = signal['left_openness'], signal['right_openness']
        left, right, time2 = left.data, right.data, left.timestamp * 1000

        Eblk = blink_detection_tomato(left, right, time)
        blk_duration = np.array([_[2] for _ in Eblk]) if len(Eblk) > 0 else np.array([0])

        _, Efix = fixation_detection(x=x, y=y, time=time, maxdist=50, mindur=100) # todo 参数
        _, Esac = saccade_detection(x=x, y=y, time=time,maxvel=100)

        fix_duration = np.array([_[2] for _ in Efix]) if len(Efix) > 0 else np.array([0])
        sac_duration = np.array([_[2] for _ in Esac]) if len(Esac) > 0 else np.array([0])

        [sac_sx, sac_sy, sac_ex, sac_ey] = [np.array([_[i] for _ in Esac]) for i in range(3, 7)]
        sac_dis = np.sqrt((sac_ex - sac_sx) * (sac_ex - sac_sx) + (sac_ey - sac_sy) * (sac_ey - sac_sy))\
            if len(Esac) > 0 else np.array([0])
        sac_velocity = sac_dis / sac_duration if len(Esac) > 0 else np.array([0])

        result = {
            'Fix_n': len(fix_duration),
            'Sac_n': len(sac_duration),
            'Blink_n': len(blk_duration),

            'Fix_meanDuration': np.mean(fix_duration),
            'Fix_sumDuration': np.sum(fix_duration),
            'Fix_maxDuration': np.max(fix_duration),

            'Sac_meanDuration': np.mean(sac_duration),
            'Sac_sumDuration': np.sum(sac_duration),
            'Sac_maxDuration': np.max(sac_duration),

            'Sac_meanDis': np.mean(sac_dis),
            'Sac_sumDis': np.sum(sac_dis),
            'Sac_meanVel': np.mean(sac_velocity),

            'Blink_meanDuration': np.mean(blk_duration),
            'Blink_sumDuration': np.sum(blk_duration),
        }
        #print(result)

        for item_name in result:
            offset = base_feature[item_name] if base_feature is not None else 0
            self.features[item_name].append(result[item_name] - offset)

        record.end()
