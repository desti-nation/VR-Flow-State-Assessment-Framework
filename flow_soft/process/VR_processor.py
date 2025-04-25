from matplotlib import pyplot as plt
from process.Top_processor import *
from signal_capture.utils.dataType import *

vr_space = ""


class VR_processor(Top_processor):
    def __init__(self, controller, player: int):
        super().__init__(controller,
                         player=player,
                         device='VR',
                         filelist=['VR_EyeTrack.csv'])

    def calc_feature2(self, Feature_cls_name=None, item_name_list=None):
        if Feature_cls_name is None and item_name_list is None:
            super().calc_feature2(Feature_cls_name='Feature_ET', item_name_list=['screen_gaze_point_x',
                                                                                 'screen_gaze_point_y',
                                                                                 'left_openness',
                                                                                 'right_openness'])
        else:
            super().calc_feature2(Feature_cls_name=Feature_cls_name, item_name_list=item_name_list)


