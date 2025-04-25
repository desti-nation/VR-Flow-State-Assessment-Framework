from process.Top_processor import *
from process.EEG_processor import EEG_processor
from process.GSR_processor import GSR_processor
from process.VR_processor import VR_processor

controller = Controller(processed=False)

def preprocess():
    for player in controller.players():
        print('[player=',player,']')
        EEG_processor(controller=controller,player=player)
        GSR_processor(controller=controller,player=player)
        VR_processor(controller=controller,player=player)

def feature_extraction():
    controller.processed = True
    for player in controller.players():
        #if player != 8:
        #    continue
        print('[player=',player,']')
        EEG_processor(controller=controller,player=player).feature_extraction()
        GSR_processor(controller=controller,player=player).feature_extraction()
        VR_processor(controller=controller,player=player).feature_extraction()

    controller.writeFeature()
#preprocess()
feature_extraction()

