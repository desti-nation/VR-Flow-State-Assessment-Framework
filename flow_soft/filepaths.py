import os


class Folder:
    root = 'your_file_folder'
    rawValue = os.path.join(root,'data') 
    processed_pkl = os.path.join(root,'processed','pkl')
    processed_csv = os.path.join(root,'processed','csv')
    feature = os.path.join(root, 'feature')
    featureANS = os.path.join(root, 'featureANS')
    image = os.path.join(root, 'image')
    modelResult = os.path.join(root,'model')

class File:
    order = os.path.join(Folder.root,'order.pkl')
    chip = os.path.join(Folder.root,'chip.pkl')
    time_align = os.path.join(Folder.root,'time_align.pkl')
    VRScore_pkl = os.path.join(Folder.root,'VRscore.pkl') 
    VRScore_csv = os.path.join(Folder.root,'VRscore.csv') 
    VRScore_norm = os.path.join(Folder.root,'VRscore_norm.pkl')
    questionnaire = os.path.join(Folder.root,'questionnaire.csv')
    features = os.path.join(Folder.root, 'feature.pkl')
    dataset = os.path.join(Folder.root, 'dataset.pkl')
    