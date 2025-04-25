from collections import namedtuple

# 标准格式数据空间定义

# 设备数据
# ========================================HR Armband(ANT+)=================================
HR_Armband_basic = namedtuple('HR_Armband_basic', [
    'Time', 'Timestamp',
            'hr', 'beat_count', 'beat_event_time'
])
HR_Armband1 = namedtuple('HR_Armband1', [
    'Time', 'Timestamp', 'transmitter_operating_time'
])
"""transmitter_operating_time是用来对齐时间戳的"""
HR_Armband2 = namedtuple('HR_Armband2', [
    'Time', 'Timestamp', 'RR_Interval_ms'
])

HR_Armband_DATA = [HR_Armband_basic, HR_Armband1, HR_Armband2]
# ========================================HR Armband(ANT+)=================================


# ===========================================EEG(port)=====================================
Neuro_EEG_Power = namedtuple('Neuro_EEG_Power', ['Time', 'Timestamp', 'attention', 'meditation', 'delta', 'theta',
                                                 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma',
                                                 'midGamma', 'poorSignal', 'blinkStrength'])
Neuro_EEG_Raw = namedtuple('Neuro_EEG_Raw', ['Time', 'Timestamp', 'rawValue'])

Neuro_EEG_DATA = [Neuro_EEG_Power, Neuro_EEG_Raw]
# ===========================================EEG(port)=====================================


# =========================================ET(vr device)===================================
VR_EyeTrack = namedtuple('VR_EyeTrack', [
    'Time', 'Timestamp',
    'combine_gaze_vector_x', 'combine_gaze_vector_y', 'combine_gaze_vector_z',
    'world_gaze_point_x','world_gaze_point_y','world_gaze_point_z',
    'screen_gaze_point_x','screen_gaze_point_y','screen_gaze_point_z',
    'left_openness', 'right_openness',
])
VR_Event = namedtuple('VR_Event',[
    'Time','Timestamp','event'
])

VR_DATA = [VR_EyeTrack,VR_Event]
# =========================================ET(vr device)===================================


# ============================================GSR(port)====================================
GSR_Watch = namedtuple('GSR_Watch', [
    'Time', 'Timestamp', 'gsr', 'accX', 'accY', 'accZ', 'angle_vX', 'angle_vY', 'angle_vZ', 'hr', 'oxi'])

GSR_Watch_DATA = [GSR_Watch]
# ============================================GSR(port)====================================

# ======================================Shimmer(port)======================================
GSR_Shimmer = namedtuple('GSR_Shimmer',[
    'Time','Timestamp','GSR_ohm','gsr_to_volts','PPG_mv','accX', 'accY','accZ'
])

GSR_Shimmer_DATA = [GSR_Shimmer]
# ======================================Shimmer(port)======================================


# ==========================================CMS50EW(port)===================================
CMS50EW_Hr_Oxi = namedtuple('CMS50EW_Hr_Oxi',
                            ['Time', 'Timestamp', 'finger', 'Pulse_rate_bpm', 'spo2', 'ppg', 'strength'])

CMS50EW_DATA = [CMS50EW_Hr_Oxi]
# ==========================================CMS50EW(port)===================================


# ======================================Emotion(camera+model)===============================
Emotion_Compute = namedtuple('Emotion_Compute',
                             ['Time', 'Timestamp',
                              'angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral'])
# 情感计算
EMOTIONS_Categories = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']

Camera_DATA = [Emotion_Compute]
# ======================================Emotion(camera+model)===============================


# 设备管理器
DEVICE_MANAGER = namedtuple('DEVICE_MANAGER', ['oxi', 'emotion', 'eeg', 'gsr', 'hr_antplus','vr'])

DEFAULT_FRAME_BEGIN = b'\xfa'
DEFAULT_FRAME_END = b'\xaf'


# 转换
class DECODER:
    @staticmethod
    def begin(num,begin=DEFAULT_FRAME_BEGIN):
        if type(num) == type(1):
            return num == int.from_bytes(begin, byteorder='little')
        elif type(num) == type(begin):
            return num == begin
        else:
            raise Exception('请传入bytes类型或int类型的数据，当前数据类型：'+str(type(num)))

    @staticmethod
    def end(num,end=DEFAULT_FRAME_END):
        if type(num) == type(1):
            return num == int.from_bytes(end, byteorder='little')
        elif type(num) == type(end):
            return num == end
        else:
            raise Exception('请传入bytes类型或int类型的数据，当前数据类型：'+str(type(num)))


class DECODE_GSR_DATA(DECODER):
    """
    1. 帧定界
    2. 将字节数据映射到指定量程，'/'前是量程大小，'/'后是字节数据的范围大小
    """
    GSR = 3.3 / 4095
    ACCELERATION = 2.0 / 32768
    ANGULAR_VELOCITY = 250.0 / 32768


class DECODE_VR_DATA(DECODER):
    FRAME_BEGIN = b'\x7E'
    FRAME_END = b'\x7F'
    @staticmethod
    def begin(num,begin=FRAME_BEGIN):
        return super(DECODE_VR_DATA,DECODE_VR_DATA).begin(num,begin)

    @staticmethod
    def end(num,end=FRAME_END):
        return super(DECODE_VR_DATA,DECODE_VR_DATA).end(num,end)

