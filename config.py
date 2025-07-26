ip_address="192.168.3.206"
port=5000

def labels_to_orders(labels):
    """将labels列表转换为对应的order数组
    
    Args:
        labels (list): 包含label字符串的列表
        
    Returns:
        std_labels_ret: std version of [labels]
        orders: 包含对应order的列表，如果label无效则对应位置为None
    """
    import labels as lbl
    orders = []
    std_label_ret=[]
    for label in labels:
        normalized = lbl.normalize_label(label)
        found = False
        for std_label, info in lbl.label_map.items():
            if normalized in info["aliases"] or normalized == std_label:
                orders.append(info["order"])
                std_label_ret.append(std_label)
                found = True
                break
        if not found:
            raise ValueError(f"has invalid label [{label}]");
    return std_label_ret, orders

"""
l: left, r: right, n: neutral

* taskLength: length of thinking each task. (in seconds)
* loopCount: how many times do you want to repeat this loop.
* transitionLength: length of the transition from (e.g.) left to right. (in Hz. 256Hz/sec)
    this is the assumed value, which is used to guarantee in the sliding window technique that windows cover the full transition from one state to another     
* cues: the list of cues that you want to generate in order.
"""
class RecordEEG:
    def __CalcDuration(self):
        return (self.taskLength+self.transitionLength)*len(self.cues)*self.loopCount;
    def __init__(self):
        self.taskLength=8;
        self.transitionLength=1;
        self.hzPerSec=256;
        # below will be set by function Setcues.
        self.loopCount=0;
        self.cues=[];
        self.cuesIdx=[];
        self.duration=0;
    def SetCues(self, cues, loopCount):
        """
        sets self.cues=cues.
        Args:
            cues: A list containing the cues that you want to generate in order. For example, ['l','r','n']. Do not repeat cues.
            loopCount: how many times do you want to repeat the cues during the recording
        """
        self.loopCount=loopCount;
        self.cues, self.cuesIdx=labels_to_orders(cues);
        self.duration=self.__CalcDuration();
    def SetStandardCuesAndIdx(self, cues, idx, loopCount):
        """
        sets self.cues=cues and self.cuesIdx=idx.
        Args:
            cues: A list containing the cues that you want to generate in order. label must be standard label (e.g., 'left', 'right')
            idx: a list containing the index of each cue in cues
            loopCount: how many times do you want to repeat the cues during the recording
        """
        self.cues=cues;
        self.cuesIdx=idx;
        self.loopCount=loopCount;
        self.duration=self.__CalcDuration();

recordEEG=RecordEEG();