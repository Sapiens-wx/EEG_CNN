ip_address="192.168.3.206"
port=5000

"""
enum for the six states
"""
l2r=0; # left->right
r2l=1; # right->left
l2s=2; # left->rest
s2l=3; # rest->left
r2s=4; # right->rest
s2r=5; # rest->right

"""
l: left, r: right, s: rest

* taskLength: length of thinking each task. (in seconds)
* loopCount: how many times do you want to repeat this loop.
* windowSize: length of the sliding window. (in Hz. 256Hz/sec)
* stepSize: step size of two consecutive sliding window. (in Hz. 256Hz/sec)
* transitionLength: length of the transition from (e.g.) left to right. (in Hz. 256Hz/sec)
    this is the assumed value, which is used to guarantee in the sliding window technique that windows cover the full transition from one state to another
* commands and commandsEnum: typically don't need to be adjusted.
* commands and how this object works is explained in training_data/record_cued_eeg.py
"""
class RecordEEG:
    def __init__(self):
        self.taskLength=5;
        self.loopCount=2;
        self.windowSize=4;
        self.stepSize=0.5;
        self.transitionLength=2;
        self.hzPerSec=256;
        self.commands=['left','right','rest','left','rest','right'];
        self.commandsEnum=[r2l,l2r,r2s,s2l,l2s,s2r];
recordEEG=RecordEEG();

def CommandEnum2Label(val):
    """
    Parameters
    val: the enum for the six states (l2r,...)
    Returns
    an all-zero array with arr[val] set to 1
    """
    labels=[0,0,0,0,0,0];
    labels[val]=1;
    return labels;
