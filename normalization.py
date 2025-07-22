import numpy as np

normalizations=['none', 'z-score', 'min-max', 'robust']

# @param -arr the eeg signals. For example, a 256-sample signal with 4 channels should have shape (256,4)
def normalize(arr, normalization):
    if normalization == normalizations[1]: # z-score
        arr = (arr - np.mean(arr, axis=0)) / np.std(arr, axis=0)
    elif normalization == normalizations[2]: # min-max
        arr = (arr - np.min(arr, axis=0)) / (np.max(arr, axis=0) - np.min(arr, axis=0))
    elif normalization == normalizations[3]: # robust
        median = np.median(arr, axis=0)
        iqr = np.percentile(arr, 75, axis=0) - np.percentile(arr, 25, axis=0)
        arr = (arr - median) / iqr
    elif normalization != normalizations[0]:
        raise ValueError("invalid normalization method")
    return arr