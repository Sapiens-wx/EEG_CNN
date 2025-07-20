import numpy as np

# 只用这5个通道
channels = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']
num_channels = len(channels)
segment_length = 2560
num_samples_per_label = 10

# 标签定义
labels = ['left', 'middle', 'right']
label_map = {'left': 0, 'middle': 1, 'right': 2}

# 生成数据
segments = []
labels_list = []
for label in labels:
    for _ in range(num_samples_per_label):
        data = np.zeros((num_channels, segment_length), dtype=np.float32)
        if label == 'left':
            data[channels.index('TP9'), :] = 1
            data[channels.index('AF7'), :] = 1
        elif label == 'right':
            data[channels.index('AF8'), :] = 1
            data[channels.index('TP10'), :] = 1
        elif label == 'middle':
            data[channels.index('AF7'), :] = 1
            data[channels.index('AF8'), :] = 1
        segments.append(data)
        labels_list.append(label_map[label])

segments = np.array(segments)  # shape: (30, 5, 2560)
labels_arr = np.array(labels_list)  # shape: (30,)

np.save('training_data/preprocessed/eeg_segments_fake.npy', segments)
np.save('training_data/preprocessed/eeg_labels_fake.npy', labels_arr)
print('Fake EEG data and labels saved.')
