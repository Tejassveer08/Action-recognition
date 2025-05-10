import os

def load_ucf101_paths(dataset_dir):
    video_paths, labels = [], []
    label_map = {}
    for i, class_name in enumerate(sorted(os.listdir(dataset_dir))):
        label_map[class_name] = i
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.endswith(".avi"):
                video_paths.append(os.path.join(class_dir, fname))
                labels.append(i)
    return video_paths, labels
