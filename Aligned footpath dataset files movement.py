import os
import shutil
import fnmatch
from sklearn.model_selection import train_test_split


src_root = r"F:\Uob Dissertation Dataset\Aligned footpath dataset_new"
dst_root = r"F:\Uob Dissertation Dataset\footpath dataset_splitted"
src_folders = os.listdir(src_root)
color_images = []
for src_folder in src_folders:
    files_path = os.path.join(src_root, src_folder)
    files = os.listdir(files_path)
    for file in files:
        if fnmatch.fnmatch(file, '*_color.png'):
            color_images.append(file)

train, test = train_test_split(color_images, train_size=0.6)
test, val = train_test_split(test, test_size=0.5)

""" Move Files """
# train files
dst_path = os.path.join(dst_root, 'train')
for file in train:
    temp_src_root = os.path.join(src_root, file[0:15])
    temp_src = os.path.join(temp_src_root, file)
    temp_dst = os.path.join(dst_path, file)
    shutil.copy(temp_src, temp_dst) # copy and move *_color.png file

    temp_src = os.path.join(temp_src_root, file.replace('_color.png', '_depth.npy'))
    temp_dst = os.path.join(dst_path, file.replace('_color.png', '_depth.npy'))
    shutil.copy(temp_src, temp_dst)  # copy and move *_depth.npy file

    temp_src = os.path.join(temp_src_root, file.replace('_color.png', '_depth_visualization.png'))
    temp_dst = os.path.join(dst_path, file.replace('_color.png', '_depth_visualization.png'))
    shutil.copy(temp_src, temp_dst)  # copy and move *_depth.npy file

# val files
dst_path = os.path.join(dst_root, 'val')
for file in val:
    temp_src_root = os.path.join(src_root, file[0:15])
    temp_src = os.path.join(temp_src_root, file)
    temp_dst = os.path.join(dst_path, file)
    shutil.copy(temp_src, temp_dst) # copy and move *_color.png file

    temp_src = os.path.join(temp_src_root, file.replace('_color.png', '_depth.npy'))
    temp_dst = os.path.join(dst_path, file.replace('_color.png', '_depth.npy'))
    shutil.copy(temp_src, temp_dst)  # copy and move *_depth.npy file

    temp_src = os.path.join(temp_src_root, file.replace('_color.png', '_depth_visualization.png'))
    temp_dst = os.path.join(dst_path, file.replace('_color.png', '_depth_visualization.png'))
    shutil.copy(temp_src, temp_dst)  # copy and move *_depth.npy file

# test files
dst_path = os.path.join(dst_root, 'test')
for file in test:
    temp_src_root = os.path.join(src_root, file[0:15])
    temp_src = os.path.join(temp_src_root, file)
    temp_dst = os.path.join(dst_path, file)
    shutil.copy(temp_src, temp_dst) # copy and move *_color.png file

    temp_src = os.path.join(temp_src_root, file.replace('_color.png', '_depth.npy'))
    temp_dst = os.path.join(dst_path, file.replace('_color.png', '_depth.npy'))
    shutil.copy(temp_src, temp_dst)  # copy and move *_depth.npy file

    temp_src = os.path.join(temp_src_root, file.replace('_color.png', '_depth_visualization.png'))
    temp_dst = os.path.join(dst_path, file.replace('_color.png', '_depth_visualization.png'))
    shutil.copy(temp_src, temp_dst)  # copy and move *_depth.npy file

