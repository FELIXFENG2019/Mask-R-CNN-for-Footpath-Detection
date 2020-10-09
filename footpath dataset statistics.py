import os
import json

subset = "test"
footpath_images = 0
highway_images = 0
obstacle_images = 0

footpath_instances = 0
highway_instances = 0
obstacle_instances = 0

dataset_dir = r"F:\Uob Dissertation Dataset\footpath dataset 2.0"
assert subset in ["train", "val", "test"]
files_dir = os.path.join(dataset_dir, subset)

# Note: In VIA 2.0, regions was changed from a dict to a list.
# VGG Image Annotator (up to version 2.0.10) saves each image in the form:
# { 'filename': 'footpath_13.png',
#   'size': 463211,
#   'regions': [
#          {
#           'shape_attributes': {
#               'name': 'polygon'
#               'all_points_x': [...],
#               'all_points_y': [...]},
#           'region_attributes': {'footpath': 'footpath'}}
#           ... more regions ...
#          ],
#   'file_attributes': {}
# }
annotations = json.load(open(os.path.join(files_dir, "via_region_data_"+subset+".json")))
annotations = list(annotations.values())  # don't need the dict keys
annotations = [a for a in annotations if a['regions']]
footpath_flag = 0
highway_flag = 0
obstacle_flag = 0
for a in annotations:
    for region in a['regions']:
        if region['region_attributes']['footpath'] == "footpath":
            footpath_instances = footpath_instances + 1
            footpath_flag = 1
        elif region['region_attributes']['footpath'] == "highway":
            highway_instances = highway_instances + 1
            highway_flag = 1
        elif region['region_attributes']['footpath'] == "obstacle":
            obstacle_instances = obstacle_instances + 1
            obstacle_flag = 1
    if footpath_flag == 1:
        footpath_images = footpath_images + 1
    if highway_flag == 1:
        highway_images = highway_images + 1
    if obstacle_flag == 1:
        obstacle_images = obstacle_images + 1

    footpath_flag = 0
    highway_flag = 0
    obstacle_flag = 0

print("Number of images containing footpath:", footpath_images)
print("Number of images containing highway:", highway_images)
print("Number of images containing obstacle:", obstacle_images)

print("The number of labelled instances of footpath:", footpath_instances)
print("The number of labelled instances of highway:", highway_instances)
print("The number of labelled instances of obstacle:", obstacle_instances)

