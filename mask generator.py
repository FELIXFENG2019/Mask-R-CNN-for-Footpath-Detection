import os
import numpy as np
import json
import skimage.draw


dataset_root = r"C:\Users\FENGSHIJIA\Desktop\footpath dataset 2.0"
subsets = ["train", "val", "test"]
image_info = []
for subset in subsets:
    dataset_dir = os.path.join(dataset_root, subset)

    annotations = json.load(open(os.path.join(dataset_dir, "via_region_data_"+subset+".json")))
    annotations = list(annotations.values())  # don't need the dict keys
    for a in annotations:
        polygons = [r['shape_attributes'] for r in a['regions']]
        names = [r['region_attributes']['footpath'] for r in a['regions']]
        name_dict = {"footpath": 1, "highway": 2, "obstacle": 3}
        name_ids = [name_dict[a] for a in names]

        image_path = os.path.join(dataset_dir, a['filename'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        image_dict = {"source": "footpath", "id": a['filename'], "path": image_path, "width": width, "height": height,
                      "polygons": polygons, "class_ids": name_ids}
        image_info.append(image_dict)

    # load mask
    for image_id in range(0, len(image_info)):
        single_image_info = image_info[image_id]
        name_ids = single_image_info['class_ids']
        info = image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        class_ids = np.array(name_ids, dtype=np.int32)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        if len(info["polygons"]) >= 2:
            for i in range(len(info["polygons"]) - 2, -1, -1):
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # Save mask array to .npy
        temp_filename = info["id"].replace('color.png', 'mask.npy')
        temp_dir = os.path.join(dataset_dir, temp_filename)
        np.save(temp_dir, mask.astype(np.bool))

    image_info = []

