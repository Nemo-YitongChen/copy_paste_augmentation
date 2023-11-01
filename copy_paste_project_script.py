import torch, detectron2

from detectron2.utils.logger import setup_logger

setup_logger()

import numpy as np
import os, json, cv2, random, sys, pyclipper

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def cp_clip(anno_list, dev=False) -> list:
    new_anno = anno_list[-1]
    new_anno_list = []
    clip = [np.array(x) for x in segmentation_to_points(new_anno['segmentation'])]
    for anno in anno_list[:-1]:
        subj = [np.array(x) for x in segmentation_to_points(anno['segmentation'])]
        for area in clip:
            pc = pyclipper.Pyclipper()
            pc.AddPath(area, pyclipper.PT_CLIP, True)
            try:
                pc.AddPaths(subj, pyclipper.PT_SUBJECT, True)
            except:
                if dev:
                    print(anno_list)
                    print('anno: ', anno)
                    print("subj: ", subj)
            subj = pc.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
            if subj == []:
                break
        if subj == []:
            continue
        else:
            anno['bbox'] = points_to_bbox(subj)
            anno["segmentation"] = points_to_segmentation(subj)
            anno['area'] = points_to_area(subj)
            new_anno_list.append(anno)
    new_anno_list.append(new_anno)
    return new_anno_list


def flip_bbox(bbox: list, width: int) -> list:
    """
    return a flipped bounding box according to the original bounding box list
    and the width of the image.

    Parameters:
    bbox    : list    , the list presentation for the original bounding box
    width   : integer , the width of the original image for flipping
    """
    new_bbox = bbox.copy()
    new_bbox[0] = width - new_bbox[0] - new_bbox[2]
    return new_bbox

def flip_segmentation(segmentation, width):
    """
    return a flipped segmentation list or dictionary according to the original
    segmentation instance and the width of the image.

    Parameters:
    segmentation  : list / dict , the list/dict presentation for the original
                                  segmentation area
    width         : integer     , the width of the original image for flipping
    """
    if len(segmentation) == 0:
        return []
    if type(segmentation) == type({}):  # RLE representation
        [height, width] = segmentation['size']  # parse the size of the image
        new_count = []
        count_sum = 0
        for idx, count in enumerate(segmentation['counts']):
            if idx % 2 == 0:
                count_sum += count
            if idx % 2 == 1:
                count_sum += count
                rows = count_sum // width
                new_count.append((2 * rows + 1) * width - count_sum)
                new_count.append(count)
        if len(new_count) != len(segmentation['counts']):
            new_count.append(height * width - sum(new_count))
        assert (len(new_count) == len(segmentation['counts']))
        return {'size': [height, width], 'counts': new_count}
    else:  # Polygon representation
        segmentations = []
        for seg in segmentation:
            new_segmentation = []
            for idx, coordinate in enumerate(seg):
                if idx % 2 == 0:
                    coordinate = width - coordinate
                new_segmentation.append(coordinate)
            segmentations.append(new_segmentation)
        return segmentations


def wrap_coordinates_affine(points, M) -> list:
    new_points = []
    for point in points:
        # new_points.append(M.dot(np.append(np.array(point), (1))))
        temp = np.append(np.array(point), (1)).reshape(3, 1)
        new_points.append(M.dot(temp).reshape(2))
    return new_points


def bbox_to_points(bbox):
    return np.array([
        [0, 0],
        [bbox[2], 0],
        [bbox[2], bbox[3]],
        [0, bbox[3]],
    ])


def segmentation_to_points(segmentation, bias=(0, 0)) -> list:
    segmentation_points = []
    for area in segmentation:
        area_points = []
        point = []
        for idx, coord in enumerate(area):
            if idx % 2 == 0:
                point.append(coord - bias[0])
            if idx % 2 == 1:
                point.append(coord - bias[1])
                area_points.append(point)
                point = []
        segmentation_points.append(area_points)
    return segmentation_points


def points_to_segmentation(segmentation_points) -> list:
    segmentation = []
    for area in segmentation_points:
        segmentation_area = []
        segmentation_xs = [point[0] for point in area]
        segmentation_ys = [point[1] for point in area]
        
        for x, y in zip(segmentation_xs, segmentation_ys):
            segmentation_area.append(x)
            segmentation_area.append(y)

        segmentation.append(segmentation_area)

    return segmentation


def points_to_bbox(segmentation_points) -> list:
    segmentation_xs = []
    segmentation_ys = []
    for area in segmentation_points:
        segmentation_xs += [point[0] for point in area]
        segmentation_ys += [point[1] for point in area]
    return [min(segmentation_xs), min(segmentation_ys), 
    max(segmentation_xs) - min(segmentation_xs), 
    max(segmentation_ys) - min(segmentation_ys)]


def points_to_area(segmentation_points) -> float:
    new_area = 0.0
    for seg_area in segmentation_points:
        new_area += Polygon(seg_area).area
    return new_area
    
    
def wrap_affine_transformation(bbox, segmentation, img, area, centre, angle=0.0, x_scale=1.0, y_scale=1.0, x_shift=0.0,
                               y_shift=0.0, cols=0, rows=0, dev=False) -> np.array:
    # defien return values
    new_img = img.copy()
    bias = [bbox[0], bbox[1]]
    segmentation_points = segmentation_to_points(segmentation, bias)

    # Transformation
    M_transformation = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    try:
        new_img = cv2.warpAffine(new_img, M_transformation, (2 * cols, 2 * rows))
    except:
        if dev:
            print("scaling failed.")
            print('Feature shape: ', img.shape)
            print('New feature shape: ', new_img.shape)
            plt.imshow(img)
            plt.imshow(new_img)
        return (False, bbox, segmentation, img, area)
    temp_seg_points = []
    for area in segmentation_points:
        temp_seg_points.append(wrap_coordinates_affine(area, M_transformation))
    segmentation_points = temp_seg_points.copy()
    centre = wrap_coordinates_affine([centre], M_transformation)[0]

    # Scaling
    try:
        new_img = cv2.resize(new_img, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_CUBIC)
    except:
        if dev:
            print("scaling failed.")
            print('Feature shape: ', img.shape)
            print('New feature shape: ', new_img.shape)
            plt.imshow(img)
            plt.imshow(new_img)
        return (False, bbox, segmentation, img, area)
    centre = [centre[0] * x_scale, centre[1] * y_scale]
    temp_seg_points = []
    for area in segmentation_points:
        temp_seg_points.append([[x * x_scale, y * y_scale] for [x, y] in area])
    segmentation_points = temp_seg_points.copy()
    centre = wrap_coordinates_affine([centre], M_transformation)[0]

    # Rotation
    M_rotation = cv2.getRotationMatrix2D((centre[0], centre[1]), angle, 1)
    try:
        new_img = cv2.warpAffine(new_img, M_rotation, (2 * cols, 2 * rows))
    except:
        if dev:
            print("Rotation failed.")
            print('Feature shape: ', img.shape)
            print('New feature shape: ', new_img.shape)
            plt.imshow(img)
            plt.imshow(new_img)
        return (False, bbox, segmentation, img, area)
    temp_seg_points = []
    for area in segmentation_points:
        temp_seg_points.append(wrap_coordinates_affine(area, M_rotation))
    segmentation_points = temp_seg_points.copy()

    # Fix bbox and segmentation
    subj = segmentation_points
    clip = [[0, 0], [0, rows], [cols, rows], [cols, 0]]
    pc = pyclipper.Pyclipper()
    pc.AddPath(clip, pyclipper.PT_CLIP, True)
    try:
        pc.AddPaths(subj, pyclipper.PT_SUBJECT, True)
    except:
        if dev:
            print("Clipping failed.")
            print("segmentation: ", segmentation)
            print('subj: ', subj)
            plt.imshow(img)
            plt.imshow(new_img)
        return (False, bbox, segmentation, img, area)
    segmentation_points = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)
    if segmentation_points == []:
        return (False, bbox, segmentation, img, area)
    temp_seg_points = []
    for area in segmentation_points:
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(area, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        temp_seg_points.append(pco.Execute(-2.0))

    if segmentation_points == []:
        return (False, bbox, segmentation, img, area)
    new_segmentation = points_to_segmentation(segmentation_points)
    new_bbox = points_to_bbox(segmentation_points)
    new_area = points_to_area(segmentation_points)

    if new_area <= 40:
        return (False, bbox, segmentation, img, area)
    return (True, new_bbox, new_segmentation, new_img[:rows, :cols], new_area)


def get_processed_image_dict(json_file, flip_image=False, remove_rle=True, dev=False) -> (dict, set):
    """
    return a dictionary of pictures info and a set of original image ids:
    used as the preparation for converting certain json files to acceptable
    COCO instance datasets.

    Parameters:
    json_file   : string  , the path to the annotation file
    percentage  : float   , percentage of copy-paste instance compared to the
                            original instance
    flip_image  : bool    , indicator for generating flipped image or not.
    """
    # json_file = os.path.join(img_dir, 'annotations','instances_val2017.json')
    with open(json_file) as f:
        imgs_anns = json.load(f)
        
    img_dir = os.path.split(os.path.split(json_file)[0])[0]
    
    # Declarations of variables
    anno_idx = 0
    imageid_data_dict = {}
    flipped_imageid_data_dict = {}
    remove_set = set()
    anno_list = imgs_anns['annotations']
    category_list = [x['id'] for x in imgs_anns['categories']]
    category_dict = {}
    for idx, id in enumerate(category_list):
      category_dict[id] = idx

    ### dataset preprocessing and registration ###
    for v in anno_list:
        image_id = str(v['image_id'])
        # Ignore images with RLE encoding if remove_rle is True
        if remove_rle:
            if v['iscrowd'] == 1:
                remove_set.add(image_id)
                if flip_image:
                    remove_set.add(image_id + '_')
                continue
        # Add features to existing entities
        if image_id in imageid_data_dict.keys() and image_id not in remove_set:
            #### Update original image
            imageid_data_dict[image_id]['annotations'].append({
                "bbox": v['bbox'],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": v['segmentation'],
                "category_id": category_dict[v['category_id']],
                "iscrowd": v['iscrowd'],
                'area': v['area'],
                'image_id': image_id,
                'id': anno_idx,
            })
            anno_idx += 1
            if flip_image == True:
                #### Update flipped image
                width = imageid_data_dict[image_id + '_']['width']
                imageid_data_dict[image_id + '_']['annotations'].append({
                    "bbox": flip_bbox(v['bbox'], width),
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": flip_segmentation(v['segmentation'], width),
                    "category_id": category_dict[v['category_id']],
                    "iscrowd": v['iscrowd'],
                    'area': v['area'],
                    'image_id': image_id + '_',
                    'id': anno_idx,
                })
                flipped_imageid_data_dict[image_id + '_']['annotations'].append(imageid_data_dict[image_id + '_']['annotations'][-1])
                anno_idx += 1
        elif image_id not in imageid_data_dict.keys() and image_id not in remove_set:
            # format annotations for new images
            record = {}

            filename = os.path.join(
                img_dir,
                '0' * (12 - len(str(image_id))) + str(image_id) + '.jpg'
            )
            image = cv2.imread(filename)
            height, width = image.shape[:2]

            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = height
            record["width"] = width
            record["annotations"] = [{
                "bbox": v['bbox'],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": v['segmentation'],
                "category_id": category_dict[v['category_id']],
                "iscrowd": v['iscrowd'],
                'area': v['area'],
                'image_id': image_id,
                'id': anno_idx,
            }]
            anno_idx += 1
            imageid_data_dict[image_id] = record

            if flip_image == True:
                # generate and record derived pictures
                new_name = os.path.join(
                    img_dir,
                    '0' * (12 - len(image_id)) + image_id + '_' + '.jpg'
                )
                image = cv2.flip(image, 1)
                cv2.imwrite(new_name, image)

                record_ = {}
                record_["file_name"] = new_name
                record_["image_id"] = image_id + '_'
                record_["height"] = height
                record_["width"] = width
                record_["annotations"] = [{
                    "bbox": flip_bbox(v['bbox'], width),
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": flip_segmentation(v['segmentation'], width),
                    "category_id": category_dict[v['category_id']],
                    "iscrowd": v['iscrowd'],
                    'area': v['area'],
                    'image_id': image_id + '_',
                    'id': anno_idx,
                }]
                anno_idx += 1
                imageid_data_dict[image_id + '_'] = record_
                flipped_imageid_data_dict[image_id + '_'] = record_

                if dev:
                    if anno_idx >= 10:
                        break
    for image_id in remove_set:
        if image_id in imageid_data_dict.keys():
            imageid_data_dict.pop(image_id)
            if flip_image and image_id[-1] == '_':
                flipped_imageid_data_dict.pop(image_id)
    ### End of dataset preprocessing and registration ###
    original_imageid_set = set([str.strip(key, '_') for key in imageid_data_dict.keys()])
    print("End of annotation preprocessing.")
    return (imageid_data_dict, original_imageid_set, flipped_imageid_data_dict)

def cp_train_test_split(imageid_data_dict, original_imageid_set, train_percentage=0.7, test_percentage=0.2,
                        val_percentage=0.1):
    train_list, test_list = train_test_split(list(original_imageid_set), train_size=train_percentage,
                                             test_size=test_percentage + val_percentage)
    t = int(len(test_list) * test_percentage / (test_percentage + val_percentage))
    test_list, val_list = (test_list[: t], test_list[t:])
    if train_list[0] + '_' in imageid_data_dict.keys():
        return [imageid_data_dict[id] for id in train_list] + [imageid_data_dict[id + '_'] for id in train_list], \
               [imageid_data_dict[id] for id in test_list] + [imageid_data_dict[id + '_'] for id in test_list], \
               [imageid_data_dict[id] for id in val_list] + [imageid_data_dict[id + '_'] for id in val_list]
    return [imageid_data_dict[id] for id in train_list], [imageid_data_dict[id] for id in test_list], [
        imageid_data_dict[id] for id in val_list]

def copy_paste_augmentation(filename, image_dicts_list, percentage=0.0, scaling=True, rotation=True, 
                            keep_aspect_ratio=False, dev=False) -> list:
    img_dir = os.path.split(os.path.split(filename)[0])[0]
    decorator = '_'
    if scaling:
        decorator += 's'
    if rotation:
        decorator += 'r'
    if keep_aspect_ratio:
        decorator += 'a'
    print("start Copy Paste Augmentation, decorator = ", decorator)
    ### Randomly adopting copy paste augmentation ###
    anno_idx = -1
    new_image_dict_list = image_dicts_list.copy()
    if percentage > 0.000001:
        count = 0
        maximum = int(len(image_dicts_list) * percentage)
        if dev:
            maximum = 3
        original_length = len(image_dicts_list)
        while (count < maximum):
            # get random feature image
            dict_idx_feature = random.randint(0, original_length - 1)
            feature_dict = image_dicts_list[dict_idx_feature]
            feature_img = cv2.imread(feature_dict['file_name'])
            feature_anno = feature_dict['annotations']

            if dev:
                print("feature image:")
                plt.imshow(feature_img)

            for feature in feature_anno:
                if type(feature['segmentation']) == type([]) and len(feature['segmentation']) == 1:
                    bbox = np.array(feature['bbox'], dtype=np.int32)

                    segmentation_points = segmentation_to_points(feature['segmentation'], [0, 0])
                    feature_mask = np.zeros((feature_dict['height'], feature_dict['width']))
                    cv2.fillPoly(feature_mask, np.array(segmentation_points, dtype=np.int32), 1)

                    # get random background image
                    dict_idx_background = random.randint(0, original_length - 1)
                    background_dict = image_dicts_list[dict_idx_background]
                    background_img = cv2.imread(background_dict['file_name'])
                    background_anno = background_dict['annotations']

                    if dev:
                        print("background image:")
                        plt.imshow(background_img)

                    feature_mask = feature_mask.astype(bool)
                    feature_block = (feature_img * np.stack([feature_mask] * 3, axis=2))[bbox[1]:bbox[1] + bbox[3],
                                    bbox[0]:bbox[0] + bbox[2]]

                    validity = False
                    try_out_time = 2
                    while (not validity and try_out_time):
                        angle = 0
                        x_scale = 1
                        y_scale = 1
                        x_shift = random.random() * (background_dict['width'] - feature_block.shape[1])
                        y_shift = random.random() * (background_dict['height'] - feature_block.shape[2])
                        if scaling:
                            x_scale = random.random() * 2.7 + 0.3
                            y_scale = random.random() * 2.7 + 0.3
                            if keep_aspect_ratio:
                                y_scale = x_scale
                        if rotation:
                            angle = random.random() * 360
                        (validity,
                        new_bbox,
                        new_segmentation,
                        affine_feature,
                        new_area) = wrap_affine_transformation(
                                                    feature['bbox'],
                                                    feature['segmentation'],
                                                    feature_block.copy(),
                                                    feature['area'],
                                                    centre=[feature_block.shape[1] / 2, feature_block.shape[0] / 2],
                                                    angle=angle,
                                                    x_scale=x_scale,
                                                    y_scale=y_scale,
                                                    x_shift=x_shift,
                                                    y_shift=y_shift,
                                                    cols=background_dict['width'],
                                                    rows=background_dict['height'])
                        try_out_time -= 1
                    if try_out_time == 0:
                      continue

                    bg_mask = np.zeros((background_dict['height'], background_dict['width']))
                    new_segmentation_points = segmentation_to_points(new_segmentation, [0, 0])
                    try:
                        cv2.fillPoly(bg_mask, [np.array(x) for x in new_segmentation_points], 1)
                    except ValueError:
                        print("handle value error for cv2.fillpoly", new_segmentation_points)
                        plt.imshow(feature_block)
                        plt.imshow(affine_feature)
                        plt.imshow(bg_mask)
                    bg_mask = bg_mask.astype(bool)
                    if dev:
                        s1 = affine_feature * np.stack([bg_mask] * 3, axis=2) + background_img * np.stack(
                            [~bg_mask] * 3, axis=2)

                    kernel = np.ones((3, 3), np.uint8)
                    bg_mask = cv2.erode(np.array(bg_mask).astype(np.float32), kernel, iterations=1).astype(bool)
                    sythesised_img = affine_feature * np.stack([bg_mask] * 3, axis=2) + background_img * np.stack(
                        [~bg_mask] * 3, axis=2)

                    if dev:
                        plt.imshow(feature_block)
                        plt.imshow(affine_feature)
                        plt.imshow(s1)
                        plt.imshow(sythesised_img)
                    
                    image_id = background_dict['image_id'] + decorator + str(anno_idx)
                    new_name = os.path.join(
                        img_dir,
                        '0' * (12 - len(image_id)) + image_id + '.jpg'
                    )
                    cv2.imwrite(new_name, sythesised_img)

                    record = {}
                    record["file_name"] = new_name
                    record["image_id"] = image_id
                    record["height"] = background_dict['height']
                    record["width"] = background_dict['width']
                    record["annotations"] = background_anno.copy()
                    record["annotations"].append({
                        "bbox": new_bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": new_segmentation,
                        "category_id": feature['category_id'],
                        "iscrowd": feature['iscrowd'],
                        'area': new_area,
                        'image_id': image_id,
                        'id': anno_idx,
                    })
                    record["annotations"] = cp_clip(record["annotations"])
                    anno_idx -= 1
                    new_image_dict_list.append(record)
                    count += 1

                    if dev:
                        img = sythesised_img
                        visualizer = Visualizer(img[:, :, ::-1], scale=0.5)
                        out = visualizer.draw_dataset_dict(record)
                        plt.imshow(out.get_image()[:, :, ::-1])
                else:
                  continue

    ### End of Randomly adopting copy paste augmentation ###
    return new_image_dict_list


def get_list(path, work_list):
    return work_list


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def train(training_set_name, test_set_name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (test_set_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 20
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.OUTPUT_DIR = os.path.join(res_dir, 'output', training_set_name)
    cfg.DATASETS.TEST = (test_set_name,)
    cfg.TEST.EVAL_PERIOD = 2
    if torch.backends.mps.is_available():
        cfg.MODEL.DEVICE = "mps"
    elif not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
      
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print('Start training: ', training_set_name)
    trainer.train()
    

if __name__ == '__main__':
    train_img_dir = sys.argv[1]
    test_img_dir = sys.argv[2]
    res_dir = sys.argv[3]
    train_file = os.path.join(train_img_dir, 'annotations', 'instances_train2017.json')
    
    if sys.argv[4] == 'test':
        train_img_dir = os.path.join(test_img_dir)
        train_file = os.path.join(test_img_dir, 'annotations', 'instances_val2017.json')

    test_img_dir = os.path.join(test_img_dir)
    test_file = os.path.join(test_img_dir, 'annotations', 'instances_val2017.json')

    with open(train_file) as f:
        imgs_anns = json.load(f)
    thing_classes = [x['name'] for x in imgs_anns['categories']]

    print('Preparing dataset: ...')
    original_dataset, id_set, _ = get_processed_image_dict(train_file, flip_image=False, remove_rle=True)
    original_dataset_list = list(original_dataset.values())
    train_dataset_list, test_list = train_test_split(original_dataset_list, train_size=0.8, test_size=0.2, random_state = 0)
    
    print('Original dataset prepared: ', len(original_dataset_list), ' images.')
    print('Coco training dataset prepared: ', len(train_dataset_list), ' images.')
    
    _, _, flipped_dataset = get_processed_image_dict(train_file, flip_image=True, remove_rle=True)
    flipped_dataset_list = list(flipped_dataset.values())
    flipped_dataset_list, _ = train_test_split(flipped_dataset_list, train_size=0.8, test_size=0.2, random_state = 0)
    
    print('Flip Augmented dataset prepared: ', len(flipped_dataset_list), ' images.')
    
    print(train_dataset_list[0]['image_id'],flipped_dataset_list[0]['image_id'] )
    # assert(train_dataset_list[0]['image_id'] + '_' == flipped_dataset_list[0]['image_id'])
    
    cp_train_list = copy_paste_augmentation(train_file, train_dataset_list, percentage=1.0, scaling=True,
                                                rotation=True, dev=False)
    print('Copy paste fully augmented dataset prepared: ', len(cp_train_list), ' images.')
    
    cp_only_train_list = cp_train_list[len(train_dataset_list):]
    print('Copy paste only, fully augmented dataset prepared: ', len(cp_only_train_list), ' images.')
    
    scaling_train_list = copy_paste_augmentation(train_file, train_dataset_list, percentage=1.0, scaling=True,
                                                  rotation=False, dev=False)
    scaling_train_list = scaling_train_list[len(train_dataset_list):]
    print('Copy paste only, scaling augmented dataset prepared: ', len(scaling_train_list), ' images.')
    
    rotation_train_list = copy_paste_augmentation(train_file, train_dataset_list, percentage=1.0, scaling=False,
                                                  rotation=True, dev=False)
    rotation_train_list = rotation_train_list[len(train_dataset_list):]
    print('Copy paste only, rotation augmented dataset prepared: ', len(rotation_train_list), ' images.')
    
    print('Test dataset prepared: ', len(test_list), ' images.')
    
    dataset_name_list = ["co_train", "flip_train", "cp_train", \
                    "cp_only_train", "scale_train", "rotation_train", "co_test"]
    
    dataset_list = [
                    train_dataset_list, \
                    flipped_dataset_list, \
                    cp_train_list, \
                    cp_only_train_list, \
                    scaling_train_list, \
                    rotation_train_list, \
                    test_list\
                    ]
    
    for name in dataset_name_list:
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
    
    for idx, name in enumerate(dataset_name_list):
        DatasetCatalog.register(name, lambda dataset_dir=dataset_dir: get_list(dataset_dir, dataset_list[idx]))
    
    for name in dataset_name_list:
        MetadataCatalog.get(name).set(thing_classes=thing_classes)
    
    print('All dataset registered.')
    
    
    test_list = [
            "co_train", \
            "cp_only_train",\
            "scale_train", \
            "rotation_train",\
                "cp_train", \
            "flip_train"
            ]

    for name in test_list:
        train(name, "co_test")
