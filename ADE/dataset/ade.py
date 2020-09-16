import json
import logging
import os
import numpy as np

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog

logger = logging.getLogger(__name__)

def load_ade_instances(json_file, image_root, dataset_type):
    """
    Args:
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
        dataset_type (str): type of this dataset. One of base_train/base_val/novel/novel_test.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format.
    """
    timer = Timer()
    ade_file = json.load(open(json_file, 'r'))
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    dataset_dicts = []

    for item in ade_file:
        record = {}
        record['file_name'] = os.path.join(image_root, item['fpath_img'])
        record['height'] = item['height']
        record['width'] = item['width']
        record['image_id'] = item['index']

        record['annotations'] = []
        K = len(item['anchors'])
        proposal_boxes = np.zeros((K, 4))
        record['proposal_bbox_mode'] = BoxMode.XYXY_ABS
        proposal_objectness_logits = np.zeros((K,))

        for i, anchor in enumerate(item['anchors']):
            anno = {}
            anno['category_id'] = anchor['label']
            x1, x2, y1, y2 = anchor['anchor']
            proposal_boxes[i] = [x1, y1, x2, y2]

            if dataset_type == 'base_train':
                # NOTE: In standard detection task, x1, x2, y1, y2 in the
                # next line should be anchor['bbox'], which is the ground
                # truth position for instances. But in our task, we will
                # conduct classification directly on proposals, so we set
                # the anno['bbox'] the same as proposal. We will use the
                # information of ground truth bbox as a supervision
                x1, x2, y1, y2 = anchor['anchor']
                anno['bbox'] = [float(p) for p in  [x1, y1, x2, y2]]
                anno['bbox_mode'] = BoxMode.XYXY_ABS
                anno['attr'] = anchor['attr']
                anno['hierarchy'] = anchor['hierarchy']
                anno['part'] = anchor['part']

            record['annotations'].append(anno)

        record['proposal_boxes'] = proposal_boxes
        record['proposal_objectness_logits'] = proposal_objectness_logits

        ## NOTE: if you want to use segmentation, remove _debug in the next line
        if dataset_type == 'base_train_debug':
            record['sem_seg_file_name'] = os.path.join(image_root, item['seg'])
            record['scene'] = item['scene']



        dataset_dicts.append(record)

    return dataset_dicts

_PREDEFINED_SPLITS_ADE = {}
_PREDEFINED_SPLITS_ADE["base_train"] = {
    "ADE_base_train": ("ADE20K_2016_07_26/", "ADE20K_2016_07_26/base_img_train.json"),
}
_PREDEFINED_SPLITS_ADE["base_val"] = {
    "ADE_base_val": ("ADE20K_2016_07_26/", "ADE20K_2016_07_26/base_img_val.json"),
}
_PREDEFINED_SPLITS_ADE["novel"] = {
    "ADE_novel_train": ("ADE20K_2016_07_26/", "ADE20K_2016_07_26/novel_img_train.json"),
    "ADE_novel_val": ("ADE20K_2016_07_26/", "ADE20K_2016_07_26/novel_img_val.json"),
}
_PREDEFINED_SPLITS_ADE["novel_test"] = {
    "ADE_novel_test_train": ("ADE20K_2016_07_26/", "ADE20K_2016_07_26/novel_img_test_train.json"),
    "ADE_novel_test_val": ("ADE20K_2016_07_26/", "ADE20K_2016_07_26/novel_img_test_val.json"),
}

def register_all_ade(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_ADE.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            DatasetCatalog.register(key,
                                    lambda x = os.path.join(root, json_file),
                                           y = os.path.join(root, image_root),
                                           z = dataset_name:
                                           load_ade_instances(x, y, z))