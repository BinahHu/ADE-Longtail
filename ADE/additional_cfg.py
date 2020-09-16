from detectron2.config import CfgNode as CN

def set_additional_cfg(cfg):
    cfg.DATASETS.ADE_ROOT = "/home/zpang/"

    cfg.INPUT.RESIZE_SHORT = 800
    cfg.INPUT.RESIZE_LONG = 1504

    cfg.MODEL.LOAD_PROPOSALS_IN_DATA_DICT = True
    cfg.MODEL.CLASSIFIER = CN()
    cfg.MODEL.CLASSIFIER.CLASSIFIER_TYPE = 'linear'
    cfg.MODEL.CLASSIFIER.NUM_CLASSES = 189
    cfg.MODEL.CLASSIFIER.IN_CHANNELS = 512

    return cfg