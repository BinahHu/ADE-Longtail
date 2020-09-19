# import some common libraries

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultTrainer
from detectron2.evaluation import DatasetEvaluators



# import ADE related package
from dataset.ade import register_all_ade
from dataset.my_mapper import MyDatasetMapper
from transforms.my_resize import MyResize
from modeling.backbone.my_build import register_my_backbone
from modeling.roi_heads.roi_cls import register_roi_cls
from modeling.meta_arch.my_rcnn import register_my_rcnn
from additional_cfg import set_additional_cfg

class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        """
                Returns:
                    iterable

                It now calls :func:`detectron2.data.build_detection_train_loader`.
                Overwrite it if you'd like a different data loader.
                """
        return build_detection_train_loader(cfg, mapper=MyDatasetMapper(cfg, is_train=True, augmentations=[
        MyResize(cfg.INPUT.RESIZE_SHORT, cfg.INPUT.RESIZE_LONG)]))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        evaluator_list = []
        # TODO: implement evaluator

        return DatasetEvaluators(evaluator_list)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg = set_additional_cfg(cfg)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def register_all(cfg):
    register_all_ade(cfg.DATASETS.ADE_ROOT)
    register_my_backbone()
    register_roi_cls()
    register_my_rcnn()

def main(args):
    cfg = setup(args)

    register_all(cfg)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
