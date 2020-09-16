import torch
import torch.nn as nn

from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, ROIHeads, select_foreground_proposals
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage

from .classifier import Classifier, CosClassifier

class ROI_CLS(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg)

        # fmt: off
        self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        classifier_type   = cfg.MODEL.CLASSIFIER.CLASSIFIER_TYPE
        num_classes       = cfg.MODEL.CLASSIFIER.NUM_CLASSES
        in_channels       = cfg.MODEL.CLASSIFIER.IN_CHANNELS
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(self.in_features) == 1

        self.pre_pooler = nn.AvgPool2d(kernel_size=3, stride=1)

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.classifier = None
        if classifier_type == 'linear':
            self.classifier = Classifier(num_classes = num_classes,
                                         in_channels = in_channels,
                                         pooler_resolution = pooler_resolution)
        elif classifier_type == 'cos':
            self.classifier = CosClassifier(num_classes = num_classes,
                                         in_channels = in_channels,
                                         pooler_resolution = pooler_resolution)
        else:
            self.classifier = Classifier(num_classes = num_classes,
                                         in_channels = in_channels,
                                         pooler_resolution = pooler_resolution)


    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """

        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        #gt_labels = [x.gt_classes for x in proposals]
        gt_labels = torch.cat([x.gt_classes for x in proposals], dim = 0)
        #gt_labels = torch.stack([x.gt_classes for x in proposals])

        for k, v in features.items():
            features[k] = self.pre_pooler(v)

        box_features = self.pooler(
            [features[f] for f in self.in_features], proposal_boxes
        )

        M, C, H, W = box_features.shape
        flatten_features = box_features.view(M, -1)


        if self.training:

            loss, acc, category_accuracy = self.classifier([flatten_features, gt_labels])
            storage = get_event_storage()
            storage.put_scalar("acc", acc * 100)
            del features
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features

            return [], {"cls_loss": loss}
        else:
            return flatten_features

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances

def register_roi_cls():
    ROI_HEADS_REGISTRY.register(ROI_CLS)