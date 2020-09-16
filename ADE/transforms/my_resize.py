from PIL import Image

from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.transform import ResizeTransform


class MyResize(Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, long_edge_length, interp=Image.BILINEAR
    ):
        """
        Args:
            short_edge_length (int):
            long_edge_length (int):
        """
        super().__init__()

        self.short = short_edge_length
        self.long = long_edge_length
        self.interp = interp

        self._init(locals())

    def get_transform(self, image):
        h, w = image.shape[:2]

        if h < w:
            newh, neww = self.short, self.long
        else:
            newh, neww = self.long, self.short

        return ResizeTransform(h, w, newh, neww, self.interp)
