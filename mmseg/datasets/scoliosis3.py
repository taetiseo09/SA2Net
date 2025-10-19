from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ScoliosisDataset3(CustomDataset):
    """Scoliosis dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('Background', 'Rib', 'Thoracic', 'Lumbar')

    #  red, green, blue
    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

    def __init__(self, split, **kwargs):
        super(ScoliosisDataset3, self).__init__(
            img_suffix='.png', seg_map_suffix='_bone_3.png', split=split, **kwargs)
        # assert osp.exists(self.img_dir) and self.split is not None
