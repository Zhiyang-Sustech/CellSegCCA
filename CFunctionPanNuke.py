import skimage.io
import numpy as np
import os
from scipy.io import loadmat

class PanNukeDataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "self_defined", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)


    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    @property
    def image_ids(self):
        return self._image_ids


    def load_dataset(self, data_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes,这里不进行细胞分类，所有都归纳为细胞核
        self.add_class("my_dataset", 1, "nuclei")

        # Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling.For example, use 256, 320, 384, 448, 512, ... etc.

        # # 获取图像张数
        # n_images = np.load(data_path + "\Images\images.npy", allow_pickle=True).shape[0]
        #
        # # Add images
        # for i in range(n_images):
        #     self.add_image("my_dataset", image_id=i, path=data_path)

        # 读取路径下的图像
        images = os.listdir(os.path.join(data_path, "images"))
        # Add images
        for i in range(len(images)):
            self.add_image("my_dataset", image_id=i, path=os.path.join(data_path, "images", images[i]))

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image_path = info['path']
        # image_path = os.path.join(image_path, 'images', '{}.png'.format(os.path.basename(image_path)))
        image = skimage.io.imread(image_path)
        image = image[:, :, :3]
        return image

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        image_path = info['path']
        mask_path = image_path[: image_path.rfind('.')] + ".mat"
        mask_path = mask_path.replace("images", "masks")
        mask_mat = loadmat(mask_path)
        mask_mat = mask_mat['inst_map']
        N = int(np.max(mask_mat))
        mask = np.zeros((mask_mat.shape[0], mask_mat.shape[1], N), dtype=int)
        for i in range(1, N + 1):
            mask[:,:,i-1][mask_mat == i] = 1

        #  这里将所有细胞核全部指定为第二类
        class_ids = np.array([1] * N)
        return mask, class_ids.astype(np.int32)

