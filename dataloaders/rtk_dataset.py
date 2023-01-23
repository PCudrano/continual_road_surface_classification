import os
from torch.utils.data import Dataset, Subset
# from utils_dataloader_rtk import *
from dataloaders.augm_rtk import *
import torch

class RtkDataset(Dataset):
    def __init__(self, img_dir, label_list=None, default_augmentation=False, transform=None, target_transform=None, to_device=None, *args, **kwds):
        super(RtkDataset, self).__init__(*args, **kwds)
        self.img_dir = img_dir
        self.to_device = to_device
        if self.to_device:
            self.img = torch.tensor([])
            self.img = self.img[None, None, None, None]
        else:
            self.img_paths = []
            self.augmentation = []  # 'none'|'shaded'|'bright'
        self.targets = []
        self.label_list = label_list
        self.transform = transform
        self.target_transform = target_transform
        self.default_augmentation = default_augmentation
        ext = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
        for root, dirs, files in os.walk(self.img_dir):
            print(root)
            xy = [(os.path.join(root, filename), os.path.basename(root)) for filename in files if filename.lower().endswith(ext)]
            if xy:
                x, y = list(zip(*xy))
                if self.to_device:
                    for i, img_path in enumerate(x):
                        self.img = torch.cat((self.img, self._load_img(img_path, 'none', self.transform).unsqueeze(0)), dim=0)
                        self.augmentation.append('none')
                        self.targets.append(self.label_list[y[i]])
                        # if default_augmentation:
                        #     self.img = torch.stack((self.img, self._load_img(img_path, 'shaded', self.transform).unsqueeze(0)), dim=0)
                        #     self.img = torch.stack((self.img, self._load_img(img_path, 'bright', self.transform).unsqueeze(0)), dim=0)
                        #     self.augmentation.extend(['shaded', 'bright'])
                        #     self.targets.extend([self.label_list[y[i]], self.label_list[y[i]]])
                else:
                    self.img_paths.extend(x)
                    self.targets.extend([self.label_list[label] for label in y] if self.label_list else y)
                    self.augmentation = ['none'] * len(self.targets)
        if self.to_device:
            self.img.to(self.to_device)
        if not self.to_device and self.default_augmentation:
            self._add_default_augmentation(self.img_paths, self.targets)

    #def __init__(self, subsetRtkDataset, default_augmentation=False, transform=None, target_transform=None, *args, **kwds):
    @classmethod
    def fromSubset(cls, subsetRtkDataset, default_augmentation=False, to_device=None):
        if not isinstance(subsetRtkDataset, Subset) or not isinstance(subsetRtkDataset.dataset, RtkDataset):
            raise TypeError("RtkDataset(): subsetRtkDataset is not a Subset[RtkDataset].")

        obj = cls(subsetRtkDataset.dataset.img_dir, label_list=subsetRtkDataset.dataset.label_list,
                  default_augmentation=False, transform=subsetRtkDataset.dataset.transform,
                  target_transform=subsetRtkDataset.dataset.target_transform,
                  to_device=False)
        obj.targets = np.array(obj.targets)[subsetRtkDataset.indices].tolist()
        obj.augmentation = np.array(obj.augmentation)[subsetRtkDataset.indices].tolist()
        img_paths = np.array(obj.img_paths)[subsetRtkDataset.indices].tolist()
        if to_device:
            obj.to_device = to_device
            obj.img = torch.stack([cls._load_img(img_path, 'none', subsetRtkDataset.dataset.transform) for img_path in img_paths], dim=0)
            obj.img.to(to_device)
            # if default_augmentation:
            #     obj.default_augmentation = True
            #     # obj.img = torch.stack([cls._load_img(img_path, 'shaded', subsetRtkDataset.dataset.transform).unsqueeze(0) for img_path in img_paths], dim=0)
                # obj.img = torch.stack([cls._load_img(img_path, 'bright', subsetRtkDataset.dataset.transform).unsqueeze(0) for img_path in img_paths], dim=0)
        else:
            obj.img_paths = img_paths
        if default_augmentation:
            obj.default_augmentation = True
            obj._add_default_augmentation(obj.img_paths, obj.targets)
        return obj

    def __getitem__(self, idx):
        label = self.targets[idx]
        augm = self.augmentation[idx]
        if self.to_device:
            image = self.img[idx,:]
        else:
            img_path = self.img_paths[idx]
            # image = read_image(img_path)
            image = self._load_img(img_path, augm, self.transform)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    @classmethod
    def _load_img(cls, img_path, augmentation, transform):
        image = cv2.imread(img_path)
        if augmentation == 'shaded':
            image = adjust_gamma(image)
        elif augmentation == 'bright':
            image = increase_brightness(image)
        if transform:
            image = transform(image)
        return image

    def __len__(self):
        return len(self.targets)

    def _add_default_augmentation(self, x, y):
        n_data = len(y)
        if self.to_device:
            t = []
            augm1_img = torch.stack([self._load_img(img_path, 'shaded', self.transform) for img_path in x], dim=0)
            augm2_img = torch.stack([self._load_img(img_path, 'bright', self.transform) for img_path in x], dim=0)
            self.img = torch.cat((self.img, augm1_img, augm2_img), dim=0)
            self.img.to(self.to_device)
        else:
            self.img_paths.extend(x.copy() * 2)
        self.targets.extend(y.copy() * 2)
        self.augmentation.extend(['shaded'] * n_data)
        self.augmentation.extend(['bright'] * n_data)
