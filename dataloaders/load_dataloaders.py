import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from dataloaders.rtk_transform import RtkImageTransform
from dataloaders.rtk_dataset import RtkDataset


def load_dataloader_rtk_paper(dataset_path, labels_dict, cropping_percentage, cropped_size, train_split, valid_split, batch_size, to_device=None, **kwargs):
    # define crop transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    RtkImageTransform(crop_percentage=cropping_percentage,
                                                      new_size=cropped_size)])  # transforms.ConvertImageDtype(torch.float)])
    # create dataset
    ds = RtkDataset(dataset_path, labels_dict, transform=transform)
    # split train,valid,test
    n = len(ds)
    split_n = [round(train_split * n), round(valid_split * n),
               n - round(train_split * n) - round(valid_split * n)]
    _train_ds, valid_ds, test_ds = \
        torch.utils.data.random_split(ds, split_n)
    # apply augmentation to train set
    train_ds = RtkDataset.fromSubset(_train_ds, default_augmentation=True, to_device=to_device)
    # create dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kwargs)  # , collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **kwargs)  # , collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kwargs)  # ,collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    return train_dataloader, valid_dataloader, test_dataloader, train_ds, valid_ds, test_ds
