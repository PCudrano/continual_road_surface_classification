import torchvision.transforms.functional as TF
import torch
import cv2
import numpy as np


class RtkImageTransform:

    def __init__(self, crop_percentage, new_size=None):
        """
        :param crop_percentage: [top, left, bottom, right] percentage to crop from each side;
                                if negative, crop to fit new_size
        :param new_size: [width, height] final image size
        """
        self.crop_percentage = crop_percentage
        self.new_size = new_size
        if self.crop_percentage[0] < 0 and self.crop_percentage[2] < 0 :
            raise ValueError("RtkImageTransform: invalid crop_percentage argument, top and bottom cannot be both negative")
        if self.crop_percentage[1] < 0 and self.crop_percentage[3] < 0:
            raise ValueError("RtkImageTransform: invalid crop_percentage argument, left and right cannot be both negative")

    def __call__(self, x):
        size_t = torch.tensor(TF.get_image_size(x)).flip(0)  # [height, width]
        crop_perc_nonneg = np.clip(self.crop_percentage, 0, None)
        crop_px_t = (torch.reshape(torch.tensor(crop_perc_nonneg), (2, 2)) * size_t).int() # [[top, left],[bottom, right]]
        new_size = self.new_size if self.new_size else (size_t - torch.sum(crop_px_t, 0)).tolist()  # [height, width]
        top = crop_px_t[0, 0] if self.crop_percentage[0] >= 0 else size_t[0]-crop_px_t[1, 0]-new_size[0]
        left = crop_px_t[0, 1] if self.crop_percentage[1] >= 0 else size_t[1]-crop_px_t[1, 1]-new_size[1]
        bottom = crop_px_t[1, 0] if self.crop_percentage[2] >= 0 else size_t[0]-crop_px_t[0, 0]-new_size[0]
        right = crop_px_t[1, 1] if self.crop_percentage[3] >= 0 else size_t[1]-crop_px_t[0, 1]-new_size[1]
        out = TF.resized_crop(x, top=top, left=left,
                              height=size_t[0]-top-bottom,
                              width=size_t[1]-left-right,
                              size=new_size)
        return out

    # def __call__(self, x):
    #     # img_arr = cv2.resize(x, dsize=(original_shape[1], original_shape[0]))
    #     original_shape = x.shape
    #     bottom_start = int(original_shape[0] * (1 - self.crop_percentage[2]))
    #     img_arr = x[bottom_start - self.crop_percentage[0]: bottom_start, :]
    #     img_arr = cv2.resize(img_arr, dsize=(self.new_size[1], self.new_size[0]))
    #     img_arr = np.uint8(img_arr)
