import random
import torch
import torchvision.transforms as tfs
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor:
    def __call__(self, pic):
        return F.to_tensor(pic)

    def randomize_parameters(self):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Resize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias="warn"):
        super().__init__()
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)

    def randomize_parameters(self):
        pass

    def __repr__(self) -> str:
        detail = f"(size={self.size}, interpolation={self.interpolation.value}, max_size={self.max_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"


class RandomCrop(torch.nn.Module):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple((size, size))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.i, self.j = -1, -1

    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        if self.i<0 and self.j<0:
            self.i = torch.randint(0, h - th + 1, size=(1,)).item()
            self.j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return self.i, self.j, th, tw
    
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def randomize_parameters(self):
        self.i, self.j = -1, -1


class CenterCrop(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return F.center_crop(img, self.size)

    def randomize_parameters(self):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class TemporalRandomCrop(torch.nn.Module):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        super(TemporalRandomCrop, self).__init__()
        self.size = size

    def forward(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out
    
    def randomize_parameters(self):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.size})"


class RandomHorizontalFlip(torch.nn.Module):

    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.p = p
        self.rd = torch.rand(1)


    def forward(self, img):
        if self.rd < self.p:
            return F.hflip(img)
        return img
    
    def randomize_parameters(self):
        self.rd = torch.rand(1)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class Normalize():
    def __init__(self, mean, std, inplace=False):
        
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def randomize_parameters(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def SpatialTransform(mode, size, crop_size, mean, std):
    assert mode in ('train', 'val', 'test')

    if mode == 'train':
        return Compose([
            Resize(size),
            RandomCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean, std),
        ])
    else:
        return Compose([
            Resize(crop_size),
            # CenterCrop(crop_size),
            ToTensor(),
            Normalize(mean, std),
        ])
