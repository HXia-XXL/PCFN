import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
from skimage import io


class CDDataset(torch.utils.data.Dataset):
    def __init__(self, IMG1_dir, IMG2_dir,
                 Label1_dir, Label2_dir, aug=1):
        self.IMG1_dir = IMG1_dir
        self.IMG2_dir = IMG2_dir

        self.Label1_dir = Label1_dir
        self.Label2_dir = Label2_dir

        # if aug>0, augmentation, if not , not augmentation
        self.aug = aug
        # 所有图片的绝对路径
        self.Names = os.listdir(IMG1_dir)

    def __len__(self):
        return len(self.Names)

    def __getitem__(self, idx):
        img1_name = os.path.join(self.IMG1_dir, self.Names[idx])
        img2_name = os.path.join(self.IMG2_dir, self.Names[idx])
        label_binary1_name = os.path.join(self.Label1_dir, self.Names[idx])
        label_binary2_name = os.path.join(self.Label2_dir, self.Names[idx])

        img1 = cv2.imread(img1_name, 1)
        img2 = cv2.imread(img2_name, 1)
        label_binary1 = cv2.imread(label_binary1_name, 0)
        label_binary2 = cv2.imread(label_binary2_name, 0)

        image_data1, label_data1, image_data2, label_data2 = data_augmentation(img1,
                                                                               label_binary1,
                                                                               img2,
                                                                               label_binary2)

        label_onehot1 = mask2onehot(label_data1, 7)
        label_onehot2 = mask2onehot(label_data2, 7)

        label_data1 = label_data1.squeeze()
        label_data2 = label_data2.squeeze()
        label_onehot1 = label_onehot1.squeeze()
        label_onehot2 = label_onehot2.squeeze()


        sample = {'image1': image_data1, 'image2': image_data2,
                  'label_binary1': label_data1, 'label_binary2': label_data2,
                  'onehot1': label_onehot1, 'onehot2': label_onehot2,
                  }

        # image_data = np.concatenate([image_data1, image_data2], axis=0)
        # # cat img1 and img2
        # sample = {'image': image_data,
        #           'label_binary1': label_data1, 'label_binary2': label_data2,
        #           'onehot1': label_onehot1, 'onehot2': label_onehot2}
        return sample

        # def __init__(self, h5_Dataset):
        #     self.img1 = np.array(h5_Dataset['image1'])
        #     self.img2 = np.array(h5_Dataset['image2'])
        #     self.label1 = np.array(h5_Dataset['label1'])
        #     self.label2 = np.array(h5_Dataset['label2'])
        #
        #     # elf.length = self.img1[:, :, :, :].shape[0]
        #
        # def __len__(self):
        #     return self.img1[:, :, :, :].shape[0]
        #
        # def __getitem__(self, item):
        #     image1 = self.img1[item, :, :, :]
        #     image2 = self.img2[item, :, :, :]
        #     label_binary1 = self.label1[item, :, :, :]
        #     label_binary2 = self.label2[item, :, :, :]
        #
        #     image_data1, label_data1, image_data2, label_data2 = data_augmentation(image1,
        #                                                                            label_binary1,
        #                                                                            image2,
        #                                                                            label_binary2)
        #
        #     label_onehot1 = mask2onehot(label_data1, 7)
        #     label_onehot2 = mask2onehot(label_data2, 7)
        #
        #     label_data1 = label_data1.squeeze()
        #     label_data2 = label_data2.squeeze()
        #     label_onehot1 = label_onehot1.squeeze()
        #     label_onehot2 = label_onehot2.squeeze()
        #
        #     image_data = np.concatenate([image_data1, image_data2], axis=0)
        #
        #     # sample = {'image1': image_data1, 'image2': image_data2,
        #     #           'label_binary1': label_data1, 'label_binary2': label_data2,
        #     #           'onehot1': label_onehot1, 'onehot2': label_onehot2}
        #
        #     # cat img1 and img2
        #     sample = {'image': image_data,
        #               'label_binary1': label_data1, 'label_binary2': label_data2,
        #               'onehot1': label_onehot1, 'onehot2': label_onehot2}
        #     return sample



def data_augmentation(image_data1, label_data1,
                      image_data2, label_data2):
    """
    args:
        image_data : ndarray in [img_rows, img_cols]
        label_data : ndarray in [img_rows, img_cols]
    return image_data and label_data with flip, rotation, blur
    """
    aug_seed_1 = np.random.randint(0, 2)  # [0,1]
    aug_seed_2 = np.random.randint(0, 2)  # [0,1]
    aug_seed_3 = np.random.randint(0, 5)  # [0,4]
    if aug_seed_1 == 1:  # rotation
        rotation_seed = np.random.randint(1, 4)  # [1,3]
        rows, cols = image_data1.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_seed * 90, 1)
        image_data1 = cv2.warpAffine(image_data1, M, (cols, rows))
        label_data1 = cv2.warpAffine(label_data1, M, (cols, rows))
        image_data2 = cv2.warpAffine(image_data2, M, (cols, rows))
        label_data2 = cv2.warpAffine(label_data2, M, (cols, rows))
    if aug_seed_2 == 1:  # flip
        flip_seed = np.random.randint(-1, 2)  # [-1,1]
        image_data1 = cv2.flip(image_data1, flip_seed)
        label_data1 = cv2.flip(label_data1, flip_seed)
        image_data2 = cv2.flip(image_data2, flip_seed)
        label_data2 = cv2.flip(label_data2, flip_seed)
    if aug_seed_3 == 1:  # blur
        image_data1 = cv2.GaussianBlur(image_data1, (3, 3), 0)
        # image_data2 = cv2.GaussianBlur(image_data2, (3, 3), 0)

    image_data1 = np.transpose(image_data1, (2, 0, 1)).astype('float32')  # [c,h,w]
    label_data1 = np.expand_dims(label_data1, 0).astype('float32')  # [c,h,w]
    image_data2 = np.transpose(image_data2, (2, 0, 1)).astype('float32')  # [c,h,w]
    label_data2 = np.expand_dims(label_data2, 0).astype('float32')  # [c,h,w]

    if image_data1.max() > 1:
        image_data1 = image_data1 / 255.0
        image_data2 = image_data2 / 255.0

    return image_data1, label_data1, image_data2, label_data2


class Val_Dataset(torch.utils.data.Dataset):
    def __init__(self, IMG1_dir, IMG2_dir):
        self.IMG1_dir = IMG1_dir
        self.IMG2_dir = IMG2_dir

        self.names = os.listdir(IMG1_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        img1_name = self.IMG1_dir + self.names[item]
        img2_name = self.IMG2_dir + self.names[item]
        img1 = cv2.imread(img1_name, 1)
        img2 = cv2.imread(img2_name, 1)

        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))

        img1 = img1 / 255.0
        img2 = img2 / 255.0

        return img1, img2, self.names[item]


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image1, image2 = sample['image1'], sample['image2']
        label1_binary, label2_binary = sample['label_binary1'], sample['label_binary2']

        h, w = image1.shape[0:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img1 = cv2.resize(image1, (new_h, new_w), interpolation=cv2.INTER_NEAREST)
        img2 = cv2.resize(image2, (new_h, new_w), interpolation=cv2.INTER_NEAREST)
        label1 = cv2.resize(label1_binary, (new_h, new_w), interpolation=cv2.INTER_NEAREST)
        label2 = cv2.resize(label2_binary, (new_h, new_w), interpolation=cv2.INTER_NEAREST)

        return {'image1': img1, 'image2': img2,
                'label_binary1': label1, 'label_binary2': label2}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        image1, image2 = sample['image1'], sample['image2']
        label1_binary, label2_binary = sample['label_binary1'], sample['label_binary2']

        if torch.rand(1) > 0.5:
            h, w = image1.shape[0:2]
            new_h, new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image1 = image1[top: top + new_h,
                     left: left + new_w]
            image2 = image2[top: top + new_h,
                     left: left + new_w]
            label1 = label1_binary[top: top + new_h,
                     left: left + new_w]
            label2 = label2_binary[top: top + new_h,
                     left: left + new_w]

            return {'image1': image1, 'image2': image2,
                    'label_binary1': label1, 'label_binary2': label2}
        else:
            return {'image1': image1, 'image2': image2,
                    'label_binary1': label1_binary, 'label_binary2': label2_binary}


class HorizontalFilp(object):
    """ Flip the image randomly in a sample """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        image1, image2 = sample['image1'], sample['image2']
        label1_binary, label2_binary = sample['label_binary1'], sample['label_binary2']

        if torch.rand(1) < self.p:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)
            label1_binary = cv2.flip(label1_binary, 1)
            label2_binary = cv2.flip(label2_binary, 1)

        return {'image1': image1,
                'image2': image2,
                'label_binary1': label1_binary,
                'label_binary2': label2_binary}


class VerticalFlip(object):
    """ Flip the image randomly in a sample """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        image1, image2 = sample['image1'], sample['image2']
        label1_binary, label2_binary = sample['label_binary1'], sample['label_binary2']

        if torch.rand(1) < self.p:
            image1 = cv2.flip(image1, 0)
            image2 = cv2.flip(image2, 0)
            label1_binary = cv2.flip(label1_binary, 0)
            label2_binary = cv2.flip(label2_binary, 0)

        return {'image1': image1,
                'image2': image2,
                'label_binary1': label1_binary,
                'label_binary2': label2_binary}


class ColorJitter(object):
    """Randomly change the brightness, contrast  of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    """

    def __init__(self, brightness=0, contrast=0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        image1, image2 = sample['image1'], sample['image2']
        label1_binary, label2_binary = sample['label_binary1'], sample['label_binary2']

        if torch.rand(1) > 0.5:
            img_jitter = cv2.addWeighted(image1, self.contrast, image1, 0, self.brightness)

            return {'image1': img_jitter,
                    'image2': image2,
                    'label_binary1': label1_binary,
                    'label_binary2': label2_binary}
        else:
            img_jitter = cv2.addWeighted(image2, self.contrast, image2, 0, self.brightness)

            return {'image1': image1,
                    'image2': img_jitter,
                    'label_binary1': label1_binary,
                    'label_binary2': label2_binary}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        image1, image2 = sample['image1'], sample['image2']
        label1_binary, label2_binary = sample['label_binary1'], sample['label_binary2']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))
        img1 = image1 / 255.0
        img2 = image2 / 255.0
        # label1_onehot = mask2onehot(label1_binary,7)
        # label2_onehot = mask2onehot(label2_binary,7)

        # label1 = torch.from_numpy(label1_binary)
        # label2 = torch.from_numpy(label2_binary)
        # # one-hot mask
        # label1_onehot = torch.nn.functional.one_hot(label1.to(torch.long), 7)
        # label2_onehot = torch.nn.functional.one_hot(label2.to(torch.long), 7)
        #
        # label1_onehot = label1_onehot.permute(2, 0, 1)
        # label2_onehot = label2_onehot.permute(2, 0, 1)

        return {'image1': img1,
                'image2': img2,
                'label_binary1': label1_binary,
                'label_binary2': label2_binary}


class Normalized(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        image1, image2 = sample['image1'], sample['image2']
        label1_binary, label2_binary = sample['label_binary1'], sample['label_binary2']

        img1 = image1 / 255.0
        img2 = image2 / 255.0

        return {'image1': img1,
                'image2': img2,
                'label_binary1': label1_binary,
                'label_binary2': label2_binary}


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask

    colormap = {
        [255, 255, 255],
        [0, 0, 255],
        [128, 128, 128],
        [0, 128, 0],
        [0, 255, 0],
        [128, 0, 0],
        [255, 0, 0]
    }
    class_name = {
        'unchanged',
        'water',
        'groud',
        'vegtation',
        'tree',
        'building',
        'court'
    }
