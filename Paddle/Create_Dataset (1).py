import numpy as np
import os
import paddle
import paddle.nn.functional as F
from paddle.io import  Dataset
import cv2



class MutliTask_CDDataset(Dataset):
    def __init__(self, IMG1_dir, IMG2_dir,
                 Label1_dir, Label2_dir, Label_dir, aug=1):
        super(MutliTask_CDDataset,self).__init__()
        self.IMG1_dir = IMG1_dir
        self.IMG2_dir = IMG2_dir

        self.Label1_dir = Label1_dir
        self.Label2_dir = Label2_dir
        self.Label_dir = Label_dir

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
        label_name = os.path.join(self.Label_dir, self.Names[idx])

        img1 = cv2.imread(img1_name, 1)
        img2 = cv2.imread(img2_name, 1)
        label_binary1 = cv2.imread(label_binary1_name, 0)
        label_binary2 = cv2.imread(label_binary2_name, 0)
        label = cv2.imread(label_name, 0)

        image_data1, label_data1, \
        image_data2, label_data2, \
        label = data_augmentation(img1, label_binary1,
                                  img2, label_binary2,
                                  label)

        # label_onehot1 = mask2onehot(label_data1, 5)
        # label_onehot2 = mask2onehot(label_data2, 5)

        # label_data1 = label_data1.squeeze()
        # label_data2 = label_data2.squeeze()
        # label = label.squeeze()

        return image_data1 ,image_data2 ,label_data1,label_data2,label

class MutliTask_CDDataset_HRSCD(Dataset):
    def __init__(self, IMG1_dir, IMG2_dir,
                 Label1_dir, Label2_dir, Label_dir, aug=1):
        super(MutliTask_CDDataset_HRSCD,self).__init__()
        self.IMG1_dir = IMG1_dir
        self.IMG2_dir = IMG2_dir

        self.Label1_dir = Label1_dir
        self.Label2_dir = Label2_dir
        self.Label_dir = Label_dir

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
        label_name = os.path.join(self.Label_dir, self.Names[idx])

        img1 = cv2.imread(img1_name, 1)
        img2 = cv2.imread(img2_name, 1)
        label_binary1 = cv2.imread(label_binary1_name, 0)
        label_binary2 = cv2.imread(label_binary2_name, 0)
        label = cv2.imread(label_name, 0)

        if self.aug == 1:
           
            image_data1, label_data1, \
            image_data2, label_data2, \
            label = data_augmentation_HRSCD(img1, label_binary1,
                                  img2, label_binary2,
                                  label)
        else:
            image_data1 = np.transpose(img1, (2, 0, 1)).astype('float32')  # [c,h,w]
            label_data1 = np.expand_dims(label_binary1, 0).astype('float32')  # [c,h,w]
            image_data2 = np.transpose(img2, (2, 0, 1)).astype('float32')  # [c,h,w]
            label_data2 = np.expand_dims(label_binary2, 0).astype('float32')  # [c,h,w]
            label = np.expand_dims(label, 0).astype('float32')  # [c,h,w]

            if image_data1.max() > 1:
                image_data1 = image_data1 / 255.0
                image_data2 = image_data2 / 255.0
        # label_onehot1 = mask2onehot(label_data1, 5)
        # label_onehot2 = mask2onehot(label_data2, 5)

        # label_data1 = label_data1.squeeze()
        # label_data2 = label_data2.squeeze()
        # label = label.squeeze()

        return image_data1 ,image_data2 ,label_data1,label_data2,label


class DISFN_Dataset(Dataset):
    def __init__(self, IMG1_dir, IMG2_dir,
                 Label1_dir, Label2_dir, Label_dir, aug=1):
        super(DISFN_Dataset,self).__init__()
        self.IMG1_dir = IMG1_dir
        self.IMG2_dir = IMG2_dir

        self.Label1_dir = Label1_dir
        self.Label2_dir = Label2_dir
        self.Label_dir = Label_dir

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
        label_name = os.path.join(self.Label_dir, self.Names[idx])

        img1 = cv2.imread(img1_name, 1)
        img2 = cv2.imread(img2_name, 1)
        label_binary1 = cv2.imread(label_binary1_name, 0)
        label_binary2 = cv2.imread(label_binary2_name, 0)
        label = cv2.imread(label_name, 0)

        image_data1, _, \
        image_data2, _, \
        label = data_augmentation(img1, label_binary1,
                                  img2, label_binary2,
                                  label)

        label = label.squeeze()
        
        label4 = cv2.resize(label, (256,256), interpolation = cv2.INTER_AREA)
        label3 = cv2.resize(label, (128,128), interpolation = cv2.INTER_AREA)
        label2 = cv2.resize(label, (64,64), interpolation = cv2.INTER_AREA)
        label1 = cv2.resize(label, (32,32), interpolation = cv2.INTER_AREA)

        
        # label_onehot1 = mask2onehot(label_data1, 5)
        # label_onehot2 = mask2onehot(label_data2, 5)

        # label_data1 = label_data1.squeeze()
        # label_data2 = label_data2.squeeze()
        # label = label.squeeze()

        return image_data1 ,image_data2,label1,label2,label3,label4,label

def data_augmentation(image_data1, label_data1,
                      image_data2, label_data2,
                      label_data):
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
        label_data = cv2.warpAffine(label_data, M, (cols, rows))
    if aug_seed_2 == 1:  # flip
        flip_seed = np.random.randint(-1, 2)  # [-1,1]
        image_data1 = cv2.flip(image_data1, flip_seed)
        label_data1 = cv2.flip(label_data1, flip_seed)
        image_data2 = cv2.flip(image_data2, flip_seed)
        label_data2 = cv2.flip(label_data2, flip_seed)
        label_data = cv2.flip(label_data, flip_seed)
    if aug_seed_3 == 1:  # blur
        blur_seed = np.random.randint(0, 4)
        if blur_seed < 2:
            image_data1 = cv2.GaussianBlur(image_data1, (3, 3), 0)
        else:
            image_data2 = cv2.GaussianBlur(image_data2, (3, 3), 0)
    if aug_seed_3 == 2:  # zoom
        h, w = image_data1.shape[0:2]
        new_h, new_w = [256, 256]

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image1 = image_data1[top: top + new_h,
                 left: left + new_w]
        image2 = image_data2[top: top + new_h,
                 left: left + new_w]
        label1 = label_data1[top: top + new_h,
                 left: left + new_w]
        label2 = label_data2[top: top + new_h,
                 left: left + new_w]
        label = label_data[top: top + new_h,
                left: left + new_w]

        image_data1 = cv2.resize(image1, (h, w), interpolation=cv2.INTER_LINEAR)
        image_data2 = cv2.resize(image2, (h, w), interpolation=cv2.INTER_LINEAR)
        label_data1 = cv2.resize(label1, (h, w), interpolation=cv2.INTER_NEAREST)
        label_data2 = cv2.resize(label2, (h, w), interpolation=cv2.INTER_NEAREST)
        label_data = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST)

    image_data1 = np.transpose(image_data1, (2, 0, 1)).astype('float32')  # [c,h,w]
    label_data1 = np.expand_dims(label_data1, 0).astype('float32')  # [c,h,w]
    image_data2 = np.transpose(image_data2, (2, 0, 1)).astype('float32')  # [c,h,w]
    label_data2 = np.expand_dims(label_data2, 0).astype('float32')  # [c,h,w]
    label_data = np.expand_dims(label_data, 0).astype('float32')  # [c,h,w]

    if image_data1.max() > 1:
        image_data1 = image_data1 / 255.0
        image_data2 = image_data2 / 255.0

    return image_data1, label_data1, image_data2, label_data2, label_data

def data_augmentation_HRSCD(image_data1, label_data1,
                      image_data2, label_data2,
                      label_data):
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
        label_data = cv2.warpAffine(label_data, M, (cols, rows))
    if aug_seed_2 == 1:  # flip
        flip_seed = np.random.randint(-1, 2)  # [-1,1]
        image_data1 = cv2.flip(image_data1, flip_seed)
        label_data1 = cv2.flip(label_data1, flip_seed)
        image_data2 = cv2.flip(image_data2, flip_seed)
        label_data2 = cv2.flip(label_data2, flip_seed)
        label_data = cv2.flip(label_data, flip_seed)

    image_data1 = np.transpose(image_data1, (2, 0, 1)).astype('float32')  # [c,h,w]
    label_data1 = np.expand_dims(label_data1, 0).astype('float32')  # [c,h,w]
    image_data2 = np.transpose(image_data2, (2, 0, 1)).astype('float32')  # [c,h,w]
    label_data2 = np.expand_dims(label_data2, 0).astype('float32')  # [c,h,w]
    label_data = np.expand_dims(label_data, 0).astype('float32')  # [c,h,w]

    if image_data1.max() > 1:
        image_data1 = image_data1 / 255.0
        image_data2 = image_data2 / 255.0

    return image_data1, label_data1, image_data2, label_data2, label_data


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.2):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def TTA(model, image1, image2, label1=None, label2=None, num_class=7):
    '''
    Test Time Augmentor
    args:
    model: trained model
    image1: T1 image
    image2: T2 image
    label1: T1 groud truth
    label2: T2 groud truth

    return label1_average,label2_average
    '''
    flip = [-1, 0, 1]
    [batch, channel, height, width] = image1.shape
    label1_average = paddle.zeros([batch, num_class, height, width], dtype=float).cuda().float()
    label2_average = paddle.zeros([batch, num_class, height, width], dtype=float).cuda().float()

    np_image1 = image1.cpu().numpy()
    np_image1 = np.transpose(np_image1, (0, 2, 3, 1))
    np_image2 = image1.cpu().numpy()
    np_image2 = np.transpose(np_image2, (0, 2, 3, 1))

    output1, output2 = model(image1, image2)
    output1 = F.softmax(output1, axis=1)
    output2 = F.softmax(output2, axis=1)

    label1_average += output1
    label2_average += output2
    for flip_seed in flip:
        image_data1 = tensorFlip(image1, flip_seed)
        image_data2 = tensorFlip(image2, flip_seed)

        output1, output2 = model(image_data1, image_data2)
        output1 = F.softmax(output1, axis=1)
        output2 = F.softmax(output2, axis=1)

        label1_average += tensorFlip(output1, flip_seed)
        label2_average += tensorFlip(output2, flip_seed)

    return label1_average, label2_average


# using cv2.flip  flip  tensor data
def tensorFlip(tensor, flip_seed):
    [batch, channel, height, width] = tensor.shape

    np_tensor = tensor.cpu().numpy()
    np_tensor = np.transpose(np_tensor, (0, 2, 3, 1))

    image = np.zeros([batch, height, width, channel], dtype=float)

    for i in range(batch):
        image[i, :, :, :] = cv2.flip(np_tensor[i, :, :, :], flip_seed)

    image = paddle.numpy(np.transpose(image, (0, 3, 1, 2))).cuda().float()

    return image





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
