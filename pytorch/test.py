from Model import *
from Create_Dataset import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = CDNet_Resnet().to(device)
    net.train()
    net.load_state_dict(torch.load('./log_files/batch4_epoch100_Focal_Loss_params_ResnetBackbone_Dilate.pkl'))

    path1 = './data/change_detection_val/val/im1/'
    path2 = './data/change_detection_val/val/im2/'
    dst_path1 = './data/change_detection_val/model_output5_adabn/im1/'
    dst_path2 = './data/change_detection_val/model_output5_adabn/im2/'

    valDataset = Val_Dataset(path1, path2)

    valLoader = DataLoader(valDataset, batch_size=4, num_workers=2)
    with torch.no_grad():
        for i, (image1, image2, name) in enumerate(valLoader):
            image1 = image1.to(device).float()
            image2 = image2.to(device).float()
            print('file_name:', name)

            output1, output2 = net(image1, image2)

            img1 = torch.argmax(output1, dim=1)
            img2 = torch.argmax(output2, dim=1)
            # print(img1)

            img1 = np.array(img1.cpu())
            img2 = np.array(img2.cpu())
            index = 0
            for names in name:
                cv2.imwrite(dst_path1 + names, img1[index, :, :])
                cv2.imwrite(dst_path2 + names, img2[index, :, :])
                index += 1
