# -*- coding:utf-8 -*-
import numpy as np



class ConfusionMatrix():
    def __init__(self, num_classes=2, streaming=False):
        # confusion_matrix
        self.hist = np.zeros([num_classes, num_classes],
                                         dtype='int64')
        self.num_classes = num_classes
        self.streaming = streaming

    def fast_hist(self, a, b, n):
        k = (a >= 0) & (a < n) & (b>=0) & (b<n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

    def get_hist(self, image, label):
        self.hist += self.fast_hist(image.flatten(), label.flatten().astype(int), self.num_classes)

    def cal_kappa(self):
        if self.hist.sum() == 0:
            po = 0
            pe = 1
            kappa = 0
        else:
            po = np.diag(self.hist).sum() / self.hist.sum()
            pe = np.matmul(self.hist.sum(1), self.hist.sum(0).T) / self.hist.sum() ** 2
            if pe == 1:
                kappa = 0
            else:
                kappa = (po - pe) / (1 - pe)
        return kappa

    def calculate(self, output, GT):
        num_data = output.shape[0]

        if self.streaming == False:
            self.hist = np.zeros([self.num_classes, self.num_classes],dtype='int64')

        for index in range(num_data):
            infer_array = output[index, :, :]
            label_array = GT[index, :, :]

            self.get_hist(infer_array, label_array)
            

    def Evaluate_revised(self):

        hist_fg = self.hist[1:, 1:]
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = self.hist[0][0]
        c2hist[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        c2hist[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        c2hist[1][1] = hist_fg.sum()

        kappa = self.cal_kappa()
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist) + 1e-8)
        IoU_fg = iu[1]
        # IoU = (iu[0] + iu[1]) / 2

        # miou
        iou = np.diag(self.hist) / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist) + 1e-8)
        mIoU = np.nanmean(iou)
        # mF1
        F1_pre = 2*np.diag(self.hist) / (self.hist.sum(1) + self.hist.sum(0) + 1e-8)
        mF1 = np.nanmean(F1_pre)
        # F1 
        F1 = 2*c2hist[1][1]/(c2hist[0][1]+c2hist[1][1]+c2hist[1][0]+c2hist[1][1])
        # accuracy
        sum_samples = np.sum(self.hist)
        acc_samples = 0
        for i in range(self.num_classes):
            acc_samples += self.hist[i][i]
        accuracy = acc_samples / sum_samples
        
        
        return F1, mF1, accuracy, F1_pre
