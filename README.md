# Deep Siamese PostClassfication Fusion Network (PCFN)
This is the implementation of Deep Siamese Post Classfication Fusion Network for Semantic Change Detection in Multi-temporal Remote Sensing Images

![](./PCFN.png "fig")

We provide Paddle & Pytorch codes of PCFN.

## Paddle 
version: PaddlePaddle 2.0.1 <br>
Batch size: 8<br>
Optimizer: SGD <br>
Lr： 0.007 <br>
Schedule： ReduceOnPlateau (Factor=0.3) <br>


## Pytorch
We only tested SCN (the sub-network of PCFN) in the compeitition hosted by SenseTime in 2020 (Ranked 9th)

Batch size: 4<br>
Optimizer: SGD <br>
Lr： 0.007 <br>
Schedule： ReduceOnPlateau (Factor=0.1) <br>
Test time augmentation (TTA) <br>
Dilation Block <br>

Loss:<br>
![image](https://user-images.githubusercontent.com/44633898/163906521-8e089ed5-79ac-4a2d-adf5-52930174b41e.png)<br>
Note that the class weights need to be set inversely proportional to the number of training examples or manually in $L_\text{SCD}$


## Datasets
We merge the land cover categories in SECOND dataset, the processed dataset can be available in [BaiduDrive](链接地址 "（可选）添加一个标题"). <br>
For the origianl SECOND dataset, please contact CAPTAIN-WHU. <br>
Since images of HRSCD cannot be distributed in the network, we could not share the HRSCD dataset. If you need, please contact Dr.Daudt.

## Future Work
1. integrate thematic indices ([RGBi](https://rdrr.io/cran/uavRst/man/rgb_indices.html "（可选）添加一个标题") or NIRi) with CNN in arieal images.<br>
e.g. <br> Vegtation Indice (VI) in RGB images <br>
![image](https://user-images.githubusercontent.com/44633898/163909549-8c9d4ea5-1bc4-476d-8414-99efa4146ac2.png)<br>
![image](https://user-images.githubusercontent.com/44633898/163909477-3cf51fff-fcf0-41aa-b2f6-5cf948d2a249.png)

2. the semi-supervised or unsupervised design of soft fusion strategy with CNN.
3. the method of 1st winner in SenseTime 2020 RS competition is excellent.
