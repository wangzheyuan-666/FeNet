# FeNet: Feature Enhancement Network for Lightweight Remote-Sensing Image Super-Resolution 

This repository is for FeNet introduced in the following paper

Wang, Zheyuan and Li, Liangliang and Xue, Yuan and Jiang, Chenchen and Wang, Jiawen and Sun, Kaipeng and Ma, Hongbing, "FeNet: Feature Enhancement Network for Lightweight Remote-Sensing Image Super-Resolution" in IEEE Transactions on Geoscience and Remote Sensing, [[paper](https://ieeexplore.ieee.org/document/9759417)]

  


The code is built on IMDN and RCAN  (PyTorch).  See [IMDN](https://github.com/Zheng222/IMDN) and [RCAN](https://github.com/yulunzhang/RCAN) for details



Runing testing:

# Set5 x2 IMDN
python test_IMDN.py --test_hr_folder Test_Datasets/Set5/ --test_lr_folder Test_Datasets/Set5_LR/x2/ --output_folder results/Set5/x2 --checkpoint checkpoints/IMDN_x2.pth --upscale_factor 2
# RealSR IMDN_AS
python test_IMDN_AS.py --test_hr_folder Test_Datasets/RealSR/ValidationGT --test_lr_folder Test_Datasets/RealSR/ValidationLR/ --output_folder results/RealSR --checkpoint checkpoints/IMDN_AS.pth
Calculating IMDN_RTC's FLOPs and parameters, input size is 240*360
