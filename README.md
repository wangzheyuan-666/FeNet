# FeNet: Feature Enhancement Network for Lightweight Remote-Sensing Image Super-Resolution 

This repository is for FeNet introduced in the following paper

Wang, Zheyuan and Li, Liangliang and Xue, Yuan and Jiang, Chenchen and Wang, Jiawen and Sun, Kaipeng and Ma, Hongbing, "FeNet: Feature Enhancement Network for Lightweight Remote-Sensing Image Super-Resolution" in IEEE Transactions on Geoscience and Remote Sensing, [[paper](https://ieeexplore.ieee.org/document/9759417)]

  


The code is built on IMDN and RCAN  (PyTorch).  See [IMDN](https://github.com/Zheng222/IMDN) and [RCAN](https://github.com/yulunzhang/RCAN) for details


# Model parameters
![performance](https://github.com/wangzheyuan-666/FeNet/blob/main/images/performance.png)

Trade-off between performance and number of parameters on Urban100 ×2 dataset.

# Architecture of FeNet

![FeNet](https://github.com/wangzheyuan-666/FeNet/blob/main/images/FeNet.png)


# Lightweight lattice block

![LLB](https://github.com/wangzheyuan-666/FeNet/blob/main/images/LLB.png)

# Visiual results
![RS_results](https://github.com/wangzheyuan-666/FeNet/blob/main/images/RS_image_results.png)

![natural results](https://github.com/wangzheyuan-666/FeNet/blob/main/images/nature_image_results.png)

Citation
If you find IMDN useful in your research, please consider citing:

@inproceedings{Hui-IMDN-2019,
  title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
  author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
  pages={2024--2032},
  year={2019}
}

@inproceedings{AIM19constrainedSR,
  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
  author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
  booktitle={The IEEE International Conference on Computer Vision (ICCV) Workshops},
  year={2019}
}

