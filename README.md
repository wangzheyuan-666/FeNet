# FeNet: Feature Enhancement Network for Lightweight Remote-Sensing Image Super-Resolution 

This repository is for FeNet introduced in the following paper

Wang, Zheyuan and Li, Liangliang and Xue, Yuan and Jiang, Chenchen and Wang, Jiawen and Sun, Kaipeng and Ma, Hongbing, "FeNet: Feature Enhancement Network for Lightweight Remote-Sensing Image Super-Resolution" in IEEE Transactions on Geoscience and Remote Sensing, [[paper](https://ieeexplore.ieee.org/document/9759417)]

  


The code is built on IMDN and RCAN  (PyTorch).  See [IMDN](https://github.com/Zheng222/IMDN) and [RCAN](https://github.com/yulunzhang/RCAN) for details


# Model parameters
![performance]()
Trade-off between performance and number of parameters on Urban100 ×2 dataset.

Running time

Trade-off between performance and running time on Set5 ×4 dataset. VDSR, DRCN, and LapSRN were implemented by MatConvNet, while DRRN, and IDN employed Caffe package. The rest EDSR-baseline, CARN, and our IMDN utilized PyTorch.

Adaptive Cropping

The diagrammatic sketch of adaptive cropping strategy (ACS). The cropped image patches in the green dotted boxes.

Visualization of feature maps

Visualization of output feature maps of the 6-th progressive refinement module (PRM).

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

