# GHM_Detection
The implementation of [Gradient Harmonized Single-stage Detector](https://arxiv.org/abs/1811.05181) published on AAAI 2019 **(Oral**).

## Loss Functions
* The GHM-C and GHM-R loss functions are available in [ghm_loss.py](https://github.com/libuyu/GHM_Detection/mmdetection/mmdet/core/loss/ghm_loss.py).
* The code works for pytorch 0.4.1 and later version. If you want to run it with pytorch 0.3.x, please checkout to the [pytorch-0.3](https://github.com/libuyu/GHM_Detection/tree/pytorch-0.3) branch.

## Training Code
* The main training code is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please see [this](https://github.com/open-mmlab/mmdetection/blob/master/INSTALL.md) for installation issues. 
* We provide training and testing scripts and configuration files for both GHM and baseline (focal loss and smooth L1 loss) in the [experiments](https://github.com/libuyu/GHM_Detection/experiments) directory.

## License and Citation
The use of this code is RESTRICTED to **non-commercial research and educational purposes**.

```
@article{li2019ghm,
  title={Gradient Harmonized Single-stage Detector},
  author={Buyu Li, Yu Liu, Xiaogang Wang},
  booktitle={AAAI},
  year={2019}
}
```
