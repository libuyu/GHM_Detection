# GHM_Detection
The implementation of [Gradient Harmonized Single-stage Detector](https://arxiv.org/abs/1811.05181) published on AAAI 2019 (Oral).

## Loss Functions
* The GHM-C and GHM-R loss functions are available in loss/ghm_loss.py.
* The code works for pytorch 0.4.1 and later version. If you want to run it with pytorch 0.3.x, please checkout to the [pytorch-0.3](https://github.com/libuyu/GHM_Detection/tree/pytorch-0.3) branch.

## Complete Training code
* Complete training code is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and is coming soon.

## License and Citation
The use of this software is RESTRICTED to **non-commercial research and educational purposes**.

```
@article{li2019ghm,
  title={Gradient Harmonized Single-stage Detector},
  author={Buyu Li, Yu Liu, Xiaogang Wang},
  booktitle={AAAI},
  year={2019}
}
```
