
# GHM_Detection
The implementation of [Gradient Harmonized Single-stage Detector](https://arxiv.org/abs/1811.05181) published on AAAI 2019 (**Oral**).

## Updates
(May 24, 2019)
- Make mmdetection a submodule to keep it up-to-date.

## Installation
This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection). 

### Requirements
- Python 3.5+
- PyTorch 1.0+ 
  (Based on the current version of mmdetection)
- CUDA 9.0+

### Setup the Environment and Packages
i. Create a new environment
We recommend Anaconda as the package & environment manager. And here is an example:
```shell
conda create -n ghm
conda activate ghm
```

ii. Install PyTorch
Follow the [official instructions](https://pytorch.org/) to install Pytorch. Here is an example using conda:
```shell
conda install pytorch torchvision -c pytorch
```
iii. Install Cython
```shell
conda install cython 
# or "pip install cython"
```

### Install GHM
i. Clone the repository
```shell
git clone --recursive https://github.com/libuyu/GHM_Detection.git
```

ii. Compile extensions
```
cd GHM_Detection/mmdetection

./compile.sh
```

iii. Setup mmdetection
```
pip install -e . 
# editable mode is convinient when debugging
# if your code in mmdetection is fixed, use "pip install ." directly
```

### Prepare Data
It is recommended to symlink the datasets root to `mmdetection/data`.
```
ln -s $YOUR_DATA_ROOT data
```
The directories should be arranged like this:
```
GHM_detection
├──	mmdetection
|	├── mmdet
|	├── tools
|	├── configs
|	├── data
|	│   ├── coco
|	│   │   ├── annotations
|	│   │   ├── train2017
|	│   │   ├── val2017
|	│   │   ├── test2017
|	│   ├── VOCdevkit
|	│   │   ├── VOC2007
|	│   │   ├── VOC2012
```


## Running
### Script
We provide training and testing scripts and configuration files for both GHM and baseline (focal loss and smooth L1 loss) in the [experiments](https://github.com/libuyu/GHM_Detection/tree/master/experiments) directory. You need specify the path of your own pre-trained model in the config files.

### Configuration
The configuration parameters are mainly in the cfg_*.py files. The parameters you most probably change are as follows:

- *work_dir*: the directory for current experiment
- *datatype*: data set name (coco, voc, etc.)
- *data_root*: Root for the data set
- *model.pretrained*: the path to the ImageNet pretrained backbone model
- *resume_from*: path or checkpoint file if resume
- *train_cfg.ghmc*: params for GHM-C loss
	- *bins*: unit region numbers
	- *momentum*: moving average parameter \alpha
- *train_cfg.ghmr*: params for GHM-R loss
	- *mu*: the \mu for ASL1 loss
	- *bins*, *momentum*: similar to ghmc 
- *total_epochs*, *lr_config.step*: set the learning rate decay strategy

### Loss Functions
* The GHM-C and GHM-R loss functions are available in [ghm_loss.py](https://github.com/libuyu/mmdetection/blob/master/mmdet/models/losses/ghm_loss.py).
* The code works for pytorch 1.0.1 and later version.

## Result

Training using the Res50-FPN backbone and testing on COCO minival.

Method | AP
-- | --
FL + SL1 | 35.6%
GHM-C + SL1 | 35.8%
GHM-C + GHM-R | 37.0%

## License
This project is released under the [MIT license](https://github.com/libuyu/GHM_Detection/blob/master/LICENSE).

## Citation
```
@inproceedings{li2019gradient,
  title={Gradient Harmonized Single-stage Detector},
  author={Li, Buyu and Liu, Yu and Wang, Xiaogang},
  booktitle={AAAI Conference on Artificial Intelligence},
  year={2019}
}
```
If the code helps you in your research, please also cite:
```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li,
             Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, Zheng Zhang, Dazhi Cheng,
             Chenchen Zhu, Tianheng Cheng, Qijie Zhao, Buyu Li, Xin Lu, Rui Zhu, Yue Wu,
             Jifeng Dai, Jingdong Wang, Jianping Shi, Wanli Ouyang, Chen Change Loy, Dahua Lin},
  journal = {arXiv preprint arXiv:1906.07155},
  year    = {2019}
}
```
