# dicom-enhance
v0.1

## install
```bash
git clone --recurse-submodules --depth 1 xxx
```

>基础环境：CUDA10.0、GCC7.3、Anaconda(py37)、OpenCV和PyTorch。

## notes
git:
```bash
git checkout --orphan latest
git add .
git commit -m "v1.0"
git branch -D master
git branch -m master
git push -f origin master
git branch -u origin/master master

git remote add origin xxx
git remote set-url origin xxx
git push -u origin master:cache
```

bash:
```bash
cd $PROJ_HOME
rm -rf data/coco  # 末尾没有斜杠
ln -s $DATA_ROOT data/coco
cd $DATA_ROOT && rm -rf coco_train.json && ln -s $DATA_TRAIN coco_train.json
cd $DATA_ROOT && rm -rf coco_test.json && ln -s $DATA_TEST coco_test.json
```

## refs

### https://github.com/facebookresearch/fastMRI
- fastMRI

### https://github.com/perone/medicaltorch
- medicaltorch.datasets, medicaltorch.transforms

### https://github.com/NVlabs/noise2noise
- Learning Image Restoration without Clean Data
- Official TensorFlow implementation of the ICML 2018 paper

### https://github.com/ceciliavision/perceptual-reflection-removal
- Single Image Reflection Separation with Perceptual Losses
- https://arxiv.org/abs/1806.05376

### https://github.com/Vandermode/ERRNet
- Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements
- https://arxiv.org/abs/1904.00637

### https://github.com/DmitryUlyanov/deep-image-prior
- 可能无法在某些GPU上收敛。我们亲身经历了Tesla V100和P40 GPU的问题。使用文字修复笔记本最容易检查。尝试设置双精度模式或关闭cudnn。

### https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- CycleGAN and pix2pix in PyTorch

### https://github.com/yunjey/StarGAN
- Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation
- https://arxiv.org/abs/1711.09020

### https://github.com/HsinYingLee/DRIT
- Diverse Image-to-Image Translation via Disentangled Representations
- https://arxiv.org/abs/1905.01270
