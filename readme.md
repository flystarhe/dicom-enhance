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
git push --set-upstream origin master

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
