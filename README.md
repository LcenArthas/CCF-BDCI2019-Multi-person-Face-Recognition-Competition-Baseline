CCF-BDCI2019-Multi-person-Face-Recognition-Competition-Baseline
===============

:sparkles:赛题链接：[CCF BDCI2019 多人种人脸识别](https://www.datafountain.cn/competitions/348)

这份代码主要是基于PyTorch框架实现，利用 MageFace 训练的预训练模型，来对赛题的测试集进行测试。

应用的模型是 InsightFace： Addittive Angular Margin Loss for Deep Face Recognition.[paper](https://arxiv.org/pdf/1801.07698.pdf)

-----------------------------------

:yum: UPDATE Oct. 8
-----
增加整合了训练部分代码即复现步骤

------------------------------------

:running: 准备工作
-----

## :one: 下载代码

```
git clone https://github.com/LcenArthas/CCF-BDCI2019-Multi-person-Face-Recognition-Competition-Baseline.git
```

## :two: 配置环境

 - Ubantu16.04

 - Python 3.6

 - PyTorch 1.0.0
 
 ------------------------------------------------------------
 
:sparkles: 训练部分
--------

## :one: 准备数据

:small_orange_diamond: 在根目录下创建文件夹 `/train_data/`，并将训练文件夹 `/training/` 放入其中：

```
mkdir train_data
```

:small_orange_diamond: 准备数据：

```
cd data
python pre_data.py
```

## :two: 开始训练

```
cd data
python pre_data.py
```

----------------------------------------------------

:sparkles: 测试提交部分
--------

## :one: 配置测试集文件

把测试集的图片文件夹 `/test/` 置于根目录，把提交例样 `submission_template.csv` 同样置于根目录

## :two: 下载预训练模型

- [SE-LResNet101E-IR](https://pan.baidu.com/s/1XHUkFgRvyhmnyf8p101v2Q) 

下载好的模型置于根目录

------------------------------------------------------

:clap: 开始！
--------

```
python test_ccf.py
```

10分钟左右，输出做后的提交结果 `submission_new.csv`

:smile:希望对小伙伴们有所启发！欢迎Star！
-------------
