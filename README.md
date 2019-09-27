CCF-BDCI2019-Multi-person-Face-Recognition-Competition-Baseline
===============

:sparkles:赛题链接：[CCF BDCI2019 多人种人脸识别](https://www.datafountain.cn/competitions/348)

这份代码主要是基于PyTorch框架实现，利用 MageFace 训练的预训练模型，来对赛题的测试集进行测试。

应用的模型是 InsightFace： Addittive Angular Margin Loss for Deep Face Recognition.[paper](https://arxiv.org/pdf/1801.07698.pdf)

-----------------------------------

:running: 复现步骤
-----

## :one: 下载代码

```
git clone https://github.com/LcenArthas/CCF-BDCI2019-Multi-person-Face-Recognition-Competition-Baseline.git
```

## :two: 配置环境

 - Ubantu16.04

 - Python 3.6

 - PyTorch 1.0.0

## :three: 配置测试集文件

把测试集的图片文件夹 `/test/` 置于根目录，把提交例样 `submission_template.csv` 同样置于根目录

## :four: 下载预训练模型

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
