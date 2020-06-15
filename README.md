## 基于paddlepaddle的FCOS实现

比赛链接：https://tianchi.aliyun.com/competition/entrance/231748/introduction?spm=5176.12281925.0.0.3b957137LONgSm

用paddlepaddle的原因就是可以在百度的AI studio上运行（免费的Tesla V100显卡）。

最开始AI Studio是支持pytorch和Tensorflow的，后来和谐掉了（pip install pytorch!!）



**代码都是刚入门时练手用的，比较简单易懂。但挺多地方写死了，不好改。**

[Encode部分](https://github.com/zehua9500/FCOS-paddlepaddle/blob/master/code/Encode.py)

在代码上，FCOS没有Anchor，需要对label进行encode。生成类似分割的ground truth。输出的3歌Head，分类，centerness，和回归。

现在没用paddlepaddle了，好在这部分代码与框架无关，还可以在其他地方复用。

[LOSS](https://github.com/zehua9500/FCOS-paddlepaddle/blob/master/code/loss.py) 

之前用的时候paddlepaddle的BUG蛮多的。。自己就提交了几个BUG..



本来还想用COCO数据集复现下结果。奈何太虚弱。跑不动。。也没进行其他预训练。Model部分仿照Pytorch写的。




