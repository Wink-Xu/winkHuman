# 实时行人分析项目  (learn from PP-Human)
The overall architecture of code refers to https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/deploy/pphuman
(工作之余的时间做这个项目，主要是想把行人相关的这一套流程走通，希望达到可以给别人使用的程度。)
集各repo与一体。

1. 把PP-Human的整个流程搞懂。
2. 之前没搞过检测，最近把yolo+deepsort搞清楚。
3. 如何服务化，服务化是在做什么，torchServe Triron 搞清楚
4. GCN行人行为分析网络，搞清楚是怎么做的。
5. 每个网络怎么训练的，用了什么数据，数据怎么标注

# todo list
1. yolov5训自己的数据集。 和zf对比。（yolo系列（包括yolox）要学会了，到可以被提问的程度。也看下centernet） √ 完成了训练，但是没对比。
2. pipleline 跑视频 完成跟踪。（deepsort） 代码整理。加注释。   √
3. 集成HRNet           √ 完成了关键点的单元测试代码。 完成了pipleline中增加关键点估计。
4. 行为识别ST-GCN，成功跑demo，inference。
5. 集成attribute
6. 集成ReID
7. tensort部署
8. 服务化
## 多目标跟踪
yolo + deepsort
This part of code refers to https://github.com/mikel-brostrom/Yolov5_DeepSort_OSNet
## 关键点检测
This part of code refers to https://github.com/HRNet/HRNet-Human-Pose-Estimation
 hrnet
## REID特征提取
This part of code refers to https://github.com/JDAI-CV/fast-reid
 MGN
## 行人属性识别
This part of code refers to https://github.com/JDAI-CV/fast-reid
 strongBaseline
只有性别 年龄几种属性。
## 行人行为分析
This part of code refers to https://github.com/open-mmlab/mmskeleton
 ST-GCN
只分析几种行为

## 服务化
torchServe Triton

## 推理引擎
tensorrt T4


