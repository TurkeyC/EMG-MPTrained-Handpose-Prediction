<div align="center">

<h1>EMG-MPTrained-Handpose-Prediction</h1>

基于手臂EMG与MeidaPipe模型训练的手部姿态预测的一次~~失败的~~尝试<br>
<br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction) [![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/blob/master/LICENSE)<br>
[![GitHub Stars](https://img.shields.io/github/stars/TurkeyC/EMG-MPTrained-Handpose-Prediction.svg)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/stargazers) [![GitHub Forks](https://img.shields.io/github/forks/TurkeyC/EMG-MPTrained-Handpose-Prediction.svg)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/network)  [![GitHub Issues](https://img.shields.io/github/issues/TurkeyC/EMG-MPTrained-Handpose-Prediction.svg)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/issues) [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/TurkeyC/EMG-MPTrained-Handpose-Prediction.svg)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/pulls)<br>

[**English**](README.md) | **中文简体**

</div>

---

## 项目概述
这个项目的最初目的是通过手臂EMG信号和MediaPipe模型训练来预测手部姿态。预期的流程如下：

 1. **数据采集**：在手臂上放置EMG传感器，同步记录EMG信号和摄像头捕捉的手部姿态。

 2. **数据处理**：将EMG信号与MediaPipe提取的手部关键点坐标对齐，用于训练。

 3. **模型训练**：通过训练找出EMG信号与手部姿态之间的关系，生成预测模型。

 4. **姿态预测**：利用训练好的模型，根据实时采集的EMG信号预测手部姿态。

但是，由于手臂EMG信号的采集和处理存在一些问题以及对项目的规划设想过于简单，导致了这个项目的失败。

---

## 待办清单

- [ ] **基础优化 ★★★★★**
  - [ ] **找出还有什么其他原因使得结果远不及预期**🤔
  - [ ] 提升信号质量
    - [ ] 将目前的单电极方案改为多电极方案
    - [ ] 增加硬件滤波电路
    - [ ] 实验尝试不同电极布局方案
    - [ ] 尝试在硬件端尽量减少运动伪影等的干扰
  - [ ] 增强数据质量
    - [ ] 设计标准化动作录制方案，例如固定角度/力度
    - [ ] 尝试编写代码自动剔除异常信号段
    - [ ] 优化数据标注流程，进行关键点修正
  - [ ] 提升泛用性
    - [ ] 将电信号采集位置从小臂改为上臂
    - [ ] 将数据集扩充到多种手势数据集
    - [ ] 尝试更多的数据集预处理、划分、标注方案
  - [ ] ……
  
  
- [ ] **特征优化 ★★★★☆**
  - [ ] 融合时频域特征
  - [ ] 实验不同窗口长度
  - [ ] ……
  
  
- [ ] **模型升级与核心算法改进 ★★★★☆**
  - [ ] 动态时间规整DTW对齐损失
  - [ ] 领域自适应（Domain Adaptation）应对个体差异
  - [ ] ~~暂时想不出来了~~
  - [ ] ……
  
  
  
- [ ] **其他的”长远“规划** ★★☆☆☆
  - [ ] ~~现阶段基本实现不了，暂且不想了吧~~
  - [ ] ……







---

## 感谢所有贡献者作出的努力

<a href="https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=TurkeyC/EMG-MPTrained-Handpose-Prediction" />
</a>

[回到顶部 🚀](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction?tab=readme-ov-file#readme)