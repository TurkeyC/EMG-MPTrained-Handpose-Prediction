<div align="center">

<h1>EMG-MPTrained-Handpose-Prediction</h1>
Hand Pose Prediction using Arm EMG and MeidaPipe<br>
<br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction) [![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/blob/master/LICENSE)<br>
[![GitHub Stars](https://img.shields.io/github/stars/TurkeyC/EMG-MPTrained-Handpose-Prediction.svg)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/stargazers) [![GitHub Forks](https://img.shields.io/github/forks/TurkeyC/EMG-MPTrained-Handpose-Prediction.svg)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/network)  [![GitHub Issues](https://img.shields.io/github/issues/TurkeyC/EMG-MPTrained-Handpose-Prediction.svg)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/issues) [![GitHub Pull Requests](https://img.shields.io/github/issues-pr/TurkeyC/EMG-MPTrained-Handpose-Prediction.svg)](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/pulls)<br>

**English** | [**中文简体**](README_zh.md)
</div>

---

## Project Overview

This project aimed to predict hand gestures using arm EMG signals and a MediaPipe model. The intended workflow was as follows:

1. **Data Acquisition:** Place EMG sensors on the arm and simultaneously record EMG signals while capturing hand gestures with a camera. 
2. **Data Processing:** Align the EMG signals with hand keypoint coordinates extracted by MediaPipe for training purposes.
3. **Model Training:** Train a model to identify the relationship between EMG signals and hand gestures.
4. **Gesture Prediction:** Utilize the trained model to predict hand gestures based on real-time EMG signal acquisition.

However, due to challenges in acquiring and processing arm EMG signals, as well as an overly simplistic project scope, this project ultimately faced setbacks.  

---

## TODO List

- [ ] **Base Optimization** ★★★★★

  - [ ] **Investigate further reasons for subpar results** 🤔
  - [ ] Improve Signal Quality
      - [ ] Transition from single to multi-electrode setup
      - [ ] Implement hardware filtering circuits
      - [ ] Experiment with various electrode placement schemes
      - [ ] Minimize motion artifacts and other interference at the hardware level
  - [ ] Enhance Data Quality
      - [ ] Design standardized action recording protocols (e.g., fixed angle/force)
      - [ ] Develop code to automatically remove anomalous signal segments
      - [ ] Optimize data annotation processes for keypoint correction

  - [ ] Boost Generalizability
      - [ ] Shift electrode signal acquisition from forearm to upper arm
      - [ ] Expand dataset to include various hand gesture datasets
      - [ ] Explore diverse dataset preprocessing, partitioning, and labeling techniques

  - [ ] ……

      

- [ ] **Feature Optimization** ★★★★☆

  - [ ] Integrate time-frequency domain features 
  
  - [ ] Experiment with different window lengths
  
  - [ ] ……
  
  
  
- [ ] **Model Upgrades** ★★★☆☆

  - [ ] Dynamic Time Warping (DTW) alignment loss
  - [ ] Domain Adaptation to address individual variations
  - [ ] ……



- [ ] **Long-Term Planning** ★★☆☆☆
  - [ ] (Placeholder for future ideas)



---

## Thanks to all contributors for their hard work!

<a href="https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=TurkeyC/EMG-MPTrained-Handpose-Prediction" />
</a>

[Back to top 🚀](https://github.com/TurkeyC/EMG-MPTrained-Handpose-Prediction?tab=readme-ov-file#readme)