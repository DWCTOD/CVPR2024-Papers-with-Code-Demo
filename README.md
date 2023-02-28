# CVPR2022-Papers-with-Code-Demo

 :star_and_crescent:**CVPR2021论文下载：https://pan.baidu.com/share/init?surl=gjfUQlPf73MCk4vM8VbzoA**

**密码：aicv**

 :star_and_crescent:**CVPR2022论文下载：添加微信: nvshenj125, 备注 CVPR 2022 即可获取全部论文pdf**
 
 :star_and_crescent:**福利 注册即可领取 200 块计算资源 : https://www.bkunyun.com/wap/console?source=aistudy**
 [使用说明](https://mp.weixin.qq.com/s?__biz=MzU4NTY4Mzg1Mw==&amp;mid=2247521550&amp;idx=1&amp;sn=db4c7f609bd61ae7734b9e012a763f98&amp;chksm=fd8413eccaf39afa686f69f2df2463f4a6a8233ba3b3edf698513bbee556c9f6c21e835b8eb8&token=705359263&lang=zh_CN#rd)


欢迎关注公众号：AI算法与图像处理

:star2: [CVPR 2022](https://cvpr2022.thecvf.com/) 持续更新最新论文/paper和相应的开源代码/code！

:car: CVPR 2022 收录列表ID：https://drive.google.com/file/d/15JFhfPboKdUcIH9LdbCMUFmGq_JhaxhC/view

:car: 官网链接：http://cvpr2022.thecvf.com/

B站demo：https://space.bilibili.com/288489574

> :hand: ​注：欢迎各位大佬提交issue，分享CVPR 2022论文/paper和开源项目！共同完善这个项目
>
> 往年顶会论文汇总：
>
> [CVPR2021](https://github.com/DWCTOD/CVPR2022-Papers-with-Code-Demo/blob/main/CVPR2021.md)
>
> [ICCV2021](https://github.com/DWCTOD/ICCV2021-Papers-with-Code-Demo)

### **:fireworks: 欢迎进群** | Welcome

CVPR 2022 论文/paper交流群已成立！已经收录的同学，可以添加微信：**nvshenj125**，请备注：**CVPR+姓名+学校/公司名称**！一定要根据格式申请，可以拉你进群。

<a name="Contents"></a>



### :hammer: **目录 |Table of Contents（点击直接跳转）**

<details open>
<summary> 目录（右侧点击可折叠）</summary>

- [Backbone](#Backbone)
- [数据集/Dataset](#Dataset)
- [NAS](#NAS)
- [Knowledge Distillation](#KnowledgeDistillation)
- [多模态 / Multimodal ](#Multimodal )
- [对比学习/Contrastive Learning](#ContrastiveLearning)
- [图神经网络 / Graph Neural Networks](#GNN)
- [胶囊网络 / Capsule Network](#CapsuleNetwork)
- [图像分类 / Image Classification](#ImageClassification)
- [目标检测/Object Detection](#ObjectDetection)
- [目标跟踪/Object Tracking](#ObjectTracking)
- [轨迹预测/Trajectory Prediction](#TrajectoryPrediction)
- [语义分割/Segmentation](#Segmentation)
- [弱监督语义分割/Weakly Supervised Semantic Segmentation](#WSSS)
- [医学图像分割](#MedicalImageSegmentation)
- [视频目标分割/Video Object Segmentation](#VideoObjectSegmentation)
- [交互式视频目标分割/Interactive Video Object Segmentation](#InteractiveVideoObjectSegmentation)
- [Visual Transformer](#VisualTransformer)
- [深度估计/Depth Estimation](#DepthEstimation)
- [人脸识别/Face Recognition](#FaceRecognition)
- [人脸检测/Face Detection](#FaceDetection)
- [人脸活体检测/Face Anti-Spoofing](#FaceAnti-Spoofing)
- [人脸年龄估计/Age Estimation](#AgeEstimation)
- [人脸表情识别/Facial Expression Recognition](#FacialExpressionRecognition)
- [人脸属性识别/Facial Attribute Recognition](#FacialAttributeRecognition)
- [人脸编辑/Facial Editing](#FacialEditing)
- [换脸/Face Swap](#FaceSwap)
- [人体姿态估计/Human Pose Estimation](#HumanPoseEstimation)
- [6D位姿估计 /6D Pose Estimation](#6DPoseEstimation)
- [手势姿态估计（重建）/Hand Pose Estimation( Hand Mesh Recovery)](#HandPoseEstimation)
- [视频动作检测/Video Action Detection](#VideoActionDetection)
- [手语翻译/Sign Language Translation](#SignLanguageTranslation)
- [3D人体重建](#3D人体重建)
- [行人重识别/Person Re-identification](#PersonRe-identification)
- [行人搜索/Person Search](#PersonSearch)
- [人群计数 / Crowd Counting](#CrowdCounting)
- [GAN](#GAN)
- [彩妆迁移 / Color-Pattern Makeup Transfer](#CPM)
- [字体生成 / Font Generation](#FontGeneration)
- [场景文本检测、识别/Scene Text Detection/Recognition](#OCR)
- [图像、视频检索 / Image Retrieval/Video retrieval](#Retrieval)
- [Image Animation](#ImageAnimation)
- [抠图/Image Matting](#ImageMatting)
- [超分辨率/Super Resolution](#SuperResolution)
- [图像复原/Image Restoration](#ImageRestoration)
- [图像补全/Image Inpainting](#ImageInpainting)
- [图像去噪/Image Denoising](#ImageDenoising)
- [图像编辑/Image Editing](#ImageEditing)
- [图像拼接/Image stitching](#Imagestitching)
- [图像匹配/Image Matching](#ImageMatching)
- [图像融合/Image Blending](#ImageBlending)
- [图像去雾/Image Dehazing](#ImageDehazing)
- [图像压缩/Image Compression](#ImageCompression)
- [反光去除/Reflection Removal](#ReflectionRemoval)
- [车道线检测/Lane Detection](#LaneDetection)
- [自动驾驶 / Autonomous Driving](#AutonomousDriving)
- [流体重建/Fluid Reconstruction](#FluidReconstruction)
- [场景重建 / Scene Reconstruction](#SceneReconstruction)
- [视频插帧/Frame Interpolation](#FrameInterpolation)
- [视频超分 / Video Super-Resolution](#VideoSuper-Resolution)
- [ 3D点云/3D point cloud]( #3DPointCloud)
- [标签噪声 / Label-Noise](#Label-Noise)
- [对抗样本/Adversarial Examples](#AdversarialExamples)
- [其他/Other](#Other)


</details>

<a name="Backbone"></a>

## Backbone



[返回目录/back](#Contents)

<a name="Dataset"></a> 

## 数据集/Dataset

**3MASSIV: Multilingual, Multimodal and Multi-Aspect dataset of Social Media Short Videos**

- 论文/Paper: http://arxiv.org/abs/2203.14456
- 代码/Code: None

**Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities**

- 论文/Paper: http://arxiv.org/abs/2203.14712
- 代码/Code: None

**DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation**

- 论文/Paper: http://arxiv.org/abs/2203.12560
- 代码/Code: https://mediatum.ub.tum.de/1650201

**Dataset Distillation by Matching Training Trajectories**

- 论文/Paper: http://arxiv.org/abs/2203.11932
- 代码/Code: https://github.com/GeorgeCazenavette/mtt-distillation

**FERV39k: A Large-Scale Multi-Scene Dataset for Facial Expression Recognition in Videos**

- 论文/Paper：https://arxiv.org/abs/2203.09463

- 代码/Code：

**GrainSpace: A Large-scale Dataset for Fine-grained and Domain-adaptive Recognition of Cereal Grains**

- 论文/Paper：https://arxiv.org/abs/2203.05306
- 代码/Code：https://github.com/hellodfan/GrainSpace

**STCrowd: A Multimodal Dataset for Pedestrian Perception in Crowded Scenes**

论文/Paper: http://arxiv.org/pdf/2204.01026

代码/Code: https://github.com/4dvlab/stcrowd

**ObjectFolder 2.0: A Multisensory Object Dataset for Sim2Real Transfer**

- 论文/Paper: http://arxiv.org/pdf/2204.02389
- 代码/Code: None

**BEHAVE: Dataset and Method for Tracking Human Object Interactions**

- 论文/Paper: http://arxiv.org/pdf/2204.06950
- 代码/Code: None

**SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos**

- 论文/Paper: http://arxiv.org/pdf/2204.06918
- 代码/Code: None

**Hephaestus: A large scale multitask dataset towards InSAR understanding**

- 论文/Paper: http://arxiv.org/pdf/2204.09435
- 代码/Code: None

**A New Dataset and Transformer for Stereoscopic Video Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2204.10039
- 代码/Code: https://github.com/H-deep/Trans-SVSR/

[返回目录/back](#Contents)

<a name="NAS"></a> 

## NAS



**Optimizing Elimination Templates by Greedy Parameter Search**

- 论文/Paper: http://arxiv.org/abs/2203.14901
- 代码/Code: None

**Searching for Network Width with Bilaterally Coupled Network**

- 论文/Paper: http://arxiv.org/pdf/2203.13714
- 代码/Code: None

**Arch-Graph: Acyclic Architecture Relation Predictor for Task-Transferable Neural Architecture Search**

- 论文/Paper: http://arxiv.org/pdf/2204.05941
- 代码/Code: None

[返回目录/back](#Contents)

<a name="KnowledgeDistillation"></a> 

## Knowledge Distillation

**Decoupled Knowledge Distillation**

- 论文/Paper：https://arxiv.org/abs/2203.08679
- 代码/Code：https://github.com/megvii-research/mdistiller

**Knowledge Distillation with the Reused Teacher Classifier**

- 论文/Paper: http://arxiv.org/abs/2203.14001
- 代码/Code: None

[返回目录/back](#Contents)

<a name="Multimodal"></a> 

## 多模态 / Multimodal

**Balanced Multimodal Learning via On-the-fly Gradient Modulation**

- 论文/Paper: http://arxiv.org/pdf/2203.15332
- 代码/Code: None

**Conditional Prompt Learning for Vision-Language Models**

- 论文/Paper：https://arxiv.org/abs/2203.05557
- 代码/Code：https://github.com/KaiyangZhou/CoOp

**Learning Hierarchical Cross-Modal Association for Co-Speech Gesture Generation**

- 论文/Paper: http://arxiv.org/abs/2203.13161
- 代码/Code: None

**Motron: Multimodal Probabilistic Human Motion Forecasting**

- 论文/Paper：https://arxiv.org/abs/2203.04132
- 代码/Code：

**StyleT2I: Toward Compositional and High-Fidelity Text-to-Image Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2203.15799
- 代码/Code: https://github.com/zhihengli-UR/StyleT2I

**Text2Pos: Text-to-Point-Cloud Cross-Modal Localization**

- 论文/Paper: http://arxiv.org/pdf/2203.15125
- 代码/Code: None

**Towards Implicit Text-Guided 3D Shape Generation**

- 论文/Paper: http://arxiv.org/abs/2203.14622
- 代码/Code: None

**UMT: Unified Multi-modal Transformers for Joint Video Moment Retrieval and Highlight Detection**

- 论文/Paper: http://arxiv.org/abs/2203.12745
- 代码/Code: None

**Versatile Multi-Modal Pre-Training for Human-Centric Perception**

- 论文/Paper: http://arxiv.org/pdf/2203.13815
- 代码/Code: None

**X-Pool: Cross-Modal Language-Video Attention for Text-Video Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2203.15086
- 代码/Code: https://github.com/layer6ai-labs/xpool

**ViSTA: Vision and Scene Text Aggregation for Cross-Modal Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2203.16778
- 代码/Code: None

**STCrowd: A Multimodal Dataset for Pedestrian Perception in Crowded Scenes**

- 论文/Paper: http://arxiv.org/pdf/2204.01026

- 代码/Code: https://github.com/4dvlab/stcrowd

**XMP-Font: Self-Supervised Cross-Modality Pre-training for Few-Shot Font Generation**

- 论文/Paper: http://arxiv.org/pdf/2204.05084
- 代码/Code: None

**Robust Cross-Modal Representation Learning with Progressive Self-Distillation**

- 论文/Paper: http://arxiv.org/pdf/2204.04588
- 代码/Code: None

**Multimodal Transformer for Nursing Activity Recognition**

- 论文/Paper: http://arxiv.org/pdf/2204.04564
- 代码/Code: https://github.com/Momilijaz96/MMT_for_NCRC

**Probabilistic Compositional Embeddings for Multimodal Image Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2204.05845
- 代码/Code: https://github.com/andreineculai/MPC.

**Are Multimodal Transformers Robust to Missing Modality?**

- 论文/Paper: http://arxiv.org/pdf/2204.05454
- 代码/Code: None

**Multimodal Token Fusion for Vision Transformers**

- 论文/Paper: http://arxiv.org/pdf/2204.08721
- 代码/Code: None

**Transformer Decoders with MultiModal Regularization for Cross-Modal Food Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2204.09730
- 代码/Code: https://github.com/mshukor/TFood

**CLIP-Art: Contrastive Pre-training for Fine-Grained Art Classification**

- 论文/Paper: http://arxiv.org/pdf/2204.14244
- 代码/Code: https://github.com/KeremTurgutlu/clip_art

**Vision-Language Pre-Training for Boosting Scene Text Detectors**

- 论文/Paper: http://arxiv.org/pdf/2204.13867
- 代码/Code: None

**Cross-modal Representation Learning for Zero-shot Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2205.01657
- 代码/Code: None

**Episodic Memory Question Answering**

- 论文/Paper: http://arxiv.org/pdf/2205.01652
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ContrastiveLearning"></a> 

## Contrastive Learning

**Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2203.13505
- 代码/Code: None

**ContrastMask: Contrastive Learning to Segment Every Thing**

- 论文/Paper: http://arxiv.org/pdf/2203.09775
- 代码/Code: None

**Fair Contrastive Learning for Facial Attribute Classification**

- 论文/Paper: http://arxiv.org/pdf/2203.16209
- 代码/Code: https://github.com/sungho-coolg/fscl

**Frame-wise Action Representations for Long Videos via Sequence Contrastive Learning**

- 论文/Paper: http://arxiv.org/abs/2203.14957
- 代码/Code: None

**Rethinking Minimal Sufficient Representation in Contrastive Learning**

- 论文/Paper：https://arxiv.org/abs/2203.07004

- 代码/Code：https://github.com/Haoqing-Wang/InfoCL

**Selective-Supervised Contrastive Learning with Noisy Labels**

- 论文/Paper：https://arxiv.org/abs/2203.04181
- 代码/Code：https://github.com/ShikunLi/Sel-CL

**Unsupervised Deraining: Where Contrastive Learning Meets Self-similarity**

- 论文/Paper: http://arxiv.org/abs/2203.11509
- 代码/Code: None

**Fine-grained Temporal Contrastive Learning for Weakly-supervised Temporal Action Localization**

- 论文/Paper: http://arxiv.org/pdf/2203.16800
- 代码/Code: https://github.com/MengyuanChen21/CVPR2022-FTCL

**Unified Contrastive Learning in Image-Text-Label Space**

- 论文/Paper: http://arxiv.org/pdf/2204.03610
- 代码/Code: https://github.com/microsoft/unicl

**Probabilistic Representations for Video Contrastive Learning**

- 论文/Paper: http://arxiv.org/abs/2204.03946
- 代码/Code: None

**Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework**

- 论文/Paper: http://arxiv.org/pdf/2204.13207
- 代码/Code: https://github.com/salesforce/hierarchicalContrastiveLearning.

**UTC: A Unified Transformer with Inter-Task Contrastive Learning for Visual Dialog**

- 论文/Paper: http://arxiv.org/pdf/2205.00423
- 代码/Code: None

[返回目录/back](#Contents)

<a name="GNN"></a> 

## 图神经网络 / Graph Neural Networks

**Lifelong Graph Learning**

- 论文/paper：https://arxiv.org/abs/2009.00647
- 代码/code：https://github.com/wang-chen/LGL

**Long-term Visual Map Sparsification with Heterogeneous GNN**

- 论文/Paper: http://arxiv.org/pdf/2203.15182
- 代码/Code: None

**SkinningNet: Two-Stream Graph Convolutional Neural Network for Skinning Prediction of Synthetic Characters**

- 论文/paper：https://arxiv.org/abs/2203.04746
- 代码/code：https://imatge-upc.github.io/skinningnet/

[返回目录/back](#Contents)

<a name="CapsuleNetwork"></a> 

# 胶囊网络 / Capsule Network

**HP-Capsule: Unsupervised Face Part Discovery by Hierarchical Parsing Capsule Network**

- 论文/Paper: http://arxiv.org/abs/2203.10699
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ImageClassification"></a> 

# 图像分类 / Image Classification

**CAD: Co-Adapting Discriminative Features for Improved Few-Shot Classification**

- 论文/Paper: http://arxiv.org/pdf/2203.13465
- 代码/Code: None

**Integrative Few-Shot Learning for Classification and Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2203.15712
- 代码/Code: None

**Matching Feature Sets for Few-Shot Image Classification**

- 论文/Paper: http://arxiv.org/pdf/2204.00949
- 代码/Code: None

**Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification**

- 论文/Paper: http://arxiv.org/pdf/2204.04567
- 代码/Code: None

**Regression or Classification? Reflection on BP prediction from PPG data using Deep Neural Networks in the scope of practical applications**

- 论文/Paper: http://arxiv.org/pdf/2204.05605
- 代码/Code: None

**Revisiting Vicinal Risk Minimization for Partially Supervised Multi-Label Classification Under Data Scarcity**

- 论文/Paper: http://arxiv.org/pdf/2204.08954
- 代码/Code: None

**Self-supervised Learning for Sonar Image Classification**

- 论文/Paper: http://arxiv.org/pdf/2204.09323
- 代码/Code: https://github.com/agrija9/ssl-sonar-images

**Generating Representative Samples for Few-Shot Classification**

- 论文/Paper: http://arxiv.org/pdf/2205.02918
- 代码/Code: https://github.com/cvlab-stonybrook/fsl-rsvae.

[返回目录/back](#Contents)

<a name="ObjectDetection"></a> 

## 目标检测/Object Detection

**A Dual Weighting Label Assignment Scheme for Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2203.09730
- 代码/Code: https://github.com/strongwolf/dw

**Implicit Motion Handling for Video Camouflaged Object Detection**

- 论文/Paper：https://arxiv.org/abs/2203.07363

- 代码/Code：

**Democracy Does Matter: Comprehensive Feature Mining for Co-Salient Object Detection**

- 论文/Paper：https://arxiv.org/abs/2203.05787
- 代码/Code：

**DeepFusion: Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection**

- 论文/Paper：https://arxiv.org/abs/2203.08195

- 代码/Code：https://github.com/tensorflow/lingvo/tree/master/lingvo/

**Efficient Two-Stage Detection of Human-Object Interactions with a Novel Unary-Pairwise Transformer**

- 论文/paper：https://arxiv.org/abs/2112.01838 | [主页](https://fredzzhang.com/unary-pairwise-transformers/)
- 代码/code：https://github.com/fredzzhang/upt

**Expanding Low-Density Latent Regions for Open-Set Object Detection**

- 论文/Paper: http://arxiv.org/abs/2203.14911
- 代码/Code: None

**Ev-TTA: Test-Time Adaptation for Event-Based Object Recognition**

- 论文/Paper: http://arxiv.org/abs/2203.12247
- 代码/Code: None

**Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes**

- 论文/paper：https://arxiv.org/abs/2011.12001
- 代码/code：https://github.com/qq456cvb/CanonicalVoting

**Back to Reality: Weakly-supervised 3D Object Detection with Shape-guided Label Enhancement**

- 论文/Paper：https://arxiv.org/abs/2203.05238
- 代码/Code：https://github.com/xuxw98/BackToReality

**LiDAR Snowfall Simulation for Robust 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2203.15118
- 代码/Code: None

**Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model**

- 论文/Paper: http://arxiv.org/abs/2203.14940
- 代码/Code: None

**Knowledge Distillation as Efficient Pre-training: Faster Convergence, Higher Data-efficiency, and Better Transferability**

- 论文/Paper：https://arxiv.org/abs/2203.05180
- 代码/Code：https://github.com/CVMI-Lab/KDEP

**Optimal Correction Cost for Object Detection Evaluation**

- 论文/Paper: http://arxiv.org/abs/2203.14438
- 代码/Code: None

**Point2Seq: Detecting 3D Objects as Sequences**

- 论文/Paper: http://arxiv.org/pdf/2203.13394
- 代码/Code: None

**Point Density-Aware Voxels for LiDAR 3D Object Detection**

- 论文/Paper：https://arxiv.org/abs/2203.05662
- 代码/Code：https://github.com/TRAILab/PDV

**MonoJSG: Joint Semantic and Geometric Cost Volume for Monocular 3D Object Detection**

- 论文/Paper：https://arxiv.org/abs/2203.08563
- 代码/Code：

**MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer**

- 论文/Paper: http://arxiv.org/abs/2203.10981
- 代码/Code: None

**MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2203.13310

**Real-time Object Detection for Streaming Perception**

- 论文/Paper: http://arxiv.org/abs/2203.12338
- 代码/Code: https://github.com/yancie-yjr/StreamYOLO

**SIOD: Single Instance Annotated Per Category Per Image for Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2203.15353
- 代码/Code: None

**SIGMA: Semantic-complete Graph Matching for Domain Adaptive Object Detection**

- 论文/Paper：https://arxiv.org/abs/2203.06398
- 代码/Code：https://github.com/CityU-AIM-Group/SIGMA

**Sparse Fuse Dense: Towards High Quality 3D Detection with Depth Completion**

- 论文/Paper: http://arxiv.org/pdf/2203.09780
- 代码/Code: None

**Task-specific Inconsistency Alignment for Domain Adaptive Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2203.15345
- 代码/Code: None

**TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers**

- 论文/Paper: http://arxiv.org/abs/2203.11496
- 代码/Code: https://github.com/XuyangBai/TransFusion

**VISTA: Boosting 3D Object Detection via Dual Cross-VIew SpaTial Attention**

- 论文/Paper: http://arxiv.org/pdf/2203.09704
- 代码/Code: https://github.com/gorilla-lab-scut/vista

**Voxel Set Transformer: A Set-to-Set Approach to 3D Object Detection from Point Clouds**

- 论文/Paper: http://arxiv.org/abs/2203.10314
- 代码/Code: None

**Rope3D: TheRoadside Perception Dataset for Autonomous Driving and Monocular 3D Object Detection Task**

- 论文/Paper: http://arxiv.org/pdf/2203.13608
- 代码/Code: None

**Understanding 3D Object Articulation in Internet Videos**

- 论文/Paper: http://arxiv.org/pdf/2203.16531
- 代码/Code: None

**AdaMixer: A Fast-Converging Query-Based Object Detector**

- 论文/Paper: http://arxiv.org/pdf/2203.16507
- 代码/Code: https://github.com/mcg-nju/adamixer

**Forecasting from LiDAR via Future Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2203.16297
- 代码/Code: None

**Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2203.16220
- 代码/Code: https://github.com/dlut-dimt/tardal

**Learning of Global Objective for Network Flow in Multi-Object Tracking**

- 论文/Paper: http://arxiv.org/pdf/2203.16210
- 代码/Code: None

**FLOAT: Factorized Learning of Object Attributes for Improved Multi-object Multi-part Scene Parsing**

- 论文/Paper: http://arxiv.org/pdf/2203.16168
- 代码/Code: None

**Omni-DETR: Omni-Supervised Object Detection with Transformers**

- 论文/Paper: http://arxiv.org/pdf/2203.16089
- 代码/Code: None

**Learning to Detect Mobile Objects from LiDAR Scans Without Labels**

- 论文/Paper: http://arxiv.org/pdf/2203.15882
- 代码/Code: https://github.com/yurongyou/modest

**Multi-Granularity Alignment Domain Adaptation for Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2203.16897
- 代码/Code: None

**CAT-Det: Contrastively Augmented Transformer for Multi-modal 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.00325

- 代码/Code: None

**R(Det)^2: Randomized Decision Routing for Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.00794

- 代码/Code: None

**Homography Loss for Monocular 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.00754
- 代码/Code: https://github.com/gujiaqivadin/HomographyLoss

**Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation**

- 论文/Paper: http://arxiv.org/pdf/2204.02136
- 代码/Code: None

**Towards Robust Adaptive Object Detection under Noisy Annotations**

- 论文/Paper: http://arxiv.org/pdf/2204.02620
- 代码/Code: None

**Towards Open-Set Object Detection and Discovery**

- 论文/Paper: http://arxiv.org/pdf/2204.05604
- 代码/Code: None

**DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.05575
- 代码/Code: https://github.com/AIR-THU/DAIR-V2X.

**HyperDet3D: Learning a Scene-conditioned 3D Object Detector**

- 论文/Paper: http://arxiv.org/pdf/2204.05599
- 代码/Code: None

**Dense Learning based Semi-Supervised Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.07300
- 代码/Code: https://github.com/chenbinghui1/DSL

**Entropy-based Active Learning for Object Detection with Progressive Diversity Constraint**

- 论文/Paper: http://arxiv.org/pdf/2204.07965
- 代码/Code: None

**Target-Relevant Knowledge Preservation for Multi-Source Domain Adaptive Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.07964
- 代码/Code: None

**Modeling Missing Annotations for Incremental Learning in Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.08766
- 代码/Code: https://github.com/fcdl94/MMA

**Augmentation of Atmospheric Turbulence Effects on Thermal Adapted Object Detection Models**

- 论文/Paper: http://arxiv.org/pdf/2204.08745
- 代码/Code: None

**Focal Sparse Convolutional Networks for 3D Object Detection**

- 论文/Paper: http://arxiv.org/abs/2204.12463
- 代码/Code: http://github.com/dvlab-research/FocalsConv

**Rotationally Equivariant 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.13630
- 代码/Code: None

**Cross Domain Object Detection by Target-Perceived Dual Branch Distillation**

- 论文/Paper: http://arxiv.org/pdf/2205.01291
- 代码/Code: https://github.com/feobi1999/tdd

**Dynamic Sparse R-CNN**

- 论文/Paper: http://arxiv.org/pdf/2205.02101
- 代码/Code: None

[返回目录/back](#Contents)



<a name="ObjectTracking"></a> 

## 目标跟踪/Object Tracking

**DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion**

- 论文/Paper：https://arxiv.org/abs/2111.14690
- 代码/Code：https://github.com/DanceTrack/DanceTrack

**Global Tracking Transformers**

- 论文/Paper: http://arxiv.org/abs/2203.13250
- 代码/Code: None

**MixFormer: End-to-End Tracking with Iterative Mixed Attention**

- 论文/Paper: http://arxiv.org/abs/2203.11082
- 代码/Code: None

**Transforming Model Prediction for Tracking**

- 论文/Paper: http://arxiv.org/abs/2203.11192
- 代码/Code: None

**TCTrack: Temporal Contexts for Aerial Tracking**

- 论文/Paper：https://arxiv.org/abs/2203.01885
- 代码/Code：https://github.com/vision4robotics/TCTrack

**Unified Transformer Tracker for Object Tracking**

- 论文/Paper: http://arxiv.org/pdf/2203.15175
- 代码/Code: None

**Learning of Global Objective for Network Flow in Multi-Object Tracking**

- 论文/Paper: http://arxiv.org/pdf/2203.16210
- 代码/Code: None

**Global Tracking via Ensemble of Local Trackers**

- 论文/Paper: http://arxiv.org/pdf/2203.16092
- 代码/Code: https://github.com/zikunzhou/gtelt

**MeMOT: Multi-Object Tracking with Memory**

- 论文/Paper: http://arxiv.org/pdf/2203.16761
- 代码/Code: None

**Unsupervised Learning of Accurate Siamese Tracking**

- 论文/Paper: http://arxiv.org/pdf/2204.01475
- 代码/Code: https://github.com/florinshum/ulast

**Visible-Thermal UAV Tracking: A Large-Scale Benchmark and New Baseline**

- 论文/Paper: http://arxiv.org/abs/2204.04120
- 代码/Code: None

**BEHAVE: Dataset and Method for Tracking Human Object Interactions**

- 论文/Paper: http://arxiv.org/pdf/2204.06950
- 代码/Code: None

**SoccerNet-Tracking: Multiple Object Tracking Dataset and Benchmark in Soccer Videos**

- 论文/Paper: http://arxiv.org/pdf/2204.06918
- 代码/Code: None

**Detecting, Tracking and Counting Motorcycle Rider Traffic Violations on Unconstrained Roads**

- 论文/Paper: http://arxiv.org/pdf/2204.08364
- 代码/Code: None

# 3D Object Tracking

**Iterative Corresponding Geometry: Fusing Region and Depth for Highly Efficient 3D Tracking of Textureless Objects**

- 视频/Demo：[Youtube](https://www.youtube.com/watch?v=qMr1RHCsnDk) 
- 论文/Paper：https://arxiv.org/abs/2203.05334
- 代码/Code：https://github.com/DLR-RM/3DObjectTracking

**Multi-Camera Multiple 3D Object Tracking on the Move for Autonomous Vehicles**

- 论文/Paper: http://arxiv.org/pdf/2204.09151
- 代码/Code: None

[返回目录/back](#Contents)

<a name="TrajectoryPrediction"></a> 

## 轨迹预测/Trajectory Prediction

**How many Observations are Enough? Knowledge Distillation for Trajectory Forecasting**

- 论文/Paper：https://arxiv.org/abs/2203.04781
- 代码/Code：

**Non-Probability Sampling Network for Stochastic Human Trajectory Prediction**

- 论文/Paper: http://arxiv.org/pdf/2203.13471
- 代码/Code: None

**Remember Intentions: Retrospective-Memory-based Trajectory Prediction**

- 论文/Paper: http://arxiv.org/abs/2203.11474
- 代码/Code: None

**Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion**

- 论文/Paper: http://arxiv.org/pdf/2203.13777
- 代码/Code: None

**Goal-driven Self-Attentive Recurrent Networks for Trajectory Prediction**

- 论文/Paper: http://arxiv.org/pdf/2204.11561
- 代码/Code: None

[返回目录/back](#Contents)

<a name="Segmentation"></a> 

## 语义分割/Segmentation

**Class-Balanced Pixel-Level Self-Labeling for Domain Adaptive Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2203.09744
- 代码/Code: https://github.com/lslrh/cpsl

**Deep Hierarchical Semantic Segmentation**

- 论文/Paper: http://arxiv.org/abs/2203.14335
- 代码/Code: None

**E2EC: An End-to-End Contour-based Method for High-Quality High-Speed Instance Segmentation**

- 论文/Paper：https://arxiv.org/abs/2203.04074
- 代码/Code：https://github.com/zhang-tao-whu/e2ec

**Hyperbolic Image Segmentation**

- 论文/Paper：https://arxiv.org/abs/2203.05898
- 代码/Code：

**Mask Transfiner for High-Quality Instance Segmentation**

- 论文/Paper： https://arxiv.org/abs/2111.13673
- 代码/Code：https://github.com/SysCV/transfiner

**Noisy Boundaries: Lemon or Lemonade for Semi-supervised Instance Segmentation?**

- 论文/Paper: http://arxiv.org/pdf/2203.13427
- 代码/Code: None

**Rethinking Semantic Segmentation: A Prototype View**

- 论文/Paper: http://arxiv.org/pdf/2203.15102
- 代码/Code: None

**Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2203.09653
- 代码/Code: https://github.com/maeve07/rca

**Representation Compensation Networks for Continual Semantic Segmentation**

- 论文/Paper：https://arxiv.org/abs/2203.05402
- 代码/Code：https://github.com/zhangchbin/RCIL

**SimT: Handling Open-set Noise for Domain Adaptive Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2203.15202
- 代码/Code: None

**Semantic Segmentation by Early Region Proxy**

- 论文/Paper: http://arxiv.org/abs/2203.14043
- 代码/Code: None

**Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels**

- 论文/Paper：https://arxiv.org/abs/2203.03884
- 代码/Code：

**SharpContour: A Contour-based Boundary Refinement Approach for Efficient and Accurate Instance Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2203.13312
- 代码/Code: None

**ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation**

- 论文/paper：https://arxiv.org/abs/2106.05095
- 代码/code：https://github.com/LiheYoung/ST-PlusPlus

**Scribble-Supervised LiDAR Semantic Segmentation**

- 论文/Paper：https://arxiv.org/abs/2203.08537
- 代码/Code：https://github.com/ouenal/scribblekitti

**Sparse Instance Activation for Real-Time Instance Segmentation**

- 论文/Paper: http://arxiv.org/abs/2203.12827
- 代码/Code: None

**Tree Energy Loss: Towards Sparsely Annotated Semantic Segmentation**

- 论文/Paper: http://arxiv.org/abs/2203.10739
- 代码/Code: None

**Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation**

- 论文/Paper: https://arxiv.org/pdf/2111.12940.pdf
- 代码/Code:https://github.com/BIT-DA/RIPU

**Weakly Supervised Semantic Segmentation using Out-of-Distribution Data**

- 论文/Paper：https://arxiv.org/abs/2203.03860
- 代码/Code：None

**ReSTR: Convolution-free Referring Image Segmentation Using Transformers**

- 论文/Paper: http://arxiv.org/pdf/2203.16768
- 代码/Code: None

**FIFO: Learning Fog-invariant Features for Foggy Scene Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.01587

- 代码/Code: None

**WildNet: Learning Domain Generalized Semantic Segmentation from the Wild**

- 论文/Paper: http://arxiv.org/pdf/2204.01446

- 代码/Code: https://github.com/suhyeonlee/wildnet

**Semantic-Aware Domain Generalized Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.00822
- 代码/Code: https://github.com/leolyj/san-saw

**FocalClick: Towards Practical Interactive Image Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.02574
- 代码/Code: https://github.com/XavierCHEN34/ClickSEG

**Modeling Motion with Multi-Modal Features for Text-Based Video Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.02547
- 代码/Code: None

**Pin the Memory: Learning to Generalize Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.03609
- 代码/Code: None

**Coarse-to-Fine Feature Mining for Video Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.03330
- 代码/Code: https://github.com/guoleisun/vss-cffm

**L2G: A Simple Local-to-Global Knowledge Transfer Framework for Weakly Supervised Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.03206
- 代码/Code: https://github.com/PengtaoJiang/L2G

**Video K-Net: A Simple, Strong, and Unified Baseline for Video Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.04656
- 代码/Code: https://github.com/lxtGH/Video-K-Net

**NightLab: A Dual-level Architecture with Hardness Detection for Segmentation at Night**

- 论文/Paper: http://arxiv.org/pdf/2204.05538
- 代码/Code: None

**TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.05525
- 代码/Code: https://github.com/hustvl/TopFormer

**Panoptic, Instance and Semantic Relations: A Relational Context Encoder to Enhance Panoptic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.05370
- 代码/Code: None

**Open-World Instance Segmentation: Exploiting Pseudo Ground Truth From Learned Pairwise Affinity**

- 论文/Paper: http://arxiv.org/pdf/2204.06107
- 代码/Code: None

**Joint Forecasting of Panoptic Segmentations with Difference Attention  (Oral)**

- 论文/Paper: http://arxiv.org/pdf/2204.07157
- 代码/Code: None

**Cross-Image Relational Knowledge Distillation for Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.06986
- 代码/Code: https://github.com/winycg/cirkd

**Learning Multi-View Aggregation In the Wild for Large-Scale 3D Semantic Segmentation  (Oral)**

- 论文/Paper: http://arxiv.org/pdf/2204.07548
- 代码/Code: https://github.com/drprojects/DeepViewAgg

**Temporally Efficient Vision Transformer for Video Instance Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.08412
- 代码/Code: https://github.com/hustvl/TeViT.

**Augmentation Invariance and Adaptive Sampling in Semantic Segmentation of Agricultural Aerial Images**

- 论文/Paper: http://arxiv.org/pdf/2204.07969
- 代码/Code: None

**Dual-Domain Image Synthesis using Segmentation-Guided GAN**

- 论文/Paper: http://arxiv.org/pdf/2204.09015
- 代码/Code: https://github.com/denabazazian/Dual-Domain-Synthesis.

**Proposal-free Lidar Panoptic Segmentation with Pillar-level Affinity**

- 论文/Paper: http://arxiv.org/pdf/2204.08744
- 代码/Code: None

**Unsupervised Domain Adaptation for Cardiac Segmentation: Towards Structure Mutual Information Maximization**

- 论文/Paper: http://arxiv.org/pdf/2204.09334
- 代码/Code: https://github.com/LOUEY233/Toward-Mutual-Information}{https://github.com/LOUEY233/Toward-Mutual-Information

**Dynamic Prototype Convolution Network for Few-Shot Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.10638
- 代码/Code: None

**Interactive Segmentation and Visualization for Tiny Objects in Multi-megapixel Images**

- 论文/Paper: http://arxiv.org/pdf/2204.10356
- 代码/Code: https://github.com/cy-xu/cosmic-conn

**Multi-Head Distillation for Continual Unsupervised Domain Adaptation in Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.11667
- 代码/Code: None

**Unsupervised Hierarchical Semantic Segmentation with Multiview Cosegmentation and Clustering Transformers**

- 论文/Paper: http://arxiv.org/pdf/2204.11432
- 代码/Code: https://github.com/twke18/HSG

**Transfer Learning from Synthetic In-vitro Soybean Pods Dataset for In-situ Segmentation of On-branch Soybean Pod**

- 论文/Paper: http://arxiv.org/pdf/2204.10902
- 代码/Code: None

**DArch: Dental Arch Prior-assisted 3D Tooth Instance Segmentation**

- 论文/Paper: http://arxiv.org/abs/2204.11911
- 代码/Code: None

**Self-Supervised Learning of Object Parts for Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.13101
- 代码/Code: None

**MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.12667
- 代码/Code: None

**Cross-view Transformers for real-time Map-view Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2205.02833
- 代码/Code: https://github.com/bradyz/cross_view_transformers

[返回目录/back](#Contents)



<a name="WSSS"></a>

## 弱监督语义分割/Weakly Supervised Semantic Segmentation



[返回目录/back](#Contents)

<a name="MedicalImageSegmentation"></a>

# 医学图像分割/Medical Image Segmentation



[返回目录/back](#Contents)

<a name="VideoObjectSegmentation"></a>

# 视频目标分割/Video Object Segmentation

**Language as Queries for Referring Video Object Segmentation**

- 论文/paper：https://arxiv.org/abs/2201.00487

- 代码/code：https://github.com/wjn922/ReferFormer

[返回目录/back](#Contents)

<a name="InteractiveVideoObjectSegmentation"></a>

# 交互式视频目标分割/Interactive Video Object Segmentation

**MSTR: Multi-Scale Transformer for End-to-End Human-Object Interaction Detection**

- 论文/Paper: http://arxiv.org/abs/2203.14709
- 代码/Code: None

**OakInk: A Large-scale Knowledge Repository for Understanding Hand-Object Interaction**

- 论文/Paper: http://arxiv.org/pdf/2203.15709
- 代码/Code: None

**What to look at and where: Semantic and Spatial Refined Transformer for detecting human-object interactions**

- 论文/Paper: http://arxiv.org/pdf/2204.00746

- 代码/Code: None

[返回目录/back](#Contents)

<a name="VisualTransformer"></a>

# Visual Transformer

**Affine Medical Image Registration with Coarse-to-Fine Vision Transformer**

- 论文/Paper: http://arxiv.org/pdf/2203.15216
- 代码/Code: https://github.com/cwmok/C2FViT

**Automated Progressive Learning for Efficient Training of Vision Transformers**

- 论文/Paper: http://arxiv.org/abs/2203.14509
- 代码/Code: None

**Attribute Surrogates Learning and Spectral Tokens Pooling in Transformers for Few-shot Learning**

- 论文/Paper：https://arxiv.org/abs/2203.09064
- 代码/Code：https://github.com/StomachCold/HCTransformers

**Cascade Transformers for End-to-End Person Search**

- 论文/Paper: http://arxiv.org/pdf/2203.09642
- 代码/Code: https://github.com/kitware/coat

**EDTER: Edge Detection with Transformer**

- 论文/Paper：https://arxiv.org/abs/2203.08566
- 代码/Code：

**Few-Shot Object Detection with Fully Cross-Transformer**

- 论文/Paper: http://arxiv.org/pdf/2203.15021
- 代码/Code: None

**Global Tracking Transformers**

- 论文/Paper: http://arxiv.org/abs/2203.13250
- 代码/Code: None

**GradViT: Gradient Inversion of Vision Transformers**

- 论文/Paper: http://arxiv.org/abs/2203.11894
- 代码/Code: https://gradvit.github.io/

**Hyperbolic Vision Transformers: Combining Improvements in Metric Learning**

- 论文/Paper: http://arxiv.org/abs/2203.10833
- 代码/Code: None

**Meta-attention for ViT-backed Continual Learning**

- 论文/Paper: http://arxiv.org/abs/2203.11684
- 代码/Code: None

**MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation**

- 论文/Paper: https://arxiv.org/pdf/2111.12707.pdf

- 代码/Code: https://github.com/Vegetebird/MHFormer

**Self-Supervised Transformers for Unsupervised Object Discovery using Normalized Cut**

- 论文/Paper：https://arxiv.org/abs/2202.11539 | [主页](https://www.m-psi.fr/Papers/TokenCut2022/)
- 代码/Code：https://github.com/YangtaoWANG95/TokenCut

**Training-free Transformer Architecture Search**

- 论文/Paper: http://arxiv.org/abs/2203.12217
- 代码/Code: None

**Towards Practical Certifiable Patch Defense with Vision Transformer**

- 论文/Paper：https://arxiv.org/abs/2203.08519

- 代码/Code：

**Towards Robust Vision Transformer**

- 论文/Paper: https://arxiv.org/abs/2105.07926
- 代码/Code: https://github.com/vtddggg/Robust-Vision-Transformer

**Collaborative Transformers for Grounded Situation Recognition**

- 论文/Paper: http://arxiv.org/pdf/2203.16518
- 代码/Code: https://github.com/jhcho99/coformer

**TubeDETR: Spatio-Temporal Video Grounding with Transformers**

- 论文/Paper: http://arxiv.org/pdf/2203.16434
- 代码/Code: https://github.com/antoyang/TubeDETR

**InstaFormer: Instance-Aware Image-to-Image Translation with Transformer**

- 论文/Paper: http://arxiv.org/pdf/2203.16248
- 代码/Code: None

**Spatial-Temporal Parallel Transformer for Arm-Hand Dynamic Estimation**

- 论文/Paper: http://arxiv.org/pdf/2203.16202
- 代码/Code: None

**Omni-DETR: Omni-Supervised Object Detection with Transformers**

- 论文/Paper: http://arxiv.org/pdf/2203.16089
- 代码/Code: None

**TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing**

- 论文/Paper: http://arxiv.org/pdf/2203.17266
- 代码/Code: https://github.com/BillyXYB/TransEditor

**VL-InterpreT: An Interactive Visualization Tool for Interpreting Vision-Language Transformers**

- 论文/Paper: http://arxiv.org/pdf/2203.17247
- 代码/Code: None

**CRAFT: Cross-Attentional Flow Transformer for Robust Optical Flow**

- 论文/Paper: http://arxiv.org/pdf/2203.16896
- 代码/Code: None

**Deformable Video Transformer**

- 论文/Paper: http://arxiv.org/pdf/2203.16795
- 代码/Code: None

**ReSTR: Convolution-free Referring Image Segmentation Using Transformers**

- 论文/Paper: http://arxiv.org/pdf/2203.16768
- 代码/Code: None

**TransRAC: Encoding Multi-scale Temporal Correlation with Transformers for Repetitive Action Counting**

- 论文/Paper: http://arxiv.org/pdf/2204.01018

- 代码/Code: https://github.com/sviprepetitioncounting/transrac

**Consistency driven Sequential Transformers Attention Model for Partially Observable Scenes**

- 论文/Paper: http://arxiv.org/pdf/2204.00656
- 代码/Code: None

**Multi-View Transformer for 3D Visual Grounding**

- 论文/Paper: http://arxiv.org/pdf/2204.02174
- 代码/Code: None

**Dual-AI: Dual-path Action Interaction Learning for Group Activity Recognition**

- 论文/Paper: http://arxiv.org/pdf/2204.02148
- 代码/Code: None

**Detector-Free Weakly Supervised Group Activity Recognition**

- 论文/Paper: http://arxiv.org/pdf/2204.02139
- 代码/Code: None

**Text Spotting Transformers**

- 论文/Paper: http://arxiv.org/pdf/2204.01918
- 代码/Code: None

**PSTR: End-to-End One-Step Person Search With Transformers**

- 论文/Paper: http://arxiv.org/pdf/2204.03340
- 代码/Code: https://github.com/jialecao001/pstr

**Consistency Learning via Decoding Path Augmentation for Transformers in Human Object Interaction Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.04836
- 代码/Code: https://github.com/mlvlab/CPChoi.

**Multimodal Transformer for Nursing Activity Recognition**

- 论文/Paper: http://arxiv.org/pdf/2204.04564
- 代码/Code: https://github.com/Momilijaz96/MMT_for_NCRC

**Learning Trajectory-Aware Transformer for Video Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2204.04216
- 代码/Code: https://github.com/researchmm/TTVSR

**TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.05525
- 代码/Code: https://github.com/hustvl/TopFormer

**Are Multimodal Transformers Robust to Missing Modality?**

- 论文/Paper: http://arxiv.org/pdf/2204.05454
- 代码/Code: None

**MiniViT: Compressing Vision Transformers with Weight Multiplexing**

- 论文/Paper: http://arxiv.org/pdf/2204.07154
- 代码/Code: https://github.com/microsoft/cream

**ViTOL: Vision Transformer for Weakly Supervised Object Localization**

- 论文/Paper: http://arxiv.org/pdf/2204.06772
- 代码/Code: https://github.com/Saurav-31/ViTOL

**Temporally Efficient Vision Transformer for Video Instance Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2204.08412
- 代码/Code: https://github.com/hustvl/TeViT.

**Safe Self-Refinement for Transformer-based Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2204.07683
- 代码/Code: None

**Multi-Frame Self-Supervised Depth with Transformers**

- 论文/Paper: http://arxiv.org/pdf/2204.07616
- 代码/Code: None

**Self-Calibrated Efficient Transformer for Lightweight Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2204.08913
- 代码/Code: https://github.com/AlexZou14/SCET.

**Multimodal Token Fusion for Vision Transformers**

- 论文/Paper: http://arxiv.org/pdf/2204.08721
- 代码/Code: None

**Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer  (Oral)**

- 论文/Paper: http://arxiv.org/pdf/2204.08680
- 代码/Code: https://github.com/zengwang430521/TCFormer

**NFormer: Robust Person Re-identification with Neighbor Transformer**

- 论文/Paper: http://arxiv.org/pdf/2204.09331
- 代码/Code: https://github.com/haochenheheda/NFormer

**Human-Object Interaction Detection via Disentangled Transformer**

- 论文/Paper: http://arxiv.org/pdf/2204.09290
- 代码/Code: None

**A New Dataset and Transformer for Stereoscopic Video Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2204.10039
- 代码/Code: https://github.com/H-deep/Trans-SVSR/

**Transformer Decoders with MultiModal Regularization for Cross-Modal Food Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2204.09730
- 代码/Code: https://github.com/mshukor/TFood

**Unsupervised Hierarchical Semantic Segmentation with Multiview Cosegmentation and Clustering Transformers**

- 论文/Paper: http://arxiv.org/pdf/2204.11432
- 代码/Code: https://github.com/twke18/HSG

**VISTA: Vision Transformer enhanced by U-Net and Image Colorfulness Frame Filtration for Automatic Retail Checkout**

- 论文/Paper: http://arxiv.org/pdf/2204.11024
- 代码/Code: None

**DearKD: Data-Efficient Early Knowledge Distillation for Vision Transformers**

- 论文/Paper: http://arxiv.org/pdf/2204.12997
- 代码/Code: None

**UTC: A Unified Transformer with Inter-Task Contrastive Learning for Visual Dialog**

- 论文/Paper: http://arxiv.org/pdf/2205.00423
- 代码/Code: None

**TransRank: Self-supervised Video Representation Learning via Ranking-based Transformation Recognition**

- 论文/Paper: http://arxiv.org/pdf/2205.02028
- 代码/Code: https://github.com/kennymckormick/TransRank

**Cross-view Transformers for real-time Map-view Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2205.02833
- 代码/Code: https://github.com/bradyz/cross_view_transformers

[返回目录/back](#Contents)

<a name="DepthEstimation"></a>

## 深度估计/Depth Estimation

**OACC-Net: Occlusion-Aware Cost Constructor for Light Field Depth Estimation**

- 论文/Paper: https://arxiv.org/pdf/2203.01576.pdf
- 代码/Code: https://github.com/YingqianWang/OACC-Net

**P3Depth: Monocular Depth Estimation with a Piecewise Planarity Prior**

- 论文/Paper: http://arxiv.org/pdf/2204.02091
- 代码/Code: None

**HiMODE: A Hybrid Monocular Omnidirectional Depth Estimation Model**

- 论文/Paper: http://arxiv.org/pdf/2204.05007
- 代码/Code: None

[返回目录/back](#Contents)

<a name="FaceRecognition"></a>

# 人脸识别/Face Recognition

**Adaface: Quality Adaptive Margin for Face Recognition**

- 论文/Paper: http://arxiv.org/pdf/2204.00964

- 代码/Code: https://github.com/mk-minchul/adaface

**WebFace260M: A Benchmark for Million-Scale Deep Face Recognition**

- 论文/Paper: http://arxiv.org/pdf/2204.10149
- 代码/Code: None

[返回目录/back](#Contents)

<a name="FaceDetection"></a>

# 人脸检测/Face Detection

**Privacy-preserving Online AutoML for Domain-Specific Face Detection**

- 论文/Paper：https://arxiv.org/abs/2203.08399
- 代码/Code：None

**Robust Neonatal Face Detection in Real-world Clinical Settings**

- 论文/Paper: http://arxiv.org/pdf/2204.00655

- 代码/Code: None

[返回目录/back](#Contents)

<a name="FaceAnti-Spoofing"></a>

# 人脸活体检测/Face Anti-Spoofing

**Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing**

- 论文/Paper: https://arxiv.org/abs/2203.05340
- 代码/Code: 

**PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition**

- 论文/Paper: http://arxiv.org/abs/2203.14325
- 代码/Code: None

**Self-supervised Learning of Adversarial Example: Towards Good Generalizations for Deepfake Detection**

- 论文/Paper: http://arxiv.org/abs/2203.12208
- 代码/Code: https://github.com/liangchen527/sladd

[返回目录/back](#Contents)

<a name="AgeEstimation"></a>

# 人脸年龄估计/Age Estimation



[返回目录/back](#Contents)

<a name="FacialExpressionRecognition"></a>

# 人脸表情识别/Facial Expression Recognition

**MDAN: Multi-level Dependent Attention Network for Visual Emotion Analysis**

- 论文/Paper: http://arxiv.org/pdf/2203.13443
- 代码/Code: None

**Towards Semi-Supervised Deep Facial Expression Recognition with An Adaptive Confidence Margin**

- 论文/Paper: http://arxiv.org/abs/2203.12341
- 代码/Code: https://github.com/hangyu94/ada-cm

[返回目录/back](#Contents)

<a name="FacialAttributeRecognition"></a>

# 人脸属性识别/Facial Attribute Recognition

**Fair Contrastive Learning for Facial Attribute Classification**

- 论文/Paper: http://arxiv.org/pdf/2203.16209
- 代码/Code: https://github.com/sungho-coolg/fscl

<a name="FacialEditing"></a>

## 人脸编辑/Facial Editing

**TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing**

- 论文/Paper: http://arxiv.org/pdf/2203.17266
- 代码/Code: https://github.com/BillyXYB/TransEditor

**Face Relighting with Geometrically Consistent Shadows**

- 论文/Paper: http://arxiv.org/pdf/2203.16681
- 代码/Code: None

**Escaping Data Scarcity for High-Resolution Heterogeneous Face Hallucination**

- 论文/Paper: http://arxiv.org/pdf/2203.16669
- 代码/Code: None

**EMOCA: Emotion Driven Monocular Face Capture and Animation**

- 论文/Paper: http://arxiv.org/pdf/2204.11312
- 代码/Code: None

[返回目录/back](#Contents)

<a name="FaceSwap"></a>

## 换脸/Face Swap

**High-resolution Face Swapping via Latent Semantics Disentanglement**

- 论文/Paper: http://arxiv.org/pdf/2203.15958
- 代码/Code: None



[返回目录/back](#Contents)

<a name="HumanPoseEstimation"></a>

# 人体姿态估计/Human Pose Estimation

**Capturing Humans in Motion: Temporal-Attentive 3D Human Pose and Shape Estimation from Monocular Video**

- 论文/Paper：https://arxiv.org/abs/2203.08534
- 代码/Code：https://mps-net.github.io/MPS-Net/

**DiffPoseNet: Direct Differentiable Camera Pose Estimation**

- 论文/Paper: http://arxiv.org/abs/2203.11174
- 代码/Code: None

**EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation**

- 论文/Paper: http://arxiv.org/abs/2203.13254
- 代码/Code: None

**GPV-Pose: Category-level Object Pose Estimation via Geometry-guided Point-wise Voting**

- 论文/Paper：https://arxiv.org/abs/2203.07918
- 代码/Code：

**MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video**

- 论文/Paper：https://arxiv.org/abs/2203.00859
- 代码/Code：https://github.com/JinluZhang1126/MixSTE

**MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation**

- 论文/Paper: https://arxiv.org/pdf/2111.12707.pdf
- 代码/Code: https://github.com/Vegetebird/MHFormer

**OSOP: A Multi-Stage One Shot Object Pose Estimation Framework**

- 论文/Paper: http://arxiv.org/pdf/2203.15533
- 代码/Code: None

**Temporal Feature Alignment and Mutual Information Maximization for Video-Based Human Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2203.15227
- 代码/Code: None

**PoseTriplet: Co-evolving 3D Human Pose Estimation, Imitation, and Hallucination under Self-supervision**

- 论文/Paper: http://arxiv.org/pdf/2203.15625
- 代码/Code: None

**Ray3D: ray-based 3D human pose estimation for monocular absolute 3D localization**

- 论文/Paper: http://arxiv.org/abs/2203.11471
- 代码/Code: https://github.com/YxZhxn/Ray3D

**Uncertainty-Aware Adaptation for Self-Supervised 3D Human Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2203.15293
- 代码/Code: None

**Templates for 3D Object Pose Estimation Revisited: Generalization to New Objects and Robustness to Occlusions**

- 论文/Paper: http://arxiv.org/pdf/2203.17234
- 代码/Code: None

**Focal Length and Object Pose Estimation via Render and Compare**

- 论文/Paper: http://arxiv.org/pdf/2204.05145
- 代码/Code: http://github.com/ponimatkin/focalpose

**DGECN: A Depth-Guided Edge Convolutional Network for End-to-End 6D Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2204.09983
- 代码/Code: None

**Coupled Iterative Refinement for 6D Multi-Object Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2204.12516
- 代码/Code: https://github.com/princeton-vl/Coupled-Iterative-Refinement.

[返回目录/back](#Contents)

<a name="6DPoseEstimation"></a>

# 6D位姿估计 /6D Pose Estimation

**FS6D: Few-Shot 6D Pose Estimation of Novel Objects**

- 论文/Paper: http://arxiv.org/abs/2203.14628
- 代码/Code: None

**Uni6D: A Unified CNN Framework without Projection Breakdown for 6D Pose Estimation**

- 论文/Paper: http://arxiv.org/abs/2203.14531
- 代码/Code: None

**ZebraPose: Coarse to Fine Surface Encoding for 6DoF Object Pose Estimation**

- 论文/Paper：https://arxiv.org/abs/2203.09418
- 代码/Code：

**RNNPose: Recurrent 6-DoF Object Pose Refinement with Robust Correspondence Field Estimation and Pose Optimization**

- 论文/Paper: http://arxiv.org/abs/2203.12870
- 代码/Code: None

**ES6D: A Computation Efficient and Symmetry-Aware 6D Pose Regression Framework**

- 论文/Paper: http://arxiv.org/pdf/2204.01080

- 代码/Code: None

[返回目录/back](#Contents)

<a name="HandPoseEstimation"></a>

## 手势姿态估计（重建）/Hand Pose Estimation( Hand Mesh Recovery



[返回目录/back](#Contents)

<a name="VideoActionDetection"></a>

## 视频动作检测/Video Action Detection

**DirecFormer: A Directed Attention in Transformer Approach to Robust Action Recognition**

- 论文/Paper: http://arxiv.org/abs/2203.10233
- 代码/Code: None

**End-to-End Semi-Supervised Learning for Video Action Detection**

- 论文/Paper：https://arxiv.org/abs/2203.04251
- 代码/Code：

**How Do You Do It? Fine-Grained Action Understanding with Pseudo-Adverbs**

- 论文/Paper: http://arxiv.org/abs/2203.12344
- 代码/Code: https://github.com/hazeld/pseudoadverbs

**Look for the Change: Learning Object States and State-Modifying Actions from Untrimmed Web Videos**

- 论文/Paper: http://arxiv.org/abs/2203.11637
- 代码/Code: https://github.com/soCzech/LookForTheChange

**RCL: Recurrent Continuous Localization for Temporal Action Detection**

- 论文/Paper：https://arxiv.org/abs/2203.07112
- 代码/Code：

**SPAct: Self-supervised Privacy Preservation for Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2203.15205
- 代码/Code: None

**An Empirical Study of End-to-End Temporal Action Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.02932
- 代码/Code: https://github.com/xlliu7/E2E-TAD

**SOS! Self-supervised Learning Over Sets Of Handled Objects In Egocentric Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2204.04796
- 代码/Code: None

**Video Action Detection: Analysing Limitations and Challenges**

- 论文/Paper: http://arxiv.org/pdf/2204.07892
- 代码/Code: None

**Hybrid Relation Guided Set Matching for Few-shot Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2204.13423
- 代码/Code: None

**Cross-modal Representation Learning for Zero-shot Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2205.01657
- 代码/Code: None

[返回目录/back](#Contents)

<a name="SignLanguageTranslation"></a>

## 手语翻译/Sign Language Translation

**A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation**

- 论文/Paper：https://arxiv.org/abs/2203.04287
- 代码/Code：

[返回目录/back](#Contents)

<a name="3D人体重建"></a>

## 3D人体重建/Person Reconstruction

**ImFace: A Nonlinear 3D Morphable Face Model with Implicit Neural Representations**

- 论文/Paper: http://arxiv.org/abs/2203.14510
- 代码/Code: None

**AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation**

- 论文/Paper：https://arxiv.org/abs/2203.09516
- 代码/Code：https://yccyenchicheng.github.io/AutoSDF/

**Learning Motion-Dependent Appearance for High-Fidelity Rendering of Dynamic Humans from a Single Camera**

- 论文/Paper: http://arxiv.org/abs/2203.12780
- 代码/Code: None

**MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation**

- 论文/Paper: https://arxiv.org/pdf/2111.12707.pdf

- 代码/Code: https://github.com/Vegetebird/MHFormer

**OcclusionFusion: Occlusion-aware Motion Estimation for Real-time Dynamic 3D Reconstruction**

- 论文/Paper：https://arxiv.org/abs/2203.07977
- 代码/Code：https://wenbin-lin.github.io/OcclusionFusion

**Structured Local Radiance Fields for Human Avatar Modeling**

- 论文/Paper: http://arxiv.org/abs/2203.14478
- 代码/Code: None

**JIFF: Jointly-aligned Implicit Face Function for High Quality Single View Clothed Human Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2204.10549
- 代码/Code: None

[返回目录/back](#Contents)

<a name="PersonRe-identification"></a>

## 行人重识别/Person Re-identification

**Camera-Conditioned Stable Feature Generation for Isolated Camera Supervised Person Re-IDentification**

- 论文/Paper: http://arxiv.org/pdf/2203.15210
- 代码/Code: None

**Part-based Pseudo Label Refinement for Unsupervised Person Re-identification**

- 论文/Paper: http://arxiv.org/abs/2203.14675
- 代码/Code: None

**Cloning Outfits from Real-World Images to 3D Characters for Generalizable Person Re-Identification**

- 论文/Paper: http://arxiv.org/pdf/2204.02611
- 代码/Code: https://github.com/Yanan-Wang-cs/ClonedPerson

**Implicit Sample Extension for Unsupervised Person Re-Identification**

- 论文/Paper: http://arxiv.org/pdf/2204.06892
- 代码/Code: https://github.com/PaddlePaddle/PaddleClas

**Clothes-Changing Person Re-identification with RGB Modality Only**

- 论文/Paper: http://arxiv.org/pdf/2204.06890
- 代码/Code: https://github.com/guxinqian/Simple-CCReID.

**NFormer: Robust Person Re-identification with Neighbor Transformer**

- 论文/Paper: http://arxiv.org/pdf/2204.09331
- 代码/Code: https://github.com/haochenheheda/NFormer

[返回目录/back](#Contents)

<a name="PersonSearch"></a>

# 行人搜索/Person Search



[返回目录/back](#Contents)

<a name="CrowdCounting"></a>

## 人群计数 / Crowd Counting

**Cross-View Cross-Scene Multi-View Crowd Counting**

- 论文/Paper: http://arxiv.org/pdf/2205.01551
- 代码/Code: None

[返回目录/back](#Contents)

<a name="GAN"></a>

## GAN

**A Style-aware Discriminator for Controllable Image Translation**

- 论文/Paper: http://arxiv.org/pdf/2203.15375
- 代码/Code: None

**Attribute Group Editing for Reliable Few-shot Image Generation**

- 论文/Paper：https://arxiv.org/abs/2203.08422
- 代码/Code：

**Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory**

- 论文/Paper: http://arxiv.org/abs/2203.13055
- 代码/Code: None

**Compound Domain Generalization via Meta-Knowledge Encoding**

- 论文/Paper: http://arxiv.org/abs/2203.13006
- 代码/Code: None

**Diverse Plausible 360-Degree Image Outpainting for Efficient 3DCG Background Creation**

- 论文/Paper: http://arxiv.org/abs/2203.14668
- 代码/Code: None

**Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization**

- 论文/Paper：https://arxiv.org/abs/2203.07740
- 代码/Code：https://github.com/YBZh/EFDM

**FlexIT: Towards Flexible Semantic Image Translation** 

- 论文/paper：https://arxiv.org/abs/2203.04705 
- 代码/code：

**GCFSR: a Generative and Controllable Face Super Resolution Method Without Facial and GAN Priors**

- 论文/Paper：https://arxiv.org/abs/2203.07319

- 代码/Code：

**GAN-Supervised Dense Visual Alignment** 

- 论文/paper：https://arxiv.org/abs/2112.05143
- 代码/code：https://github.com/wpeebles/gangealing

**GIRAFFE HD: A High-Resolution 3D-aware Generative Model**

- 论文/Paper: http://arxiv.org/abs/2203.14954
- 代码/Code: None

**HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing**

- 论文/paper：https://arxiv.org/abs/2111.15666 | [主页](https://yuval-alaluf.github.io/hyperstyle/)
- 代码/code：https://github.com/yuval-alaluf/hyperstyle

**Look Outside the Room: Synthesizing A Consistent Long-Term 3D Scene Video from A Single Image**

- 论文/Paper：https://arxiv.org/abs/2203.09457
- 代码/Code：https://xrenaa.github.io/look-outside-room/

**Modulated Contrast for Versatile Image Synthesis**

- 论文/Paper：https://arxiv.org/abs/2203.09333
- 代码/Code：https://github.com/fnzhan/MoNCE

**Maximum Spatial Perturbation Consistency for Unpaired Image-to-Image Translation**

- 论文/Paper: http://arxiv.org/abs/2203.12707
- 代码/Code: None

**Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer**

- 论文/Paper: http://arxiv.org/abs/2203.13248
- 代码/Code: None

**QS-Attn: Query-Selected Attention for Contrastive Learning in I2I Translation**

- 论文/Paperhttps://arxiv.org/abs/2203.08483
- 代码/Code：

**RGB-Depth Fusion GAN for Indoor Depth Completion**

- 论文/Paper: http://arxiv.org/abs/2203.10856
- 代码/Code: None

**Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation**

- 论文/Paper: http://arxiv.org/pdf/2203.09811
- 代码/Code: https://github.com/dongxingning/sha-gcl-for-sgg

**Style Transformer for Image Inversion and Editing**

- 论文/Paper：https://arxiv.org/abs/2203.07932
- 代码/Code：

**Unsupervised Domain Adaptation for Nighttime Aerial Tracking**

- 论文/Paper: http://arxiv.org/abs/2203.10541
- 代码/Code: None

**Wavelet Knowledge Distillation: Towards Efficient Image-to-Image Translation**

- 论文/Paper：https://arxiv.org/abs/2203.06321
- 代码/Code：

**Industrial Style Transfer with Large-scale Geometric Warping and Content Preservation**

- 论文/Paper: http://arxiv.org/abs/2203.12835
- 代码/Code: None

**TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing**

- 论文/Paper: http://arxiv.org/pdf/2203.17266
- 代码/Code: https://github.com/BillyXYB/TransEditor

**TransEditor: Transformer-Based Dual-Space GAN for Highly Controllable Facial Editing**

- 论文/Paper: http://arxiv.org/pdf/2203.17266
- 代码/Code: https://github.com/BillyXYB/TransEditor

**Marginal Contrastive Correspondence for Guided Image Generation**

- 论文/Paper: http://arxiv.org/pdf/2204.00442
- 代码/Code: None

**Style-Based Global Appearance Flow for Virtual Try-On**

- 论文/Paper: http://arxiv.org/pdf/2204.01046

- 代码/Code: https://github.com/senhe/flow-style-vton

**Arbitrary-Scale Image Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2204.02273
- 代码/Code: https://github.com/vglsd/ScaleParty

**Unsupervised Image-to-Image Translation with Generative Prior**

- 论文/Paper: http://arxiv.org/pdf/2204.03641
- 代码/Code: https://github.com/williamyang1991/gp-unit

**Commonality in Natural Images Rescues GANs: Pretraining GANs with Generic and Privacy-free Synthetic Data**

- 论文/Paper: http://arxiv.org/pdf/2204.04950
- 代码/Code: None

**medXGAN: Visual Explanations for Medical Classifiers through a Generative Latent Space**

- 论文/Paper: http://arxiv.org/abs/2204.05376
- 代码/Code: https://github.com/avdravid/medXGAN_explanations

**Multi-View Consistent Generative Adversarial Networks for 3D-aware Image Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2204.06307
- 代码/Code: None

**Dual-Domain Image Synthesis using Segmentation-Guided GAN**

- 论文/Paper: http://arxiv.org/pdf/2204.09015
- 代码/Code: https://github.com/denabazazian/Dual-Domain-Synthesis.

**ClothFormer:Taming Video Virtual Try-on in All Module**

- 论文/Paper: http://arxiv.org/abs/2204.12151
- 代码/Code: None

**OSSGAN: Open-Set Semi-Supervised Image Generation**

- 论文/Paper: http://arxiv.org/pdf/2204.14249
- 代码/Code: https://github.com/raven38/ossgan

**Fix the Noise: Disentangling Source Feature for Transfer Learning of StyleGAN**

- 论文/Paper: http://arxiv.org/pdf/2204.14079
- 代码/Code: None

**GenDR: A Generalized Differentiable Renderer**

- 论文/Paper: http://arxiv.org/pdf/2204.13845
- 代码/Code: https://github.com/Felix-Petersen/gendr

**HL-Net: Heterophily Learning Network for Scene Graph Generation**

- 论文/Paper: http://arxiv.org/pdf/2205.01316
- 代码/Code: https://github.com/siml3/HL-Net.

**RU-Net: Regularized Unrolling Network for Scene Graph Generation**

- 论文/Paper: http://arxiv.org/pdf/2205.01297
- 代码/Code: https://github.com/siml3/RU-Net

**Comparison of CoModGANs, LaMa and GLIDE for Art Inpainting- Completing M.C Escher's Print Gallery**

- 论文/Paper: http://arxiv.org/pdf/2205.01741
- 代码/Code: None

**Generate and Edit Your Own Character in a Canonical View**

- 论文/Paper: http://arxiv.org/pdf/2205.02974
- 代码/Code: None

**Scene Graph Expansion for Semantics-Guided Image Outpainting**

- 论文/Paper: http://arxiv.org/pdf/2205.02958
- 代码/Code: None

[返回目录/back](#Contents)

<a name="CPM"></a>

## 彩妆迁移 / Color-Pattern Makeup Transfer



[返回目录/back](#Contents)

<a name="FontGeneration"></a>

## 字体生成 / Font Generation



[返回目录/back](#Contents)

<a name="OCR"></a>

## OCR

**Fourier Document Restoration for Robust Document Dewarping and Recognition**

- 论文/Paper: http://arxiv.org/pdf/2203.09910
- 代码/Code: None

**SimAN: Exploring Self-Supervised Representation Learning of Scene Text via Similarity-Aware Normalization**

- 论文/Paper: http://arxiv.org/abs/2203.10492
- 代码/Code: None

### **文字图像处理（超分辨率增强、文字分割、文档版面分析）**

**A Text Attention Network for Spatial Deformation Robust Scene Text Image Super-resolution**

- 论文/Paper：https://arxiv.org/abs/2203.09388
- 代码/Code：https://github.com/mjq11302010044/TATT

### 场景文本检测、识别/Scene Text Detection/Recognition

**Kernel Proposal Network for Arbitrary Shape Text Detection**

- 论文/Paper：https://arxiv.org/abs/2203.06410
- 代码/Code：https://github.com/GXYM/KPN

**SwinTextSpotter: Scene Text Spotting via Better Synergy between Text Detection and Text Recognition**

- 论文/Paper: http://arxiv.org/abs/2203.10209
- 代码/Code: None

**Towards End-to-End Unified Scene Text Detection and Layout Analysis**

- 论文/Paper: http://arxiv.org/pdf/2203.15143
- 代码/Code: None

**Pushing the Performance Limit of Scene Text Recognizer without Human Annotation**

- 论文/Paper: http://arxiv.org/pdf/2204.07714
- 代码/Code: None

**Vision-Language Pre-Training for Boosting Scene Text Detectors**

- 论文/Paper: http://arxiv.org/pdf/2204.13867
- 代码/Code: None

### **端到端文字识别**

**Open-set Text Recognition via Character-Context Decoupling**

- 论文/Paper: http://arxiv.org/pdf/2204.05535
- 代码/Code: None

### **手写文字分析与识别**



### **其它（文档图像预训练模型，Text VQA、数据集，Retrieval , 应用）**





[返回目录/back](#Contents)

<a name="Retrieval"></a>

## 图像、视频检索 / Image Retrieval/Video retrieval

**Correlation Verification for Image Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2204.01458

- 代码/Code: https://github.com/sungonce/cvnet

**Sketching without Worrying: Noise-Tolerant Sketch-Based Image Retrieval**

- 论文/Paper: http://arxiv.org/abs/2203.14817
- 代码/Code: None

**Beyond Cross-view Image Retrieval: Highly Accurate Vehicle Localization Using Satellite Image**

- 论文/Paper: http://arxiv.org/pdf/2204.04752
- 代码/Code: None

**Probabilistic Compositional Embeddings for Multimodal Image Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2204.05845
- 代码/Code: https://github.com/andreineculai/MPC.

[返回目录/back](#Contents)

<a name="ImageAnimation"></a>

## Image Animation

**Thin-Plate Spline Motion Model for Image Animation**

- 论文/Paper: http://arxiv.org/abs/2203.14367
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ImageMatting"></a>

## 抠图/Image Matting/Video Matting



[返回目录/back](#Contents)

<a name="SuperResolution"></a>

# 超分辨率/Super Resolution

**Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution**

- 论文/Paper：https://arxiv.org/abs/2203.09195
- 代码/Code：https://github.com/csjliang/LDL

**Learning Graph Regularisation for Guided Super-Resolution**

- 论文/Paper: http://arxiv.org/abs/2203.14297
- 代码/Code: None

**Reflash Dropout in Image Super-Resolution**

- 论文/Paper：https://arxiv.org/pdf/2112.12089.pdf
- 代码/Code：https://github.com/Xiangtaokong/Reflash-Dropout-in-Image-Super-Resolution

**Look Back and Forth: Video Super-Resolution with Explicit Temporal Difference Modeling**

- 论文/Paper: http://arxiv.org/pdf/2204.07114
- 代码/Code: None

**Fast and Memory-Efficient Network Towards Efficient Image Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2204.08397
- 代码/Code: https://github.com/NJU-Jet/FMEN.

**Self-Calibrated Efficient Transformer for Lightweight Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2204.08913
- 代码/Code: https://github.com/AlexZou14/SCET.

**Edge-enhanced Feature Distillation Network for Efficient Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2204.08759
- 代码/Code: https://github.com/icandle/EFDN.

**A New Dataset and Transformer for Stereoscopic Video Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2204.10039
- 代码/Code: https://github.com/H-deep/Trans-SVSR/

**FS-NCSR: Increasing Diversity of the Super-Resolution Space via Frequency Separation and Noise-Conditioned Normalizing Flow**

- 论文/Paper: http://arxiv.org/pdf/2204.09679
- 代码/Code: None

**IMDeception: Grouped Information Distilling Super-Resolution Network**

- 论文/Paper: http://arxiv.org/pdf/2204.11463
- 代码/Code: None

**Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites**

- 论文/Paper: http://arxiv.org/pdf/2205.02031
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ImageRestoration"></a>

# 图像复原/Image Restoration

**Exploring and Evaluating Image Restoration Potential in Dynamic Scenes**

- 论文/Paper: http://arxiv.org/abs/2203.11754
- 代码/Code: None

**Interacting Attention Graph for Single Image Two-Hand Reconstruction**

- 论文/Paper：https://arxiv.org/abs/2203.09364
- 代码/Code：https://github.com/Dw1010/IntagHand

**Deep Generalized Unfolding Networks for Image Restoration**

- 论文/Paper: http://arxiv.org/pdf/2204.13348
- 代码/Code: https://github.com/MC-E/Deep-Generalized-Unfolding-Networks-for-Image-Restoration.

[返回目录/back](#Contents)

<a name="ImageInpainting"></a>

# 图像补全/Image Inpainting

**Bridging Global Context Interactions for High-Fidelity Image Completion**

- 论文/Paper：https://arxiv.org/abs/2104.00845
- 代码/Code：https://github.com/lyndonzheng/TFill

**MAT: Mask-Aware Transformer for Large Hole Image Inpainting**

- 论文/Paper: http://arxiv.org/pdf/2203.15270
- 代码/Code: None

**MISF: Multi-level Interactive Siamese Filtering for High-Fidelity Image Inpainting**

- 论文/Paper：https://arxiv.org/abs/2203.06304
- 代码/Code：https://github.com/tsingqguo/misf

**Towards An End-to-End Framework for Flow-Guided Video Inpainting**

- 论文/Paper: http://arxiv.org/pdf/2204.02663
- 代码/Code: https://github.com/MCG-NKU/E2FGVI

[返回目录/back](#Contents)

<a name="ImageDenoising"></a>

## 图像去噪/Image Denoising

**AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network**

- 论文/Paper: http://arxiv.org/abs/2203.11799
- 代码/Code: None

**Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots**

- 论文/Paper：https://arxiv.org/abs/2203.06967
- 代码/Code：https://github.com/demonsjin/Blind2Unblind

**CVF-SID: Cyclic multi-Variate Function for Self-Supervised Image Denoising by Disentangling Noise from Image**

- 论文/Paper: http://arxiv.org/abs/2203.13009
- 代码/Code: None

**Learning to Deblur using Light Field Generated and Real Defocus Images**

- 论文/Paper: http://arxiv.org/pdf/2204.00367
- 代码/Code: https://github.com/lingyanruan/DRBNet

**Dancing under the stars: video denoising in starlight**

- 论文/Paper: http://arxiv.org/abs/2204.04210
- 代码/Code: None

**Multiple Degradation and Reconstruction Network for Single Image Denoising via Knowledge Distillation**

- 论文/Paper: http://arxiv.org/pdf/2204.13873
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ImageEditing"></a>

# 图像编辑/Image Editing



[返回目录/back](#Contents)

<a name="Imagestitching"></a>

# 图像拼接/Image stitching

**Deep Rectangling for Image Stitching: A Learning Baseline**

- 论文/Paper：https://arxiv.org/abs/2203.03831
- 代码/Code：https://github.com/nie-lang/DeepRectangling

[返回目录/back](#Contents)

<a name="ImageMatching"></a>

# 图像匹配/Image Matching



[返回目录/back](#Contents)

<a name="ImageBlending"></a>

# 图像融合/Image Blending



[返回目录/back](#Contents)

<a name="ImageDehazing"></a>

## 图像去雾/Image Dehazing



[返回目录/back](#Contents)

<a name="ImageCompression"></a>

## 图像压缩/Image Compression

**ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding**

- 论文/Paper: http://arxiv.org/abs/2203.10886
- 代码/Code: None

**Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression**

- 论文/Paper: http://arxiv.org/abs/2203.10897
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ReflectionRemoval"></a>

## 反光去除/Reflection Removal



[返回目录/back](#Contents)

<a name="LaneDetection"></a>

## 车道线检测/Lane Detection

**CLRNet: Cross Layer Refinement Network for Lane Detection**

- 论文/Paper: http://arxiv.org/abs/2203.10350
- 代码/Code: None

**Eigenlanes: Data-Driven Lane Descriptors for Structurally Diverse Lanes**

- 论文/Paper: http://arxiv.org/pdf/2203.15302
- 代码/Code: None

**Rethinking Efficient Lane Detection via Curve Modeling**

- 论文/Paper：https://arxiv.org/abs/2203.02431
- 代码/Code：https://github.com/voldemortX/pytorch-auto-drive

**Towards Driving-Oriented Metric for Lane Detection Models**

- 论文/Paper: http://arxiv.org/pdf/2203.16851
- 代码/Code: None

**A Keypoint-based Global Association Network for Lane Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.07335
- 代码/Code: https://github.com/Wolfwjs/GANet.

**ONCE-3DLanes: Building Monocular 3D Lane Detection**

- 论文/Paper: http://arxiv.org/pdf/2205.00301
- 代码/Code: None

[返回目录/back](#Contents)

<a name="AutonomousDriving"></a>

## 自动驾驶 / Autonomous Driving

**Learning from All Vehicles**

- 论文/Paper: http://arxiv.org/abs/2203.11934
- 代码/Code: https://github.com/dotchen/LAV

[返回目录/back](#Contents)

<a name="FluidReconstruction"></a>

## 流体重建/Fluid Reconstruction

[返回目录/back](#Contents)

<a name="SceneReconstruction"></a>

## 场景重建 / Scene Reconstruction

**3D Shape Reconstruction from 2D Images with Disentangled Attribute Flow**

- 论文/Paper: http://arxiv.org/pdf/2203.15190
- 代码/Code: None

**NeRFusion: Fusing Radiance Fields for Large-Scale Scene Reconstruction**

- 论文/Paper: http://arxiv.org/abs/2203.11283
- 代码/Code: None

**PlaneMVS: 3D Plane Reconstruction from Multi-View Stereo**

- 论文/Paper: http://arxiv.org/abs/2203.12082
- 代码/Code: None

**Neural 3D Scene Reconstruction with the Manhattan-world Assumption**

- 论文/Paper: http://arxiv.org/pdf/2205.02836
- 代码/Code: https://github.com/zju3dv/manhattan_sdf

[返回目录/back](#Contents)

<a name="FrameInterpolation"></a>

## 视频插帧/Frame Interpolation

**Long-term Video Frame Interpolation via Feature Propagation**

- 论文/Paper: http://arxiv.org/pdf/2203.15427
- 代码/Code: None

**TimeReplayer: Unlocking the Potential of Event Cameras for Video Interpolation**

- 论文/Paper: http://arxiv.org/abs/2203.13859
- 代码/Code: None

**Unifying Motion Deblurring and Frame Interpolation with Events**

- 论文/Paper: http://arxiv.org/abs/2203.12178
- 代码/Code: None

**Time Lens++: Event-based Frame Interpolation with Parametric Non-linear Flow and Multi-scale Fusion**

- 论文/Paper: http://arxiv.org/pdf/2203.17191
- 代码/Code: None

**Many-to-many Splatting for Efficient Video Frame Interpolation**

- 论文/Paper: http://arxiv.org/pdf/2204.03513
- 代码/Code: https://github.com/feinanshan/m2m_vfi

[返回目录/back](#Contents)

<a name="VideoSuper-Resolution"></a>

## 视频超分 / Video Super-Resolution

**Reference-based Video Super-Resolution Using Multi-Camera Video Triplets**

- 论文/Paper: http://arxiv.org/abs/2203.14537
- 代码/Code: None

[返回目录/back](#Contents)

<a name="3DPointCloud"></a>

## 3D点云/3D point cloud

**ART-Point: Improving Rotation Robustness of Point Cloud Classifiers via Adversarial Rotation**

- 论文/Paper：https://arxiv.org/abs/2203.03888
- 代码/Code：

**AziNorm: Exploiting the Radial Symmetry of Point Cloud for Azimuth-Normalized 3D Perception**

- 论文/Paper: http://arxiv.org/abs/2203.13090
- 代码/Code: None

**Contrastive Boundary Learning for Point Cloud Segmentation**

- 论文/Paper：https://arxiv.org/abs/2203.05272
- 代码/Code：https://github.com/LiyaoTang/contrastBoundary

**Equivariant Point Cloud Analysis via Learning Orientations for Message Passing**

- 论文/Paper: http://arxiv.org/abs/2203.14486
- 代码/Code: None

**IDEA-Net: Dynamic 3D Point Cloud Interpolation via Deep Embedding Alignment**

- 论文/Paper: http://arxiv.org/abs/2203.11590
- 代码/Code: None

**Learning a Structured Latent Space for Unsupervised Point Cloud Completion**

- 论文/Paper: http://arxiv.org/pdf/2203.15580
- 代码/Code: None

**Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds**

- 论文/Paper: http://arxiv.org/abs/2203.11139
- 代码/Code: None

**No Pain, Big Gain: Classify Dynamic Point Cloud Sequences with Static Models by Fitting Feature-level Space-time Surfaces**

- 论文/Paper: http://arxiv.org/abs/2203.11113
- 代码/Code: None

**REGTR: End-to-end Point Cloud Correspondences with Transformers**

- 论文/Paper: http://arxiv.org/abs/2203.14517
- 代码/Code: None

**SC^2-PCR: A Second Order Spatial Compatibility for Efficient and Robust Point Cloud Registration**

- 论文/Paper: http://arxiv.org/abs/2203.14453
- 代码/Code: None

**Stratified Transformer for 3D Point Cloud Segmentation**

- 论文/Paper: http://arxiv.org/abs/2203.14508
- 代码/Code: None

**Shape-invariant 3D Adversarial Point Clouds**

- 论文/Paper：https://arxiv.org/abs/2203.04041
- 代码/Code：https://github.com/shikiw/SI-Adv

**WarpingGAN: Warping Multiple Uniform Priors for Adversarial 3D Point Cloud Generation**

- 论文/Paper: http://arxiv.org/abs/2203.12917
- 代码/Code: None

**Deformation and Correspondence Aware Unsupervised Synthetic-to-Real Scene Flow Estimation for Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2203.16895
- 代码/Code: None

**Deformation and Correspondence Aware Unsupervised Synthetic-to-Real Scene Flow Estimation for Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2203.16895
- 代码/Code: None

**Learning Local Displacements for Point Cloud Completion**

- 论文/Paper: http://arxiv.org/pdf/2203.16600
- 代码/Code: None

**3DeformRS: Certifying Spatial Deformations on Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2204.05687
- 代码/Code: None

**Reconstructing Surfaces for Sparse Point Clouds with On-Surface Priors**

- 论文/Paper: http://arxiv.org/pdf/2204.10603
- 代码/Code: https://github.com/mabaorui/onsurfaceprior

**Surface Reconstruction from Point Clouds by Learning Predictive Context Priors**

- 论文/Paper: http://arxiv.org/pdf/2204.11015
- 代码/Code: None

**Density-preserving Deep Point Cloud Compression**

- 论文/Paper: http://arxiv.org/pdf/2204.12684
- 代码/Code: None

**Why Discard if You can Recycle?: A Recycling Max Pooling Module for 3D Point Cloud Analysis**

- 论文/Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Why_Discard_if_You_Can_Recycle_A_Recycling_Max_Pooling_CVPR_2022_paper.pdf
- 代码/Code: https://github.com/jiajingchen113322/Recycle_Maxpooling_Module

**Boosting 3D Object Detection by Simulating Multimodality on Point Clouds**

- 论文/Paper: https://arxiv.org/abs/2206.14971
- 代码/Code: None

[返回目录/back](#Contents)

<a name="Label-Noise"></a>

# 标签噪声 Label-Noise



[返回目录/back](#Contents)

<a name="AdversarialExamples"></a>

# 对抗样本 / Adversarial Examples

**LAS-AT: Adversarial Training with Learnable Attack Strategy**

- 论文/Paper：https://arxiv.org/pdf/2203.06616.pdf

- 代码/Code：https://github.com/jiaxiaojunQAQ/LAS-AT

 [返回目录/back](#Contents)

<a name="Other"></a>

## 其他/Other

**DINE: Domain Adaptation from Single and Multiple Black-box Predictors**

- 论文/Paper：https://arxiv.org/abs/2104.01539

- 代码/Code：https://github.com/tim-learn/DINE

**It's About Time: Analog clock Reading in the Wild**

- 论文/Paper：https://arxiv.org/abs/2111.09162
- 代码/Code：https://github.com/charigyang/itsabouttime

**Neural Face Identification in a 2D Wireframe Projection of a Manifold Object**

- 论文/Paper：https://arxiv.org/abs/2203.04229 | [主页](https://manycore-research.github.io/faceformer/)
- 代码/Code：https://github.com/manycore-research/faceformer

**Probabilistic Warp Consistency for Weakly-Supervised Semantic Correspondences**

- 论文/Paper：https://arxiv.org/abs/2203.04279
- 代码/Code：https://github.com/PruneTruong/DenseMatching

**TeachAugment: Data Augmentation Optimization Using Teacher Knowledge**

- 论文/Paper：https://arxiv.org/abs/2202.12513

- 代码/Code：https://github.com/DensoITLab/TeachAugment

**UKPGAN: Unsupervised KeyPoint GANeration**

- 论文/Paper：

- 代码/Code：https://github.com/qq456cvb/UKPGAN

**DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos**

- 论文/Paper：https://arxiv.org/abs/2203.03996
- 代码/Code：

**Generative Cooperative Learning for Unsupervised Video Anomaly Detection**

- 论文/Paper：https://arxiv.org/abs/2203.03962
- 代码/Code：

**Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon**

- 论文/Paper：https://arxiv.org/abs/2203.03818
- 代码/Code：

**Unknown-Aware Object Detection: Learning What You Don't Know from Videos in the Wild**

- 论文/Paper：https://arxiv.org/abs/2203.03800
- 代码/Code：https://github.com/deeplearning-wisc/stud

**On Generalizing Beyond Domains in Cross-Domain Continual Learning**

- 论文/Paper：https://arxiv.org/abs/2203.03970
- 代码/Code：

**Generating 3D Bio-Printable Patches Using Wound Segmentation and Reconstruction to Treat Diabetic Foot Ulcers**

- 论文/Paper：https://arxiv.org/abs/2203.03814
- 代码/Code：

**What Matters For Meta-Learning Vision Regression Tasks?**

- 论文/Paper：https://arxiv.org/abs/2203.04905
- 代码/Code：

**ChiTransformer:Towards Reliable Stereo from Cues**

- 论文/Paper：https://arxiv.org/abs/2203.04554

- 代码/Code：

**Dynamic Dual-Output Diffusion Models**

- 论文/Paper：https://arxiv.org/abs/2203.04304

- 代码/Code：

**Spatial Commonsense Graph for Object Localisation in Partial Scenes**

- 论文/Paper：https://arxiv.org/abs/2203.05380

- 代码/Code：https://fgiuliari.github.io/projects/SpatialCommonsenseGraph/

**Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack**

- 论文/Paper：https://arxiv.org/abs/2203.05154

- 代码/Code：https://github.com/liuye6666/adaptive_auto_attack

**Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity**

- 论文/Paper：https://arxiv.org/abs/2203.05151

- 代码/Code： 

**REX: Reasoning-aware and Grounded Explanation**

- 论文/Paper：https://arxiv.org/abs/2203.06107
- 代码/Code：

**FLAG: Flow-based 3D Avatar Generation from Sparse Observations**

- 论文/Paper：https://arxiv.org/abs/2203.05789
- 代码/Code：

**Learning Distinctive Margin toward Active Domain Adaptation**

- 论文/Paper：https://arxiv.org/abs/2203.05738
- 代码/Code：https://github.com/TencentYoutuResearch/ActiveLearning-SDM

**Active Learning by Feature Mixing**

- 论文/Paper：https://arxiv.org/abs/2203.07034
- 代码/Code：

**UniVIP: A Unified Framework for Self-Supervised Visual Pre-training**

- 论文/Paper：https://arxiv.org/abs/2203.06965
- 代码/Code：

**Forward Compatible Few-Shot Class-Incremental Learning**

- 论文/Paper：https://arxiv.org/abs/2203.06953
- 代码/Code：https://github.com/zhoudw-zdw/CVPR22-Fact

**XYLayoutLM: Towards Layout-Aware Multimodal Networks For Visually-Rich Document Understanding**

- 论文/Paper：https://arxiv.org/abs/2203.06947
- 代码/Code：

**Accelerating DETR Convergence via Semantic-Aligned Matching**

- 论文/Paper：https://arxiv.org/abs/2203.06883
- 代码/Code：https://github.com/ZhangGongjie/SAM-DETR

**ADAS: A Direct Adaptation Strategy for Multi-Target Domain Adaptive Semantic Segmentation**

- 论文/Paper：https://arxiv.org/abs/2203.06811
- 代码/Code：

**Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs**

- 论文/Paper：https://arxiv.org/abs/2203.06717
- 代码/Code：https://github.com/megvii-research/RepLKNet

**LAS-AT: Adversarial Training with Learnable Attack Strategy**

- 论文/Paper：https://arxiv.org/abs/2203.06616
- 代码/Code：https://github.com/jiaxiaojunQAQ/LAS-AT

**Depth-Aware Generative Adversarial Network for Talking Head Video Generation**

- 论文/Paper：https://arxiv.org/abs/2203.06605
- 代码/Code：https://github.com/harlanhong/CVPR2022-DaGAN

**AutoGPart: Intermediate Supervision Search for Generalizable 3D Part Segmentation**

- 论文/Paper：https://arxiv.org/abs/2203.06558

- 代码/Code：

**Sparse Local Patch Transformer for Robust Face Alignment and Landmarks Inherent Relation Learning**

- 论文/Paper：https://arxiv.org/abs/2203.06541
- 代码/Code：https://github.com/Jiahao-UTS/SLPT-master

**Implicit Feature Decoupling with Depthwise Quantization**

- 论文/Paper：https://arxiv.org/abs/2203.08080
- 代码/Code：

**Interspace Pruning: Using Adaptive Filter Representations to Improve Training of Sparse CNNs**

- 论文/Paper：https://arxiv.org/abs/2203.07808
- 代码/Code：

**Learning What Not to Segment: A New Perspective on Few-Shot Segmentation**

- 论文/Paper：https://arxiv.org/abs/2203.07615
- 代码/Code：https://github.com/chunbolang/BAM

**Can Neural Nets Learn the Same Mode**

**l Twice? Investigating Reproducibility and Double Descent from the Decision Boundary Perspective**

- 论文/Paper：https://arxiv.org/abs/2203.08124
- 代码/Code：https://github.com/somepago/dbViz

**Scalable Penalized Regression for Noise Detection in Learning with Noisy Labels**

- 论文/Paper：https://arxiv.org/abs/2203.07788
- 代码/Code：https://github.com/Yikai-Wang/SPR-LNL

**Deep vanishing point detection: Geometric priors make dataset variations vanish**

- 论文/Paper：https://arxiv.org/abs/2203.08586
- 代码/Code：https://github.com/yanconglin/VanishingPoint_HoughTransform_GaussianSphere

**Non-isotropy Regularization for Proxy-based Deep Metric Learning**

- 论文/Paper：https://arxiv.org/abs/2203.08563

- 代码/Code：https://github.com/ExplainableML/NonIsotropicProxyDML

**Integrating Language Guidance into Vision-based Deep Metric Learning**

- 论文/Paper：https://arxiv.org/abs/2203.08543
- 代码/Code：https://github.com/ExplainableML/LanguageGuidance_for_DML

**Pseudo-Q: Generating Pseudo Language Queries for Visual Grounding**

- 论文/Paper：https://arxiv.org/abs/2203.08481
- 代码/Code：https://github.com/LeapLabTHU/Pseudo-Q

**The Devil Is in the Details: Window-based Attention for Image Compression**

- 论文/Paper：https://arxiv.org/abs/2203.08450
- 代码/Code：https://github.com/Googolxx/STF

**Represent, Compare, and Learn: A Similarity-Aware Framework for Class-Agnostic Counting**

- 论文/Paper：https://arxiv.org/abs/2203.08354
- 代码/Code：https://github.com/flyinglynx/Bilinear-Matching-Network

**Vox2Cortex: Fast Explicit Reconstruction of Cortical Surfaces from 3D MRI Scans with Geometric Deep Neural Networks**

- 论文/Paper：https://arxiv.org/abs/2203.09446
- 代码/Code：

**Bi-directional Object-context Prioritization Learning for Saliency Ranking**

- 论文/Paper：https://arxiv.org/abs/2203.09416
- 代码/Code：https://github.com/GrassBro/OCOR

**Object Localization under Single Coarse Point Supervision**

- 论文/Paper：https://arxiv.org/abs/2203.09338
- 代码/Code：https://github.com/ucas-vg/PointTinyBenchmark/

**Neural Compression-Based Feature Learning for Video Restoration**

- 论文/Paper：https://arxiv.org/abs/2203.09208
- 代码/Code：

**MuKEA: Multimodal Knowledge Extraction and Accumulation for Knowledge-based Visual Question Answering**

- 论文/Paper：https://arxiv.org/abs/2203.09138
- 代码/Code：https://github.com/AndersonStra/MuKEA

**Improving the Transferability of Targeted Adversarial Examples through Object-Based Diverse Input**

- 论文/Paper：https://arxiv.org/abs/2203.09123
- 代码/Code：https://github.com/dreamflake/ODI

**DATA: Domain-Aware and Task-Aware Pre-training**

- 论文/Paper：https://arxiv.org/abs/2203.09041
- 代码/Code：https://github.com/GAIA-vision/GAIA-ssl

**Fine-tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning**

- 论文/Paper：https://arxiv.org/abs/2203.09249

- 代码/Code：

**Global Convergence of MAML and Theory-Inspired Neural Architecture Search for Few-Shot Learning**

- 论文/Paper：https://arxiv.org/abs/2203.09137
- 代码/Code：https://github.com/YiteWang/MetaNTK-NAS

**Learning Affordance Grounding from Exocentric Images**

- 论文/Paper: http://arxiv.org/pdf/2203.09905
- 代码/Code: https://github.com/lhc1224/cross-view-affordance-grounding

**DTA: Physical Camouflage Attacks using Differentiable Transformation Network**

- 论文/Paper: http://arxiv.org/pdf/2203.09831
- 代码/Code: None

**Cross-Modal Perceptionist: Can Face Geometry be Gleaned from Voices?**

- 论文/Paper: http://arxiv.org/pdf/2203.09824
- 代码/Code: None

**Revisiting Domain Generalized Stereo Matching Networks from a Feature Consistency Perspective**

- 论文/Paper: http://arxiv.org/abs/2203.10887
- 代码/Code: None

**ViM: Out-Of-Distribution with Virtual-logit Matching**

- 论文/Paper: http://arxiv.org/abs/2203.10807
- 代码/Code: None

**Delving into the Estimation Shift of Batch Normalization in a Network**

- 论文/Paper: http://arxiv.org/abs/2203.10778
- 代码/Code: None

**Depth Estimation by Combining Binocular Stereo and Monocular Structured-Light**

- 论文/Paper: http://arxiv.org/abs/2203.10493
- 代码/Code: None

**TVConv: Efficient Translation Variant Convolution for Layout-aware Visual Processing**

- 论文/Paper: http://arxiv.org/abs/2203.10489
- 代码/Code: None

**Portrait Eyeglasses and Shadow Removal by Leveraging 3D Synthetic Data**

- 论文/Paper: http://arxiv.org/abs/2203.10474
- 代码/Code: None

**Discovering Objects that Can Move**

- 论文/Paper: http://arxiv.org/abs/2203.10159
- 代码/Code: None

**φ-SfT: Shape-from-Template with a Physics-Based Deformation Model**

- 论文/Paper: http://arxiv.org/abs/2203.11938
- 代码/Code: None

**Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation**

- 论文/Paper: http://arxiv.org/abs/2203.11483
- 代码/Code: None

**Mixed Differential Privacy in Computer Vision**

- 论文/Paper: http://arxiv.org/abs/2203.11481
- 代码/Code: None

**Global Matching with Overlapping Attention for Optical Flow Estimation**

- 论文/Paper: http://arxiv.org/abs/2203.11335
- 代码/Code: None

**DR.VIC: Decomposition and Reasoning for Video Individual Counting**

- 论文/Paper: http://arxiv.org/abs/2203.12335
- 代码/Code: https://github.com/taohan10200/drnet

**DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning for Histopathology Whole Slide Image Classification**

- 论文/Paper: http://arxiv.org/abs/2203.12081
- 代码/Code:https://github.com/hrzhang1123/DTFD-MIL

**Efficient Translation Variant Convolution for Layout-aware Visual Processing**

- 论文/Paper：http://arxiv.org/abs/2203.10489
- 代码/Code：https://github.com/JierunChen/TVConv

**Moving Window Regression: A Novel Approach to Ordinal Regression**

- 论文/Paper: http://arxiv.org/abs/2203.13122
- 代码/Code: None

**Egocentric Prediction of Action Target in 3D**

- 论文/Paper: http://arxiv.org/abs/2203.13116
- 代码/Code: None

**Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction**

- 论文/Paper: http://arxiv.org/abs/2203.12997
- 代码/Code: None

**Neural Reflectance for Shape Recovery with Shadow Handling**

- 论文/Paper: http://arxiv.org/abs/2203.12909
- 代码/Code: None

**DyRep: Bootstrapping Training with Dynamic Re-parameterization**

- 论文/Paper: http://arxiv.org/abs/2203.12868
- 代码/Code: None

**Multidimensional Belief Quantification for Label-Efficient Meta-Learning**

- 论文/Paper: http://arxiv.org/abs/2203.12768
- 代码/Code: None

**Give Me Your Attention: Dot-Product Attention Considered Harmful for Adversarial Patch Robustness**

- 论文/Paper: http://arxiv.org/pdf/2203.13639
- 代码/Code: None

**Unsupervised Pre-training for Temporal Action Localization Tasks**

- 论文/Paper: http://arxiv.org/pdf/2203.13609
- 代码/Code: None

**Continual Test-Time Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2203.13591
- 代码/Code: None

**Self-Supervised Predictive Learning: A Negative-Free Method for Sound Source Localization in Visual Scenes**

- 论文/Paper: http://arxiv.org/pdf/2203.13412
- 代码/Code: None

**NPBG++: Accelerating Neural Point-Based Graphics**

- 论文/Paper: http://arxiv.org/pdf/2203.13318
- 代码/Code: None

**Weakly-Supervised Online Action Segmentation in Multi-View Instructional Videos**

- 论文/Paper: http://arxiv.org/pdf/2203.13309
- 代码/Code: None

**Probing Representation Forgetting in Supervised and Unsupervised Continual Learning**

- 论文/Paper: http://arxiv.org/pdf/2203.13381
- 代码/Code: None

**Energy-based Latent Aligner for Incremental Learning**

- 论文/Paper: http://arxiv.org/abs/2203.14952
- 代码/Code: None

**Controllable Dynamic Multi-Task Architectures**

- 论文/Paper: http://arxiv.org/abs/2203.14949
- 代码/Code: None

**Attributable Visual Similarity Learning**

- 论文/Paper: http://arxiv.org/abs/2203.14932
- 代码/Code: None

**Learning Where to Learn in Cross-View Self-Supervised Learning**

- 论文/Paper: http://arxiv.org/abs/2203.14898
- 代码/Code: None

**Doodle It Yourself: Class Incremental Learning by Drawing a Few Sketches**

- 论文/Paper: http://arxiv.org/abs/2203.14843
- 代码/Code: None

**Partially Does It: Towards Scene-Level FG-SBIR with Partial Input**

- 论文/Paper: http://arxiv.org/abs/2203.14804
- 代码/Code: None

**Bi-level Doubly Variational Learning for Energy-based Latent Variable Models**

- 论文/Paper: http://arxiv.org/abs/2203.14702
- 代码/Code: None

**Sketch3T: Test-Time Training for Zero-Shot SBIR**

- 论文/Paper: http://arxiv.org/abs/2203.14691
- 代码/Code: None

**Brain-inspired Multilayer Perceptron with Spiking Neurons**

- 论文/Paper: http://arxiv.org/abs/2203.14679
- 代码/Code: None

**Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection**

- 论文/Paper: http://arxiv.org/abs/2203.14506
- 代码/Code: None

**NOC-REK: Novel Object Captioning with Retrieved Vocabulary from External Knowledge**

- 论文/Paper: http://arxiv.org/abs/2203.14499
- 代码/Code: None

**ARCS: Accurate Rotation and Correspondence Search**

- 论文/Paper: http://arxiv.org/abs/2203.14493
- 代码/Code: None

**iPLAN: Interactive and Procedural Layout Planning**

- 论文/Paper: http://arxiv.org/abs/2203.14412
- 代码/Code: None

**Locality-Aware Inter-and Intra-Video Reconstruction for Self-Supervised Correspondence Learning**

- 论文/Paper: http://arxiv.org/abs/2203.14333
- 代码/Code: None

**Local-Adaptive Face Recognition via Graph-based Meta-Clustering and Regularized Adaptation**

- 论文/Paper: http://arxiv.org/abs/2203.14327
- 代码/Code: None

**Unsupervised Vision-Language Parsing: Seamlessly Bridging Visual Scene Graphs with Language Structures via Dependency Relationships**

- 论文/Paper: http://arxiv.org/abs/2203.14260
- 代码/Code: None

**Knowledge Mining with Scene Text for Fine-Grained Recognition**

- 论文/Paper: http://arxiv.org/abs/2203.14215
- 代码/Code: None

**Long-Tailed Recognition via Weight Balancing**

- 论文/Paper: http://arxiv.org/abs/2203.14197
- 代码/Code: None

**HINT: Hierarchical Neuron Concept Explainer**

- 论文/Paper: http://arxiv.org/abs/2203.14196
- 代码/Code: None

**Bridge-Prompt: Towards Ordinal Action Understanding in Instructional Videos**

- 论文/Paper: http://arxiv.org/abs/2203.14104
- 代码/Code: None

**Learning to Answer Questions in Dynamic Audio-Visual Scenarios**

- 论文/Paper: http://arxiv.org/abs/2203.14072
- 代码/Code: None

**Neural MoCon: Neural Motion Control for Physically Plausible Human Motion Capture**

- 论文/Paper: http://arxiv.org/abs/2203.14065
- 代码/Code: None

**Visual Abductive Reasoning**

- 论文/Paper: http://arxiv.org/abs/2203.14040
- 代码/Code: None

**RSCFed: Random Sampling Consensus Federated Semi-supervised Learning**

- 论文/Paper: http://arxiv.org/abs/2203.13993
- 代码/Code: None

**GEN-VLKT: Simplify Association and Enhance Interaction Understanding for HOI Detection**

- 论文/Paper: http://arxiv.org/abs/2203.13954
- 代码/Code: None

**Sylph: A Hypernetwork Framework for Incremental Few-shot Object Detection**

- 论文/Paper: http://arxiv.org/abs/2203.13903
- 代码/Code: None

**Causality Inspired Representation Learning for Domain Generalization**

- 论文/Paper: http://arxiv.org/abs/2203.14237
- 代码/Code: None

**Transformer-empowered Multi-scale Contextual Matching and Aggregation for Multi-contrast MRI Super-resolution**

- 论文/Paper: http://arxiv.org/abs/2203.13963
- 代码/Code: None

**CHEX: CHannel EXploration for CNN Model Compression**

- 论文/Paper: http://arxiv.org/pdf/2203.15794
- 代码/Code: None

**FisherMatch: Semi-Supervised Rotation Regression via Entropy-based Filtering**

- 论文/Paper: http://arxiv.org/pdf/2203.15765
- 代码/Code: None

**EnvEdit: Environment Editing for Vision-and-Language Navigation**

- 论文/Paper: http://arxiv.org/pdf/2203.15685
- 代码/Code: None

**Exploring Frequency Adversarial Attacks for Face Forgery Detection**

- 论文/Paper: http://arxiv.org/pdf/2203.15674
- 代码/Code: None

**BARC: Learning to Regress 3D Dog Shape from Images by Exploiting Breed Information**

- 论文/Paper: http://arxiv.org/pdf/2203.15536
- 代码/Code: None

**Learning Structured Gaussians to Approximate Deep Ensembles**

- 论文/Paper: http://arxiv.org/pdf/2203.15485
- 代码/Code: None

**Quantifying Societal Bias Amplification in Image Captioning**

- 论文/Paper: http://arxiv.org/pdf/2203.15395
- 代码/Code: None

**Alignment-Uniformity aware Representation Learning for Zero-shot Video Classification**

- 论文/Paper: http://arxiv.org/pdf/2203.15381
- 代码/Code: https://github.com/ShipuLoveMili/CVPR2022-AURL

**Self-Supervised Image Representation Learning with Geometric Set Consistency**

- 论文/Paper: http://arxiv.org/pdf/2203.15361
- 代码/Code: None

**Nested Collaborative Learning for Long-Tailed Visual Recognition**

- 论文/Paper: http://arxiv.org/pdf/2203.15359
- 代码/Code: None

**Online Continual Learning on a Contaminated Data Stream with Blurry Task Boundaries**

- 论文/Paper: http://arxiv.org/pdf/2203.15355
- 代码/Code: None

**CNN Filter DB: An Empirical Investigation of Trained Convolutional Filters**

- 论文/Paper: http://arxiv.org/pdf/2203.15331
- 代码/Code: None

**Dressing in the Wild by Watching Dance Videos**

- 论文/Paper: http://arxiv.org/pdf/2203.15320
- 代码/Code: None

**Eigencontours: Novel Contour Descriptors Based on Low-Rank Approximation**

- 论文/Paper: http://arxiv.org/pdf/2203.15259
- 代码/Code: None

**Pop-Out Motion: 3D-Aware Image Deformation via Learning the Shape Laplacian**

- 论文/Paper: http://arxiv.org/pdf/2203.15235
- 代码/Code: None

**Zero-Query Transfer Attacks on Context-Aware Object Detectors**

- 论文/Paper: http://arxiv.org/pdf/2203.15230
- 代码/Code: None

**ASM-Loc: Action-aware Segment Modeling for Weakly-Supervised Temporal Action Localization**

- 论文/Paper: http://arxiv.org/pdf/2203.15187
- 代码/Code: None

**Registering Explicit to Implicit: Towards High-Fidelity Garment mesh Reconstruction from Single Images**

- 论文/Paper: http://arxiv.org/pdf/2203.15007
- 代码/Code: None

**Clean Implicit 3D Structure from Noisy 2D STEM Images**

- 论文/Paper: http://arxiv.org/pdf/2203.15434
- 代码/Code: None

**Equivariance Allows Handling Multiple Nuisance Variables When Analyzing Pooled Neuroimaging Datasets**

- 论文/Paper: http://arxiv.org/pdf/2203.15234
- 代码/Code: None

**Large-Scale Pre-training for Person Re-identification with Noisy Labels**

- 论文/Paper: http://arxiv.org/pdf/2203.16533
- 代码/Code: https://github.com/dengpanfu/luperson-nl

**Understanding 3D Object Articulation in Internet Videos**

- 论文/Paper: http://arxiv.org/pdf/2203.16531
- 代码/Code: None

**CaDeX: Learning Canonical Deformation Coordinate Space for Dynamic Surface Representation via Neural Homeomorphism**

- 论文/Paper: http://arxiv.org/pdf/2203.16529
- 代码/Code: None

**Unseen Classes at a Later Time? No Problem**

- 论文/Paper: http://arxiv.org/pdf/2203.16517
- 代码/Code: https://github.com/sumitramalagi/unseen-classes-at-a-later-time

**Fast Light-Weight Near-Field Photometric Stereo**

- 论文/Paper: http://arxiv.org/pdf/2203.16515
- 代码/Code: None

**AdaMixer: A Fast-Converging Query-Based Object Detector**

- 论文/Paper: http://arxiv.org/pdf/2203.16507
- 代码/Code: https://github.com/mcg-nju/adamixer

**Fast, Accurate and Memory-Efficient Partial Permutation Synchronization**

- 论文/Paper: http://arxiv.org/pdf/2203.16505
- 代码/Code: None

**Balanced MSE for Imbalanced Visual Regression**

- 论文/Paper: http://arxiv.org/pdf/2203.16427
- 代码/Code: None

**Multi-Robot Active Mapping via Neural Bipartite Graph Matching**

- 论文/Paper: http://arxiv.org/pdf/2203.16319
- 代码/Code: None

**Image-to-Lidar Self-Supervised Distillation for Autonomous Driving Data**

- 论文/Paper: http://arxiv.org/pdf/2203.16258
- 代码/Code: https://github.com/valeoai/slidr

**FLOAT: Factorized Learning of Object Attributes for Improved Multi-object Multi-part Scene Parsing**

- 论文/Paper: http://arxiv.org/pdf/2203.16168
- 代码/Code: None

**STRPM: A Spatiotemporal Residual Predictive Model for High-Resolution Video Prediction**

- 论文/Paper: http://arxiv.org/pdf/2203.16084
- 代码/Code: None

**Learning Program Representations for Food Images and Cooking Recipes**

- 论文/Paper: http://arxiv.org/pdf/2203.16071
- 代码/Code: None

**AxIoU: An Axiomatically Justified Measure for Video Moment Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2203.16062
- 代码/Code: None

**Progressively Generating Better Initial Guesses Towards Next Stages for High-Quality Human Motion Prediction**

- 论文/Paper: http://arxiv.org/pdf/2203.16051
- 代码/Code: None

**Iterative Deep Homography Estimation**

- 论文/Paper: http://arxiv.org/pdf/2203.15982
- 代码/Code: https://github.com/imdumpl78/ihn

**PSMNet: Position-aware Stereo Merging Network for Room Layout Estimation**

- 论文/Paper: http://arxiv.org/pdf/2203.15965
- 代码/Code: None

**Disentangled3D: Learning a 3D Generative Model with Disentangled Geometry and Appearance from Monocular Images**

- 论文/Paper: http://arxiv.org/pdf/2203.15926
- 代码/Code: None

**Learning to Detect Mobile Objects from LiDAR Scans Without Labels**

- 论文/Paper: http://arxiv.org/pdf/2203.15882
- 代码/Code: https://github.com/yurongyou/modest

**Proactive Image Manipulation Detection**

- 论文/Paper: http://arxiv.org/pdf/2203.15880
- 代码/Code: https://github.com/vishal3477/proactive_imd

**NICGSlowDown: Evaluating the Efficiency Robustness of Neural Image Caption Generation Models**

- 论文/Paper: http://arxiv.org/pdf/2203.15859
- 代码/Code: https://github.com/seekingdream/nicgslowdown

**Practical Learned Lossless JPEG Recompression with Multi-Level Cross-Channel Entropy Model in the DCT Domain**

- 论文/Paper: http://arxiv.org/pdf/2203.16357
- 代码/Code: None

**Bringing Old Films Back to Life**

- 论文/Paper: http://arxiv.org/pdf/2203.17276
- 代码/Code: https://github.com/raywzy/Bringing-Old-Films-Back-to-Life

**Generating High Fidelity Data from Low-density Regions using Diffusion Models**

- 论文/Paper: http://arxiv.org/pdf/2203.17260
- 代码/Code: None

**Continuous Scene Representations for Embodied AI**

- 论文/Paper: http://arxiv.org/pdf/2203.17251
- 代码/Code: None

**SimVQA: Exploring Simulated Environments for Visual Question Answering**

- 论文/Paper: http://arxiv.org/pdf/2203.17219
- 代码/Code: None

**Leverage Your Local and Global Representations: A New Self-Supervised Learning Strategy**

- 论文/Paper: http://arxiv.org/pdf/2203.17205
- 代码/Code: None

**AEGNN: Asynchronous Event-based Graph Neural Networks**

- 论文/Paper: http://arxiv.org/pdf/2203.17149
- 代码/Code: None

**It's All In the Teacher: Zero-Shot Quantization Brought Closer to the Teacher**

- 论文/Paper: http://arxiv.org/pdf/2203.17008
- 代码/Code: None

**Towards Robust Rain Removal Against Adversarial Attacks: A Comprehensive Benchmark Analysis and Beyond**

- 论文/Paper: http://arxiv.org/pdf/2203.16931
- 代码/Code: None

**End-to-End Trajectory Distribution Prediction Based on Occupancy Grid Maps**

- 论文/Paper: http://arxiv.org/pdf/2203.16910
- 代码/Code: None

**Reflection and Rotation Symmetry Detection via Equivariant Learning**

- 论文/Paper: http://arxiv.org/pdf/2203.16787
- 代码/Code: None

**Stochastic Backpropagation: A Memory Efficient Strategy for Training Video Models**

- 论文/Paper: http://arxiv.org/pdf/2203.16755
- 代码/Code: None

**Personalized Image Aesthetics Assessment with Rich Attributes**

- 论文/Paper: http://arxiv.org/pdf/2203.16754
- 代码/Code: None

**Constrained Few-shot Class-incremental Learning**

- 论文/Paper: http://arxiv.org/pdf/2203.16588
- 代码/Code: None

**Counterfactual Cycle-Consistent Learning for Instruction Following and Generation in Vision-Language Navigation**

- 论文/Paper: http://arxiv.org/pdf/2203.16586
- 代码/Code: None

**Exploiting Explainable Metrics for Augmented SGD**

- 论文/Paper: http://arxiv.org/pdf/2203.16723
- 代码/Code: None

**Task Adaptive Parameter Sharing for Multi-Task Learning**

- 论文/Paper: http://arxiv.org/pdf/2203.16708
- 代码/Code: None

**D-Grasp: Physically Plausible Dynamic Grasp Synthesis for Hand-Object Interactions**

- 论文/Paper: http://arxiv.org/pdf/2112.03028
- 代码/Code: None

**On the Importance of Asymmetry for Siamese Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2204.00613

- 代码/Code: https://github.com/facebookresearch/asym-siam

**DIP: Deep Inverse Patchmatch for High-Resolution Optical Flow**

- 论文/Paper: http://arxiv.org/pdf/2204.00330

- 代码/Code: https://github.com/zihuazheng/dip

**Unimodal-Concentrated Loss: Fully Adaptive Label Distribution Learning for Ordinal Regression**

- 论文/Paper: http://arxiv.org/pdf/2204.00309

- 代码/Code: None

**Perception Prioritized Training of Diffusion Models**

- 论文/Paper: http://arxiv.org/pdf/2204.00227

- 代码/Code: https://github.com/jychoi118/p2-weighting

**Bridging the Gap between Classification and Localization for Weakly Supervised Object Localization**

- 论文/Paper: http://arxiv.org/pdf/2204.00220

- 代码/Code: None

**GraftNet: Towards Domain Generalized Stereo Matching with a Broad-Spectrum and Task-Oriented Feature**

- 论文/Paper: http://arxiv.org/pdf/2204.00179

- 代码/Code: https://github.com/spadeliu/graft-psmnet

**LASER: LAtent SpacE Rendering for 2D Visual Localization**

论文/Paper: http://arxiv.org/pdf/2204.00157

代码/Code: None

**TransGeo: Transformer Is All You Need for Cross-view Image Geo-localization**

- 论文/Paper: http://arxiv.org/pdf/2204.00097

- 代码/Code: https://github.com/jeff-zilence/transgeo2022

**Investigating Top-$k$ White-Box and Transferable Black-box Attack**

- 论文/Paper: http://arxiv.org/pdf/2204.00089

- 代码/Code: None

**Efficient Maximal Coding Rate Reduction by Variational Forms**

- 论文/Paper: http://arxiv.org/pdf/2204.00077
- 代码/Code: None 

**Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos**

- 论文/Paper: http://arxiv.org/pdf/2204.01696

- 代码/Code: None

**LISA: Learning Implicit Shape and Appearance of Hands**

- 论文/Paper: http://arxiv.org/pdf/2204.01695

- 代码/Code: None

**Exemplar-bsaed Pattern Synthesis with Implicit Periodic Field Network**

- 论文/Paper: http://arxiv.org/pdf/2204.01671

- 代码/Code: None

**Degradation-agnostic Correspondence from Resolution-asymmetric Stereo**

- 论文/Paper: http://arxiv.org/pdf/2204.01429

- 代码/Code: None

**RayMVSNet: Learning Ray-based 1D Implicit Fields for Accurate Multi-View Stereo**

- 论文/Paper: http://arxiv.org/pdf/2204.01320

- 代码/Code: None

**Exploiting Temporal Relations on Radar Perception for Autonomous Driving**

- 论文/Paper: http://arxiv.org/pdf/2204.01184

- 代码/Code: None

**BNV-Fusion: Dense 3D Reconstruction using Bi-level Neural Volume Fusion**

- 论文/Paper: http://arxiv.org/pdf/2204.01139

- 代码/Code: None

**Neural Global Shutter: Learn to Restore Video from a Rolling Shutter Camera with Global Reset Feature**

- 论文/Paper: http://arxiv.org/pdf/2204.00974

- 代码/Code: https://github.com/lightchaserx/neural-global-shutter

**DST: Dynamic Substitute Training for Data-free Black-box Attack**

- 论文/Paper: http://arxiv.org/pdf/2204.00972

- 代码/Code: None

**Progressive Minimal Path Method with Embedded CNN**

- 论文/Paper: http://arxiv.org/pdf/2204.00944

- 代码/Code: None

**Online Convolutional Re-parameterization**

- 论文/Paper: http://arxiv.org/pdf/2204.00826

- 代码/Code: None

**SIMBAR: Single Image-Based Scene Relighting For Effective Data Augmentation For Automated Driving Vision Tasks**

- 论文/Paper: http://arxiv.org/pdf/2204.00644
- 代码/Code: None 

**Rethinking Visual Geo-localization for Large-Scale Applications**

- 论文/Paper: http://arxiv.org/pdf/2204.02287
- 代码/Code: None

**IRON: Inverse Rendering by Optimizing Neural SDFs and Materials from Photometric Images**

- 论文/Paper: http://arxiv.org/pdf/2204.02232
- 代码/Code: None

**SNUG: Self-Supervised Neural Dynamic Garments**

- 论文/Paper: http://arxiv.org/pdf/2204.02219
- 代码/Code: None

**Leveraging Equivariant Features for Absolute Pose Regression**

- 论文/Paper: http://arxiv.org/pdf/2204.02163
- 代码/Code: None

**MonoTrack: Shuttle trajectory reconstruction from monocular badminton video**

- 论文/Paper: http://arxiv.org/pdf/2204.01899
- 代码/Code: None

**Revisiting Near/Remote Sensing with Geospatial Attention**

- 论文/Paper: http://arxiv.org/pdf/2204.01807
- 代码/Code: None

**Temporal Alignment Networks for Long-term Video**

- 论文/Paper: http://arxiv.org/pdf/2204.02968
- 代码/Code: None

**"The Pedestrian next to the Lamppost" Adaptive Object Graphs for Better Instantaneous Mapping**

- 论文/Paper: http://arxiv.org/pdf/2204.02944
- 代码/Code: None

**Masking Adversarial Damage: Finding Adversarial Saliency for Robust and Sparse Network**

- 论文/Paper: http://arxiv.org/pdf/2204.02738
- 代码/Code: None

**Aesthetic Text Logo Synthesis via Content-aware Layout Inferring**

- 论文/Paper: http://arxiv.org/pdf/2204.02701
- 代码/Code: https://github.com/yizhiwang96/TextLogoLayout

**Learning to Anticipate Future with Dynamic Context Removal**

- 论文/Paper: http://arxiv.org/pdf/2204.02587
- 代码/Code: https://github.com/AllenXuuu/DCR.

**SqueezeNeRF: Further factorized FastNeRF for memory-efficient inference**

- 论文/Paper: http://arxiv.org/pdf/2204.02585
- 代码/Code: None

**Gait Recognition in the Wild with Dense 3D Representations and A Benchmark**

- 论文/Paper: http://arxiv.org/pdf/2204.02569
- 代码/Code: None

**MixFormer: Mixing Features across Windows and Dimensions**

- 论文/Paper: http://arxiv.org/pdf/2204.02557
- 代码/Code: https://github.com/PaddlePaddle/PaddleClas

**RODD: A Self-Supervised Approach for Robust Out-of-Distribution Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.02553
- 代码/Code: None

**Adversarial Robustness through the Lens of Convolutional Filters**

- 论文/Paper: http://arxiv.org/pdf/2204.02481
- 代码/Code: website:
  https://github.com/paulgavrikov/cvpr22w_RobustnessThroughTheLens

**Learning Optimal K-space Acquisition and Reconstruction using Physics-Informed Neural Networks**

- 论文/Paper: http://arxiv.org/pdf/2204.02480
- 代码/Code: None

**Total Variation Optimization Layers for Computer Vision**

- 论文/Paper: http://arxiv.org/pdf/2204.03643
- 代码/Code: https://github.com/raymondyeh07/tv_layers_for_cv

**Pre-train, Self-train, Distill: A simple recipe for Supersizing 3D Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2204.03642
- 代码/Code: None

**Class-Incremental Learning with Strong Pre-trained Models**

- 论文/Paper: http://arxiv.org/pdf/2204.03634
- 代码/Code: None

**AutoRF: Learning 3D Object Radiance Fields from Single View Observations**

- 论文/Paper: http://arxiv.org/pdf/2204.03593
- 代码/Code: None

**Deep Visual Geo-localization Benchmark**

- 论文/Paper: http://arxiv.org/pdf/2204.03444
- 代码/Code: None

**Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality**

- 论文/Paper: http://arxiv.org/pdf/2204.03162
- 代码/Code: None

**UIGR: Unified Interactive Garment Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2204.03111
- 代码/Code: https://github.com/brandonhanx/compfashion

**AUV-Net: Learning Aligned UV Maps for Texture Transfer and Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2204.03105
- 代码/Code: None

**Hierarchical Self-supervised Representation Learning for Movie Understanding**

- 论文/Paper: http://arxiv.org/pdf/2204.03101
- 代码/Code: None

**Learning from Untrimmed Videos: Self-Supervised Video Representation Learning with Hierarchical Consistency**

- 论文/Paper: http://arxiv.org/pdf/2204.03017
- 代码/Code: None

**Multi-Scale Memory-Based Video Deblurring**

- 论文/Paper: http://arxiv.org/pdf/2204.02977
- 代码/Code: https://github.com/jibo27/memdeblur

**Gravitationally Lensed Black Hole Emission Tomography**

- 论文/Paper: http://arxiv.org/abs/2204.03715
- 代码/Code: None

**General Incremental Learning with Domain-aware Categorical Representations**

- 论文/Paper: http://arxiv.org/abs/2204.04078
- 代码/Code: None

**Identifying Ambiguous Similarity Conditions via Semantic Matching**

- 论文/Paper: http://arxiv.org/abs/2204.04053
- 代码/Code: None

**Does Robustness on ImageNet Transfer to Downstream Tasks?**

- 论文/Paper: http://arxiv.org/abs/2204.03934
- 代码/Code: None

**Deep Hyperspectral-Depth Reconstruction Using Single Color-Dot Projection**

- 论文/Paper: http://arxiv.org/abs/2204.03929
- 代码/Code: None

**CD$^2$-pFed: Cyclic Distillation-guided Channel Decoupling for Model Personalization in Federated Learning**

- 论文/Paper: http://arxiv.org/abs/2204.03880
- 代码/Code: None

**Reusing the Task-specific Classifier as a Discriminator: Discriminator-free Adversarial Domain Adaptation**

- 论文/Paper: http://arxiv.org/abs/2204.03838
- 代码/Code: https://github.com/xiaoachen98/DALN

**TorMentor: Deterministic dynamic-path, data augmentations with fractals**

- 论文/Paper: http://arxiv.org/abs/2204.03776
- 代码/Code: None

**TemporalUV: Capturing Loose Clothing with Temporally Coherent UV Coordinates**

- 论文/Paper: http://arxiv.org/abs/2204.03671
- 代码/Code: None

**Single-Photon Structured Light**

- 论文/Paper: http://arxiv.org/pdf/2204.05300
- 代码/Code: None

**Pyramid Grafting Network for One-Stage High Resolution Saliency Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.05041
- 代码/Code: None

**Structure-Aware Motion Transfer with Deformable Anchor Model**

- 论文/Paper: http://arxiv.org/pdf/2204.05018
- 代码/Code: None

**Reasoning with Multi-Structure Commonsense Knowledge in Visual Dialog**

- 论文/Paper: http://arxiv.org/pdf/2204.04680
- 代码/Code: None

**NAN: Noise-Aware NeRFs for Burst-Denoising**

- 论文/Paper: http://arxiv.org/pdf/2204.04668
- 代码/Code: None

**Learning Pixel-Level Distinctions for Video Highlight Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.04615
- 代码/Code: None

**Explaining Deep Convolutional Neural Networks via Latent Visual-Semantic Filter Attention**

- 论文/Paper: http://arxiv.org/pdf/2204.04601
- 代码/Code: None

**DeepLIIF: An Online Platform for Quantification of Clinical Pathology Slides**

- 论文/Paper: http://arxiv.org/pdf/2204.04494
- 代码/Code: None

**ManiTrans: Entity-Level Text-Guided Image Manipulation via Token-wise Semantic Alignment and Generation**

- 论文/Paper: http://arxiv.org/pdf/2204.04428
- 代码/Code: None

**FedCorr: Multi-Stage Federated Learning for Label Noise Correction**

- 论文/Paper: http://arxiv.org/pdf/2204.04677
- 代码/Code: https://github.com/Xu-Jingyi/FedCorr

**Adaptive Differential Filters for Fast and Communication-Efficient Federated Learning**

- 论文/Paper: http://arxiv.org/pdf/2204.04424
- 代码/Code: None

**The Two Dimensions of Worst-case Training and the Integrated Effect for Out-of-domain Generalization**

- 论文/Paper: http://arxiv.org/pdf/2204.04384
- 代码/Code: None

**Continual Predictive Learning from Videos**

- 论文/Paper: http://arxiv.org/pdf/2204.05624
- 代码/Code: https://github.com/jc043/CPL

**Few-shot Learning with Noisy Labels**

- 论文/Paper: http://arxiv.org/pdf/2204.05494
- 代码/Code: None

**Out-Of-Distribution Detection In Unsupervised Continual Learning**

- 论文/Paper: http://arxiv.org/pdf/2204.05462
- 代码/Code: None

**Generalizing Adversarial Explanations with Grad-CAM**

- 论文/Paper: http://arxiv.org/pdf/2204.05427
- 代码/Code: None

**Recognition of Freely Selected Keypoints on Human Limbs**

- 论文/Paper: http://arxiv.org/pdf/2204.06326
- 代码/Code: None

**3D-SPS: Single-Stage 3D Visual Grounding via Referred Point Progressive Selection**

- 论文/Paper: http://arxiv.org/pdf/2204.06272
- 代码/Code: None

**Defensive Patches for Robust Recognition in the Physical World**

- 论文/Paper: http://arxiv.org/pdf/2204.06213
- 代码/Code: https://github.com/nlsde-safety-team/DefensivePatch

**COAP: Compositional Articulated Occupancy of People**

- 论文/Paper: http://arxiv.org/pdf/2204.06184
- 代码/Code: None

**What's in your hands? 3D Reconstruction of Generic Objects in Hands**

- 论文/Paper: http://arxiv.org/pdf/2204.07153
- 代码/Code: None

**GIFS: Neural Implicit Function for General Shape Representation**

- 论文/Paper: http://arxiv.org/pdf/2204.07126
- 代码/Code: None

**The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark**

- 论文/Paper: http://arxiv.org/pdf/2204.06972
- 代码/Code: None

**Semi-Supervised Training to Improve Player and Ball Detection in Soccer**

- 论文/Paper: http://arxiv.org/pdf/2204.06859
- 代码/Code: https://github.com/rvandeghen/SST

**Pyramidal Attention for Saliency Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.06788
- 代码/Code: https://github.com/tanveer-hussain/EfficientSOD2

**OccAM's Laser: Occlusion-based Attribution Maps for 3D Object Detectors on LiDAR Data**

- 论文/Paper: http://arxiv.org/pdf/2204.06577
- 代码/Code: https://github.com/dschinagl/occam

**Patch-wise Contrastive Style Learning for Instagram Filter Removal**

- 论文/Paper: http://arxiv.org/pdf/2204.07486
- 代码/Code: None

**Guiding Attention using Partial-Order Relationships for Image Captioning**

- 论文/Paper: http://arxiv.org/pdf/2204.07476
- 代码/Code: None

**MetaSets: Meta-Learning on Point Sets for Generalizable Representations**

- 论文/Paper: http://arxiv.org/pdf/2204.07311
- 代码/Code: None

**Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference**

- 论文/Paper: http://arxiv.org/pdf/2204.07305
- 代码/Code: None

**Imposing Consistency for Optical Flow Estimation**

- 论文/Paper: http://arxiv.org/pdf/2204.07262
- 代码/Code: None

**Measuring Compositional Consistency for Video Question Answering**

- 论文/Paper: http://arxiv.org/pdf/2204.07190
- 代码/Code: None

**Deep Equilibrium Optical Flow Estimation**

- 论文/Paper: http://arxiv.org/pdf/2204.08442
- 代码/Code: None

**Unsupervised domain adaptation and super resolution on drone images for autonomous dry herbage biomass estimation**

- 论文/Paper: http://arxiv.org/pdf/2204.08271
- 代码/Code: None

**OMG: Observe Multiple Granularities for Natural Language-Based Vehicle Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2204.08209
- 代码/Code: https://github.com/dyhBUPT/OMG.

**Towards a Deeper Understanding of Skeleton-based Gait Recognition**

- 论文/Paper: http://arxiv.org/pdf/2204.07855
- 代码/Code: None

**Interactiveness Field in Human-Object Interactions**

- 论文/Paper: http://arxiv.org/pdf/2204.07718
- 代码/Code: https://github.com/Foruck/Interactiveness-Field.

**It is Okay to Not Be Okay: Overcoming Emotional Bias in Affective Image Captioning by Contrastive Data Collection**

- 论文/Paper: http://arxiv.org/pdf/2204.07660
- 代码/Code: None

**Deep Unlearning via Randomized Conditionally Independent Hessians**

- 论文/Paper: http://arxiv.org/pdf/2204.07655
- 代码/Code: https://github.com/vsingh-group/LCODEC-deep-unlearning

**Learning to Imagine: Diversify Memory for Incremental Learning using Unlabeled Data**

- 论文/Paper: http://arxiv.org/pdf/2204.08932
- 代码/Code: https://github.com/TOM-tym/Learn-to-Imagine

**An Efficient Domain-Incremental Learning Approach to Drive in All Weather Conditions**

- 论文/Paper: http://arxiv.org/pdf/2204.08817
- 代码/Code: None

**Incorporating Semi-Supervised and Positive-Unlabeled Learning for Boosting Full Reference Image Quality Assessment**

- 论文/Paper: http://arxiv.org/pdf/2204.08763
- 代码/Code: https://github.com/happycaoyue/JSPL


**Self-Supervised Equivariant Learning for Oriented Keypoint Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.08613
- 代码/Code: None

**GazeOnce: Real-Time Multi-Person Gaze Estimation**

- 论文/Paper: http://arxiv.org/pdf/2204.09480
- 代码/Code: None

**Epistemic Uncertainty-Weighted Loss for Visual Bias Mitigation**

- 论文/Paper: http://arxiv.org/pdf/2204.09389
- 代码/Code: None

**Reinforced Structured State-Evolution for Vision-Language Navigation**

- 论文/Paper: http://arxiv.org/pdf/2204.09280
- 代码/Code: None

**SpiderNet: Hybrid Differentiable-Evolutionary Architecture Search via Train-Free Metrics**

- 论文/Paper: http://arxiv.org/pdf/2204.09320
- 代码/Code: None

**A Deeper Look into Aleatoric and Epistemic Uncertainty Disentanglement**

- 论文/Paper: http://arxiv.org/pdf/2204.09308
- 代码/Code: None

**Does Interference Exist When Training a Once-For-All Network?**

- 论文/Paper: http://arxiv.org/pdf/2204.09210
- 代码/Code: https://github.com/Jordan-HS/RSS-Interference-CVPRW2022.

**Importance is in your attention: agent importance prediction for autonomous driving**

- 论文/Paper: http://arxiv.org/pdf/2204.09121
- 代码/Code: None

**SelfD: Self-Learning Large-Scale Driving Policies From the Web**

- 论文/Paper: http://arxiv.org/pdf/2204.10320
- 代码/Code: None

**SmartPortraits: Depth Powered Handheld Smartphone Dataset of Human Portraits for State Estimation, Reconstruction and Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2204.10211
- 代码/Code: None

**A case for using rotation invariant features in state of the art feature matchers**

- 论文/Paper: http://arxiv.org/pdf/2204.10144
- 代码/Code: None

**Toward Fast, Flexible, and Robust Low-Light Image Enhancement**

- 论文/Paper: http://arxiv.org/pdf/2204.10137
- 代码/Code: https://github.com/vis-opt-group/SCI

**OSSO: Obtaining Skeletal Shape from Outside**

- 论文/Paper: http://arxiv.org/pdf/2204.10129
- 代码/Code: None

**Is Neuron Coverage Needed to Make Person Detection More Robust?**

- 论文/Paper: http://arxiv.org/pdf/2204.10027
- 代码/Code: None

**Progressive Training of A Two-Stage Framework for Video Restoration**

- 论文/Paper: http://arxiv.org/pdf/2204.09924
- 代码/Code: None

**CNLL: A Semi-supervised Approach For Continual Noisy Label Learning**

- 论文/Paper: http://arxiv.org/pdf/2204.09881
- 代码/Code: None

**Persistent-Transient Duality in Human Behavior Modeling**

- 论文/Paper: http://arxiv.org/pdf/2204.09875
- 代码/Code: None

**Self-Supervised Learning to Guide Scientifically Relevant Categorization of Martian Terrain Images**

- 论文/Paper: http://arxiv.org/pdf/2204.09854
- 代码/Code: https://github.com/TejasPanambur/mastcam

**Exposure Correction Model to Enhance Image Quality**

- 论文/Paper: http://arxiv.org/pdf/2204.10648
- 代码/Code: https://github.com/yamand16/exposurecorrection

**Spacing Loss for Discovering Novel Categories**

- 论文/Paper: http://arxiv.org/pdf/2204.10595
- 代码/Code: https://github.com/josephkj/awesome-novel-class-discovery

**DiRA: Discriminative, Restorative, and Adversarial Learning for Self-supervised Medical Image Analysis**

- 论文/Paper: http://arxiv.org/pdf/2204.10437
- 代码/Code: https://github.com/jlianglab/dira

**The 6th AI City Challenge**

- 论文/Paper: http://arxiv.org/pdf/2204.10380
- 代码/Code: None

**Contrastive Test-Time Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2204.10377
- 代码/Code: None

**Proto2Proto: Can you recognize the car, the way I do?**

- 论文/Paper: http://arxiv.org/pdf/2204.11830
- 代码/Code: None

**Multi-Layer Modeling of Dense Vegetation from Aerial LiDAR Scans**

- 论文/Paper: http://arxiv.org/pdf/2204.11620
- 代码/Code: https://github.com/ekalinicheva/multi_layer_vegetation.

**Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG Images with Curriculum Learning**

- 论文/Paper: http://arxiv.org/pdf/2204.11433
- 代码/Code: None

**Can domain adaptation make object recognition work for everyone?**

- 论文/Paper: http://arxiv.org/pdf/2204.11122
- 代码/Code: None

**Investigating Neural Architectures by Synthetic Dataset Design**

- 论文/Paper: http://arxiv.org/pdf/2204.11045
- 代码/Code: None

**Revealing Occlusions with 4D Neural Fields**

- 论文/Paper: http://arxiv.org/pdf/2204.10916
- 代码/Code: None

**Identity Preserving Loss for Learned Image Compression**

- 论文/Paper: http://arxiv.org/pdf/2204.10869
- 代码/Code: None

**Towards Data-Free Model Stealing in a Hard Label Setting**

- 论文/Paper: http://arxiv.org/pdf/2204.11022
- 代码/Code: None

**Context-Aware Sequence Alignment using 4D Skeletal Augmentation**

- 论文/Paper: http://arxiv.org/abs/2204.12223
- 代码/Code: None

**Few-Shot Head Swapping in the Wild**

- 论文/Paper: http://arxiv.org/pdf/2204.13100
- 代码/Code: None

**Attention Consistency on Visual Corruptions for Single-Source Domain Generalization**

- 论文/Paper: http://arxiv.org/pdf/2204.13091
- 代码/Code: None

**Collaborative Learning for Hand and Object Reconstruction with Attention-guided Graph Convolution**

- 论文/Paper: http://arxiv.org/pdf/2204.13062
- 代码/Code: None

**A Scalable Combinatorial Solver for Elastic Geometrically Consistent 3D Shape Matching**

- 论文/Paper: http://arxiv.org/pdf/2204.12805
- 代码/Code: http://github.com/paul0noah/sm-comb

**Leveraging Unlabeled Data for Sketch-based Understanding**

- 论文/Paper: http://arxiv.org/pdf/2204.12522
- 代码/Code: None

**Towards assessing agricultural land suitability with causal machine learning**

- 论文/Paper: http://arxiv.org/pdf/2204.12956
- 代码/Code: None

**Conformer and Blind Noisy Students for Improved Image Quality Assessment**

- 论文/Paper: http://arxiv.org/pdf/2204.12819
- 代码/Code: None

**NeurMiPs: Neural Mixture of Planar Experts for View Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2204.13696
- 代码/Code: None

**Learning from Pixel-Level Noisy Label : A New Perspective for Light Field Saliency Detection**

- 论文/Paper: http://arxiv.org/pdf/2204.13456
- 代码/Code: https://github.com/OLobbCode/NoiseLF.

**A Challenging Benchmark of Anime Style Recognition**

- 论文/Paper: http://arxiv.org/pdf/2204.14034
- 代码/Code: https://github.com/nkjcqvcpi/asr

**AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-time Image Enhancement**

- 论文/Paper: http://arxiv.org/pdf/2204.13983
- 代码/Code: https://github.com/ImCharlesY/AdaInt.

**SCS-Co: Self-Consistent Style Contrastive Learning for Image Harmonization**

- 论文/Paper: http://arxiv.org/pdf/2204.13962
- 代码/Code: https://github.com/ychang686/scs-co

**Learning Adaptive Warping for Real-World Rolling Shutter Correction**

- 论文/Paper: http://arxiv.org/pdf/2204.13886
- 代码/Code: https://github.com/ljzycmd/bsrsc



**Stability-driven Contact Reconstruction From Monocular Color Images**

- 论文/Paper: http://arxiv.org/pdf/2205.00848
- 代码/Code: None

**GPUNet: Searching the Deployable Convolution Neural Networks for GPUs**

- 论文/Paper: http://arxiv.org/pdf/2205.00841
- 代码/Code: None

**MUTR3D: A Multi-camera Tracking Framework via 3D-to-2D Queries**

- 论文/Paper: http://arxiv.org/pdf/2205.00613
- 代码/Code: https://github.com/a1600012888/MUTR3D

**LayoutBERT: Masked Language Layout Model for Object Insertion**

- 论文/Paper: http://arxiv.org/pdf/2205.00347
- 代码/Code: None

**Improving Visual Grounding with Visual-Linguistic Verification and Iterative Reasoning**

- 论文/Paper: http://arxiv.org/pdf/2205.00272
- 代码/Code: https://github.com/yangli18/vltvg

**Look Closer to Supervise Better: One-Shot Font Generation via Component-Based Discriminator**

- 论文/Paper: http://arxiv.org/pdf/2205.00146
- 代码/Code: None

**Dual Cross-Attention Learning for Fine-Grained Visual Categorization and Object Re-Identification**

- 论文/Paper: http://arxiv.org/pdf/2205.02151
- 代码/Code: None

**Self-Taught Metric Learning without Labels**

- 论文/Paper: http://arxiv.org/pdf/2205.01903
- 代码/Code: None

**Fixing Malfunctional Objects With Learned Physical Simulation and Functional Prediction**

- 论文/Paper: http://arxiv.org/pdf/2205.02834
- 代码/Code: None

**Holistic Approach to Measure Sample-level Adversarial Vulnerability and its Utility in Building Trustworthy Systems**

- 论文/Paper: http://arxiv.org/pdf/2205.02604
- 代码/Code: None

**P3IV: Probabilistic Procedure Planning from Instructional Videos with Weak Supervision**

- 论文/Paper: http://arxiv.org/pdf/2205.02300
- 代码/Code: None

**Prompt Distribution Learning**

- 论文/Paper: http://arxiv.org/pdf/2205.03340
- 代码/Code: None

 [返回目录/back](#Contents)









参考：

[如何评价 CVPR2022 的论文接收结果？](https://www.zhihu.com/question/519162597/)

