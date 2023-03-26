# CVPR2023-Papers-with-Code-Demo



 :star_and_crescent:**CVPR2022论文下载：添加微信: nvshenj125, 备注 CVPR 2022 即可获取全部论文pdf**

 :star_and_crescent:**福利 注册即可领取 200 块计算资源 : https://www.bkunyun.com/wap/console?source=aistudy**
 [使用说明](https://mp.weixin.qq.com/s?__biz=MzU4NTY4Mzg1Mw==&amp;mid=2247521550&amp;idx=1&amp;sn=db4c7f609bd61ae7734b9e012a763f98&amp;chksm=fd8413eccaf39afa686f69f2df2463f4a6a8233ba3b3edf698513bbee556c9f6c21e835b8eb8&token=705359263&lang=zh_CN#rd)


欢迎关注公众号：AI算法与图像处理

:star2: [CVPR 2023](https://cvpr2023.thecvf.com/) 持续更新最新论文/paper和相应的开源代码/code！



B站demo：https://space.bilibili.com/288489574

> :hand: ​注：欢迎各位大佬提交issue，分享CVPR 2022论文/paper和开源项目！共同完善这个项目
>
> 往年顶会论文汇总：
>
> [CVPR2021](https://github.com/DWCTOD/CVPR2023-Papers-with-Code-Demo/blob/main/CVPR2021.md)
>
> [CVPR2022](https://github.com/DWCTOD/CVPR2023-Papers-with-Code-Demo/blob/main/CVPR2022.md)
>
> [ICCV2021](https://github.com/DWCTOD/ICCV2021-Papers-with-Code-Demo)
>
> [ECCV2022](https://github.com/DWCTOD/ECCV2022-Papers-with-Code-Demo)

### **:fireworks: 欢迎进群** | Welcome

CVPR 2023 论文/paper交流群已成立！已经收录的同学，可以添加微信：**nvshenj125**，请备注：**CVPR+姓名+学校/公司名称**！一定要根据格式申请，可以拉你进群。

<a name="Contents"></a>



### :hammer: **目录 |Table of Contents（点击直接跳转）**

<details open>
<summary> 目录（右侧点击可折叠）</summary>

- [Backbone](#Backbone)
- [数据集/Dataset](#Dataset)
- [Diffusion Model](#DiffusionModel)
- [NAS](#NAS)
- [NeRF](#NeRF)
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
- [人脸重建/Face Reconstruction](#FaceReconstruction)
- [Talking Face](#TalkingFace)
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
- [3D Reconstruction](#3DReconstruction)
- [视频插帧/Frame Interpolation](#FrameInterpolation)
- [视频超分 / Video Super-Resolution](#VideoSuper-Resolution)
- [3D点云/3D point cloud](#3DPointCloud)
- [标签噪声 / Label-Noise](#Label-Noise)
- [对抗样本/Adversarial Examples](#AdversarialExamples)
- [Anomaly Detection](#AnomalyDetection)
- [其他/Other](#Other)


</details>

<a name="Backbone"></a>

## Backbone

**Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks**

- 论文/Paper:https://arxiv.org/abs/2303.03667
- 代码/Code:https://github.com/JierunChen/FasterNet

[返回目录/back](#Contents)

<a name="Dataset"></a> 

## 数据集/Dataset

**Spring: A High-Resolution High-Detail Dataset and Benchmark for Scene Flow, Optical Flow and Stereo**

- 论文/Paper: http://arxiv.org/pdf/2303.01943
- 代码/Code: None

**Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes**

- 论文/Paper: http://arxiv.org/pdf/2303.02760
- 代码/Code: None



[返回目录/back](#Contents)

<a name="DiffusionModel"></a> 

# Diffusion Model

**Unifying Layout Generation with a Decoupled Diffusion Model**

- 论文/Paper: http://arxiv.org/pdf/2303.05049
- 代码/Code: None

**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration**

- 论文/Paper: http://arxiv.org/pdf/2303.06885
- 代码/Code: None

**LayoutDM: Discrete Diffusion Model for Controllable Layout Generation**

- 论文/Paper: http://arxiv.org/pdf/2303.08137
- 代码/Code: https://github.com/CyberAgentAILab/layout-dm

**Controllable Mesh Generation Through Sparse Latent Point Diffusion Models**

- 论文/Paper: http://arxiv.org/pdf/2303.07938
- 代码/Code: None

**Decomposed Diffusion Models for High-Quality Video Generation**

- 论文/Paper: http://arxiv.org/pdf/2303.08320
- 代码/Code: None

**Taming Diffusion Models for Audio-Driven Co-Speech Gesture Generation**

- 论文/Paper: http://arxiv.org/pdf/2303.09119
- 代码/Code: https://github.com/advocate99/diffgesture

**Leapfrog Diffusion Model for Stochastic Trajectory Prediction**

- 论文/Paper: http://arxiv.org/pdf/2303.10895
- 代码/Code: https://github.com/mediabrain-sjtu/led

[返回目录/back](#Contents)

<a name="NAS"></a> 

## NAS



[返回目录/back](#Contents)

<a name="NeRF"></a> 

# NeRF

**Nerflets: Local Radiance Fields for Efficient Structure-Aware 3D Scene Representation from 2D Supervisio**

- 论文/Paper: http://arxiv.org/pdf/2303.03361
- 代码/Code: None

**NeRFLiX: High-Quality Neural View Synthesis by Learning a Degradation-Driven Inter-viewpoint MiXer**

- 论文/Paper: http://arxiv.org/pdf/2303.06919
- 代码/Code: None

**PartNeRF: Generating Part-Aware Editable 3D Shapes without 3D Supervision**

- 论文/Paper: http://arxiv.org/pdf/2303.09554
- 代码/Code: None

**StyleRF: Zero-shot 3D Style Transfer of Neural Radiance Fields**

- 论文/Paper: http://arxiv.org/pdf/2303.10598
- 代码/Code: None

**SINE: Semantic-driven Image-based NeRF Editing with Prior-guided Editing Field**

- 论文/Paper: http://arxiv.org/pdf/2303.13277
- 代码/Code: None

[返回目录/back](#Contents)

<a name="KnowledgeDistillation"></a> 

## Knowledge Distillation

**Paper title: Generic-to-Specific Distillation of Masked Autoencoders**

- 论文/Paper: https://arxiv.org/abs/2302.14771
- 代码/Code: https://github.com/pengzhiliang/G2SD

**X$^3$KD: Knowledge Distillation Across Modalities, Tasks and Stages for Multi-Camera 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.02203
- 代码/Code: None

[返回目录/back](#Contents)

<a name="Multimodal"></a> 

## 多模态 / Multimodal

**PolyFormer: Referring Image Segmentation as Sequential Polygon Generation**

- 论文/Paper: https://arxiv.org/abs/2302.14771
- 代码/Code: None

**Multimodal Industrial Anomaly Detection via Hybrid Fusion**

- 论文/Paper: http://arxiv.org/pdf/2303.00601
- 代码/Code: https://github.com/nomewang/m3dm

**Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision**

- 论文/Paper: http://arxiv.org/pdf/2303.00462
- 代码/Code: https://github.com/toytiny/cmflow

**AMIGO: Sparse Multi-Modal Graph Transformer with Shared-Context Processing for Representation Learning of Giga-pixel Images**

- 论文/Paper: http://arxiv.org/pdf/2303.00865
- 代码/Code: None

**Multimodal Prompting with Missing Modalities for Visual Recognition**

- 论文/Paper: http://arxiv.org/pdf/2303.03369
- 代码/Code: https://github.com/yilunlee/missing_aware_prompts

**FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks**

- 论文/Paper: http://arxiv.org/pdf/2303.02483
- 代码/Code: None

**Virtual Sparse Convolution for Multimodal 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.02314
- 代码/Code: https://github.com/hailanyi/virconv

**LoGoNet: Towards Accurate 3D Object Detection with Local-to-Global Cross-Modal Fusion**

- 论文/Paper: http://arxiv.org/pdf/2303.03595
- 代码/Code: https://github.com/sankin97/LoGoNet

**Understanding and Constructing Latent Modality Structures in Multi-modal Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.05952
- 代码/Code: None

**Align and Attend: Multimodal Summarization with Dual Contrastive Losses**

- 论文/Paper: http://arxiv.org/pdf/2303.07284
- 代码/Code: None

**Multimodal Feature Extraction and Fusion for Emotional Reaction Intensity Estimation and Expression Classification in Videos with Transformers**

- 论文/Paper: http://arxiv.org/pdf/2303.09164
- 代码/Code: None

**Self-Supervised Learning for Multimodal Non-Rigid 3D Shape Matching**

- 论文/Paper: http://arxiv.org/pdf/2303.10971
- 代码/Code: https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching

**Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2303.12501

[返回目录/back](#Contents)

<a name="ContrastiveLearning"></a> 

## Contrastive Learning

**Twin Contrastive Learning with Noisy Labels**

- 论文/Paper: http://arxiv.org/pdf/2303.06930
- 代码/Code: https://github.com/Hzzone/TCL

**TranSG: Transformer-Based Skeleton Graph Prototype Contrastive Learning with Structure-Trajectory Prompted Reconstruction for Person Re-Identification**

- 论文/Paper: http://arxiv.org/pdf/2303.06819
- 代码/Code: https://github.com/kali-hac/transg

**MobileVOS: Real-Time Video Object Segmentation Contrastive Learning meets Knowledge Distillation**

- 论文/Paper: http://arxiv.org/pdf/2303.07815
- 代码/Code: None

**Learning Audio-Visual Source Localization via False Negative Aware Contrastive Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.11302
- 代码/Code: \url{https://github.com/weixuansun/FNAC-AVL}.

**Actionlet-Dependent Contrastive Learning for Unsupervised Skeleton-Based Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2303.10904
- 代码/Code: None

**Dynamic Graph Enhanced Contrastive Learning for Chest X-ray Report Generation**

- 论文/Paper: http://arxiv.org/pdf/2303.10323
- 代码/Code: https://github.com/mlii0117/dcl

**CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.12793
- 代码/Code: https://github.com/FangyunWei/SLRT

**MaskCon: Masked Contrastive Learning for Coarse-Labelled Dataset**

- 论文/Paper: http://arxiv.org/pdf/2303.12756
- 代码/Code: https://github.com/MrChenFeng/MaskCon_CVPR2023

[返回目录/back](#Contents)

<a name="CapsuleNetwork"></a> 

# 胶囊网络 / Capsule Network



[返回目录/back](#Contents)

<a name="ImageClassification"></a> 

# 图像分类 / Image Classification

**Fine-Grained Classification with Noisy Labels**

- 论文/Paper: http://arxiv.org/pdf/2303.02404
- 代码/Code: None

**Task-specific Fine-tuning via Variational Information Bottleneck for Weakly-supervised Pathology Whole Slide Image Classification**

- 论文/Paper: http://arxiv.org/pdf/2303.08446
- 代码/Code: https://github.com/invoker-LL/WSI-finetuning

**Boosting Verified Training for Robust Image Classifications via Abstraction**

- 论文/Paper: http://arxiv.org/pdf/2303.11552
- 代码/Code: https://github.com/zhangzhaodi233/abscert

**Curvature-Balanced Feature Manifold Learning for Long-Tailed Classification**

- 论文/Paper: http://arxiv.org/pdf/2303.12307
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ObjectDetection"></a> 

## 目标检测/Object Detection

**Towards Domain Generalization for Multi-view 3D Object Detection in Bird-Eye-View**

- 论文/Paper: http://arxiv.org/pdf/2303.01686
- 代码/Code: None

**Virtual Sparse Convolution for Multimodal 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.02314
- 代码/Code: https://github.com/hailanyi/virconv

**LoGoNet: Towards Accurate 3D Object Detection with Local-to-Global Cross-Modal Fusion**

- 论文/Paper: http://arxiv.org/pdf/2303.03595
- 代码/Code: https://github.com/sankin97/LoGoNet

**NIFF: Alleviating Forgetting in Generalized Few-Shot Object Detection via Neural Instance Feature Forging**

- 论文/Paper: http://arxiv.org/pdf/2303.04958
- 代码/Code: None

**Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.05892
- 代码/Code: https://github.com/LutingWang/OADP.

**Bi3D: Bi-domain Active Learning for Cross-domain 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.05886
- 代码/Code: https://github.com/PJLabADG/3DTrans

**Uni3D: A Unified Baseline for Multi-dataset 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.06880
- 代码/Code: https://github.com/PJLab-ADG/3DTrans

**Lite DETR : An Interleaved Multi-Scale Encoder for Efficient DETR**

- 论文/Paper: http://arxiv.org/pdf/2303.07335
- 代码/Code: https://github.com/IDEA-Research/Lite-DETR

**PiMAE: Point Cloud and Image Interactive Masked Autoencoders for 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.08129
- 代码/Code: https://github.com/blvlab/pimae

**Weakly Supervised Monocular 3D Object Detection using Multi-View Projection and Direction Consistency**

- 论文/Paper: http://arxiv.org/pdf/2303.08686
- 代码/Code: https://github.com/weakmono3d/weakmono3d

**Active Teacher for Semi-Supervised Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.08348
- 代码/Code: https://github.com/hunterj-lin/activeteacher

**MSF: Motion-guided Sequential Fusion for Efficient 3D Object Detection from Point Cloud Sequences**

- 论文/Paper: http://arxiv.org/pdf/2303.08316
- 代码/Code: https://github.com/skyhehe123/MSF

**MixTeacher: Mining Promising Labels with Mixed Scale Teacher for Semi-Supervised Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.09061
- 代码/Code: https://github.com/lliuz/MixTeacher

**DiGeo: Discriminative Geometry-Aware Learning for Generalized Few-Shot Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.09674
- 代码/Code: https://github.com/Phoenix-V/DiGeo

**VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking**

- 论文/Paper: http://arxiv.org/pdf/2303.11301
- 代码/Code: https://github.com/dvlab-research/VoxelNeXt

**Benchmarking Robustness of 3D Object Detection to Common Corruptions in Autonomous Driving**

- 论文/Paper: http://arxiv.org/pdf/2303.11040
- 代码/Code: https://github.com/kkkcx/3D_Corruptions_AD.

**CAPE: Camera View Position Embedding for Multi-View 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.10209
- 代码/Code: https://github.com/PaddlePaddle/Paddle3D

**STDLens: Model Hijacking-resilient Federated Learning for Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.11511
- 代码/Code: https://github.com/git-disl/stdlens

**MonoATT: Online Monocular 3D Object Detection with Adaptive Token Transformer**

- 论文/Paper: http://arxiv.org/pdf/2303.13018
- 代码/Code: None

**Dense Distinct Query for End-to-End Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.12776
- 代码/Code: https://github.com/jshilong/ddq

**OcTr: Octree-based Transformer for 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.12621
- 代码/Code: None

[返回目录/back](#Contents)



<a name="ObjectTracking"></a> 

# 目标跟踪/Object Tracking

**Referring Multi-Object Tracking**

- 论文/Paper: http://arxiv.org/pdf/2303.03366
- 代码/Code: https://github.com/wudongming97/rmot

**Unsupervised Contour Tracking of Live Cells by Mechanical and Cycle Consistency Losses**

- 论文/Paper: http://arxiv.org/pdf/2303.08364
- 代码/Code: https://github.com/junbongjang/contour-tracking

**VoxelNeXt: Fully Sparse VoxelNet for 3D Object Detection and Tracking**

- 论文/Paper: http://arxiv.org/pdf/2303.11301
- 代码/Code: https://github.com/dvlab-research/VoxelNeXt

**Visual Prompt Multi-Modal Tracking**

- 论文/Paper: http://arxiv.org/pdf/2303.10826
- 代码/Code: https://github.com/jiawen-zhu/ViPT.

**MotionTrack: Learning Robust Short-term and Long-term Motions for Multi-Object Tracking**

- 论文/Paper: http://arxiv.org/pdf/2303.10404
- 代码/Code: None

[返回目录/back](#Contents)

# 3D Object Tracking



[返回目录/back](#Contents)

<a name="TrajectoryPrediction"></a> 

## 轨迹预测/Trajectory Prediction

**IPCC-TP: Utilizing Incremental Pearson Correlation Coefficient for Joint Multi-Agent Trajectory Prediction**

- 论文/Paper: http://arxiv.org/pdf/2303.00575
- 代码/Code: None

**Trajectory-Aware Body Interaction Transformer for Multi-Person Pose Forecasting**

- 论文/Paper: http://arxiv.org/pdf/2303.05095
- 代码/Code: None

**Leapfrog Diffusion Model for Stochastic Trajectory Prediction**

- 论文/Paper: http://arxiv.org/pdf/2303.10895
- 代码/Code: https://github.com/mediabrain-sjtu/led

[返回目录/back](#Contents)

<a name="Segmentation"></a> 

## 语义分割/Segmentation

**Interactive Segmentation as Gaussian Process Classification**

- 论文/Paper: http://arxiv.org/pdf/2302.14578
- 代码/Code: None

**Foundation Model Drives Weakly Incremental Learning for Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2302.14250
- 代码/Code: None

**PolyFormer: Referring Image Segmentation as Sequential Polygon Generation**

- 论文/Paper: https://arxiv.org/abs/2302.14771
- 代码/Code: None

**ISBNet: a 3D Point Cloud Instance Segmentation Network with Instance-aware Sampling and Box-aware Dynamic Convolution**

- 论文/Paper: http://arxiv.org/pdf/2303.00246
- 代码/Code: None

**Self-Supervised Image-to-Point Distillation via Semantically Tolerant Contrastive Loss**

- 论文/Paper: https://arxiv.org/abs/2301.05709
- 代码/Code: None

**Delivering Arbitrary-Modal Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.01480
- 代码/Code: None

**Conflict-Based Cross-View Consistency for Semi-Supervised Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.01276
- 代码/Code: https://github.com/xiaoyao3302/CCVC

**Token Contrast for Weakly-Supervised Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.01267
- 代码/Code: https://github.com/rulixiang/toco 

**Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models**

- 论文/Paper: http://arxiv.org/pdf/2303.04803
- 代码/Code: None

**MP-Former: Mask-Piloted Transformer for Image Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.07336
- 代码/Code: https://github.com/IDEA-Research/MP-Former

**Efficient Semantic Segmentation by Altering Resolutions for Compressed Videos**

- 论文/Paper: http://arxiv.org/pdf/2303.07224
- 代码/Code: https://github.com/thu-lyj-lab/ar-seg

**InstMove: Instance Motion for Object-centric Video Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.08132
- 代码/Code: https://github.com/wjf5203/vnext

**DynaMask: Dynamic Mask Selection for Instance Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.07868
- 代码/Code: https://github.com/lslrh/dynamask

**MobileVOS: Real-Time Video Object Segmentation Contrastive Learning meets Knowledge Distillation**

- 论文/Paper: http://arxiv.org/pdf/2303.07815
- 代码/Code: None

**MSeg3D: Multi-modal 3D Semantic Segmentation for Autonomous Driving**

- 论文/Paper: http://arxiv.org/pdf/2303.08600
- 代码/Code: https://github.com/jialeli1/lidarseg3d

**FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.08594
- 代码/Code: https://github.com/junjiehe96/FastInst

**SIM: Semantic-aware Instance Mask Generation for Box-Supervised Instance Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.08578
- 代码/Code: https://github.com/lslrh/sim

**Unified Mask Embedding and Correspondence Learning for Self-Supervised Video Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.10100
- 代码/Code: https://github.com/0liliulei/Mask-VOS

**Generative Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.11316
- 代码/Code: https://github.com/fudan-zvg/gss

**Reliability in Semantic Segmentation: Are We on the Right Track?**

- 论文/Paper: http://arxiv.org/pdf/2303.11298
- 代码/Code: https://github.com/naver/relis

**Less is More: Reducing Task and Model Complexity for 3D Point Cloud Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.11203
- 代码/Code: https://github.com/l1997i/lim3d

**Explicit Visual Prompting for Low-Level Structure Segmentations**

- 论文/Paper: http://arxiv.org/pdf/2303.10883
- 代码/Code: https://github.com/nifangbaage/explict-visual-prompt

**Two-shot Video Object Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.12078
- 代码/Code: https://github.com/yk-pku/Two-shot-Video-Object-Segmentation

**Focused and Collaborative Feedback Integration for Interactive Image Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.11880
- 代码/Code: https://github.com/veizgyauzgyauz/fcfi

**Orthogonal Annotation Benefits Barely-supervised Medical Image Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.13090
- 代码/Code: https://github.com/hengcai-nju/desco

[返回目录/back](#Contents)



<a name="WSSS"></a>

## 弱监督语义分割/Weakly Supervised Semantic Segmentation



[返回目录/back](#Contents)

<a name="MedicalImageSegmentation"></a>

# 医学图像分割/Medical Image Segmentation



[返回目录/back](#Contents)

<a name="VideoObjectSegmentation"></a>

# 视频目标分割/Video Object Segmentation



[返回目录/back](#Contents)

<a name="InteractiveVideoObjectSegmentation"></a>

# 交互式视频目标分割/Interactive Video Object Segmentation



[返回目录/back](#Contents)

<a name="VisualTransformer"></a>

# Visual Transformer

**Mask3D: Pre-training 2D Vision Transformers by Learning Masked 3D Priors**

- 论文/Paper: http://arxiv.org/pdf/2302.14746
- 代码/Code: None

**ProxyFormer: Proxy Alignment Assisted Point Cloud Completion with Missing Part Sensitive Transformer**

- 论文/Paper: http://arxiv.org/pdf/2302.14435
- 代码/Code: https://github.com/I2-Multimedia-Lab/ProxyFormer

**Visual Atoms: Pre-training Vision Transformers with Sinusoidal Waves**

- 论文/Paper: http://arxiv.org/pdf/2303.01112
- 代码/Code: None

**MP-Former: Mask-Piloted Transformer for Image Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.07336
- 代码/Code: https://github.com/IDEA-Research/MP-Former

**TranSG: Transformer-Based Skeleton Graph Prototype Contrastive Learning with Structure-Trajectory Prompted Reconstruction for Person Re-Identification**

- 论文/Paper: http://arxiv.org/pdf/2303.06819
- 代码/Code: https://github.com/kali-hac/transg

**BiFormer: Vision Transformer with Bi-Level Routing Attention**

- 论文/Paper: http://arxiv.org/pdf/2303.08810
- 代码/Code: https://github.com/rayleizhu/biformer

**Making Vision Transformers Efficient from A Token Sparsification View**

- 论文/Paper: http://arxiv.org/pdf/2303.08685
- 代码/Code: None

**Rotation-Invariant Transformer for Point Cloud Matching**

- 论文/Paper: http://arxiv.org/pdf/2303.08231
- 代码/Code: None

**Graph Transformer GANs for Graph-Constrained House Generation**

- 论文/Paper: http://arxiv.org/pdf/2303.08225
- 代码/Code: None

**PSVT: End-to-End Multi-person 3D Pose and Shape Estimation with Progressive Video Transformers**

- 论文/Paper: http://arxiv.org/pdf/2303.09187
- 代码/Code: None

**Multimodal Feature Extraction and Fusion for Emotional Reaction Intensity Estimation and Expression Classification in Videos with Transformers**

- 论文/Paper: http://arxiv.org/pdf/2303.09164
- 代码/Code: None

**Dual-path Adaptation from Image to Video Transformers**

- 论文/Paper: http://arxiv.org/pdf/2303.09857
- 代码/Code: https://github.com/park-jungin/DualPath

**Patch-Mix Transformer for Unsupervised Domain Adaptation: A Game Perspective**

- 论文/Paper: http://arxiv.org/pdf/2303.13434
- 代码/Code: None

**POTTER: Pooling Attention Transformer for Efficient Human Mesh Recovery**

- 论文/Paper: http://arxiv.org/pdf/2303.13357
- 代码/Code: None

**MonoATT: Online Monocular 3D Object Detection with Adaptive Token Transformer**

- 论文/Paper: http://arxiv.org/pdf/2303.13018
- 代码/Code: None

**MELTR: Meta Loss Transformer for Learning to Fine-tune Video Foundation Models**

- 论文/Paper: http://arxiv.org/pdf/2303.13009
- 代码/Code: https://github.com/mlvlab/MELTR

**Spherical Transformer for LiDAR-based 3D Recognition**

- 论文/Paper: http://arxiv.org/pdf/2303.12766
- 代码/Code: https://github.com/dvlab-research/sphereformer

**OcTr: Octree-based Transformer for 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.12621
- 代码/Code: None

**Text with Knowledge Graph Augmented Transformer for Video Captioning**

- 论文/Paper: http://arxiv.org/pdf/2303.12423
- 代码/Code: None

**MAGVLT: Masked Generative Vision-and-Language Transformer**

- 论文/Paper: http://arxiv.org/pdf/2303.12208
- 代码/Code: None

[返回目录/back](#Contents)

<a name="DepthEstimation"></a>

## 深度估计/Depth Estimation

**Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation**

- 论文/Paper: https://arxiv.org/abs/2211.13202
- 代码/Code:https://github.com/noahzn/Lite-Mono

**Fully Self-Supervised Depth Estimation from Defocus Clue**

- 论文/Paper: http://arxiv.org/pdf/2303.10752
- 代码/Code: https://github.com/ehzoahis/dered

[返回目录/back](#Contents)

<a name="Retrieval"></a>

# 图像、视频检索 / Image Retrieval/Video retrieval

**Data-Free Sketch-Based Image Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2303.07775
- 代码/Code: https://github.com/abhrac/data-free-sbir

**CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not**

- 论文/Paper: http://arxiv.org/pdf/2303.13440
- 代码/Code: None

[返回目录/back](#Contents)

<a name="SuperResolution"></a>

## 超分辨率/Super Resolution

**OPE-SR: Orthogonal Position Encoding for Designing a Parameter-free Upsampling Module in Arbitrary-scale Image Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2303.01091
- 代码/Code: None

**Super-Resolution Neural Operator**

- 论文/Paper: http://arxiv.org/pdf/2303.02584
- 代码/Code: https://github.com/2y7c3/super-resolution-neural-operator

**Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2303.05156
- 代码/Code: None

**Towards High-Quality and Efficient Video Super-Resolution via Spatial-Temporal Data Overfitting**

- 论文/Paper: http://arxiv.org/pdf/2303.08331
- 代码/Code: https://github.com/coulsonlee/STDO-CVPR2023.git

[返回目录/back](#Contents)

<a name="ImageDenoising"></a>

## 图像去噪/Image Denoising - 1 篇 

**Masked Image Training for Generalizable Deep Image Denoising**

- 论文/Paper: http://arxiv.org/pdf/2303.13132
- 代码/Code: https://github.com/haoyuc/maskeddenoising

[返回目录/back](#Contents)

<a name="ImageEditing"></a>

# 图像编辑/Image Editing

**CoralStyleCLIP: Co-optimized Region and Layer Selection for Image Editing**

- 论文/Paper: http://arxiv.org/pdf/2303.05031
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ImageCompression"></a>

# 图像压缩/Image Compression

**Context-Based Trit-Plane Coding for Progressive Image Compression**

- 论文/Paper: http://arxiv.org/pdf/2303.05715
- 代码/Code: https://github.com/seungminjeon-github/ctc

[返回目录/back](#Contents)

<a name="FaceRecognition"></a>

# 人脸识别/Face Recognition

**Attribute-preserving Face Dataset Anonymization via Latent Code Optimization**

- 论文/Paper: http://arxiv.org/pdf/2303.11296
- 代码/Code: https://github.com/chi0tzp/falco

**Graphics Capsule: Learning Hierarchical 3D Face Representations from 2D Images**

- 论文/Paper: http://arxiv.org/pdf/2303.10896
- 代码/Code: None

**Sibling-Attack: Rethinking Transferable Adversarial Attacks against Face Recognition**

- 论文/Paper: http://arxiv.org/pdf/2303.12512
- 代码/Code: None

[返回目录/back](#Contents)

<a name="FaceDetection"></a>

# 人脸检测/Face Detection



[返回目录/back](#Contents)

<a name="FaceAnti-Spoofing"></a>

# 人脸活体检测/Face Anti-Spoofing



[返回目录/back](#Contents)

<a name="FaceReconstruction"></a>

## 人脸重建/Face Reconstruction

**DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration**

- 论文/Paper: http://arxiv.org/pdf/2303.06885
- 代码/Code: None

[返回目录/back](#Contents)

<a name="VideoActionDetection"></a>

# 视频动作检测/Video Action Detection

**TriDet: Temporal Action Detection with Relative Boundary Modeling**

- 论文/Paper: http://arxiv.org/pdf/2303.07347
- 代码/Code: https://github.com/sssste/tridet



[返回目录/back](#Contents)

<a name="SignLanguageTranslation"></a>

# 手语翻译/Sign Language Translation

**Continuous Sign Language Recognition with Correlation Network**

- 论文/Paper: http://arxiv.org/pdf/2303.03202
- 代码/Code: None

**Natural Language-Assisted Sign Language Recognition**

- 论文/Paper: http://arxiv.org/pdf/2303.12080
- 代码/Code: https://github.com/FangyunWei/SLRT

[返回目录/back](#Contents)

<a name="PersonRe-identification"></a>

# 行人重识别/Person Re-identification

**TranSG: Transformer-Based Skeleton Graph Prototype Contrastive Learning with Structure-Trajectory Prompted Reconstruction for Person Re-Identification**

- 论文/Paper: http://arxiv.org/pdf/2303.06819
- 代码/Code: https://github.com/kali-hac/transg



[返回目录/back](#Contents)

<a name="TalkingFace"></a>

## Talking Face

**SadTalker： Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation**

- 论文/Paper: https://arxiv.org/abs/2211.12194
- 代码/Code: https://github.com/Winfredy/SadTalker

[返回目录/back](#Contents)

<a name="HumanPoseEstimation"></a>

# 人体姿态估计/Human Pose Estimation

**PoseExaminer: Automated Testing of Out-of-Distribution Robustness in Human Pose and Shape Estimation**

- 论文/Paper: http://arxiv.org/pdf/2303.07337
- 代码/Code: https://github.com/qihao067/poseexaminer

**Mutual Information-Based Temporal Difference Learning for Human Pose Estimation in Video**

- 论文/Paper: http://arxiv.org/pdf/2303.08475
- 代码/Code: None

**SLOPER4D: A Scene-Aware Dataset for Global 4D Human Pose Estimation in Urban Environments**

- 论文/Paper: http://arxiv.org/pdf/2303.09095
- 代码/Code: None

**Self-Correctable and Adaptable Inference for Generalizable Human Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2303.11180
- 代码/Code: None

**3D Human Mesh Estimation from Virtual Markers**

- 论文/Paper: http://arxiv.org/pdf/2303.11726
- 代码/Code: https://github.com/ShirleyMaxx/VirtualMarker.

**Rigidity-Aware Detection for 6D Object Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2303.12396
- 代码/Code: None

**Object Pose Estimation with Statistical Guarantees: Conformal Keypoint Detection and Geometric Uncertainty Propagation**

- 论文/Paper: http://arxiv.org/pdf/2303.12246
- 代码/Code: None

## 

[返回目录/back](#Contents)

<a name="GAN"></a>

# GAN

**Improving GAN Training via Feature Space Shrinkage**

- 论文/Paper: http://arxiv.org/pdf/2303.01559
- 代码/Code: https://github.com/WentianZhang-ML/AdaptiveMix

**Scaling up GANs for Text-to-Image Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2303.05511
- 代码/Code: None

**Graph Transformer GANs for Graph-Constrained House Generation**

- 论文/Paper: http://arxiv.org/pdf/2303.08225
- 代码/Code: None

**Cross-GAN Auditing: Unsupervised Identification of Attribute Level Similarities and Differences between Pretrained Generative Models**

- 论文/Paper: http://arxiv.org/pdf/2303.10774
- 代码/Code: https://github.com/mattolson93/cross_gan_auditing

## 

[返回目录/back](#Contents)

<a name="AgeEstimation"></a>

# 人脸年龄估计/Age Estimation



[返回目录/back](#Contents)

<a name="FacialExpressionRecognition"></a>

# 人脸表情识别/Facial Expression Recognition



[返回目录/back](#Contents)

<a name="HandPoseEstimation"></a>

## 手势姿态估计（重建）/Hand Pose Estimation( Hand Mesh Recovery)

**Im2Hands: Learning Attentive Implicit Representation of Interacting Two-Hand Shapes**

- 论文/Paper: http://arxiv.org/pdf/2302.14348
- 代码/Code: https://github.com/jyunlee/Im2Hands

**ACR: Attention Collaboration-based Regressor for Arbitrary Two-Hand Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2303.05938
- 代码/Code: https://github.com/zhengdiyu/arbitrary-hands-3d-reconstruction

[返回目录/back](#Contents)

<a name="3DReconstruction"></a>

## 3D Reconstruction

**Unsupervised 3D Shape Reconstruction by Part Retrieval and Assembly**

- 论文/Paper: http://arxiv.org/pdf/2303.01999
- 代码/Code: None

**MobileBrick: Building LEGO for 3D Reconstruction on Mobile Devices**

- 论文/Paper: http://arxiv.org/pdf/2303.01932
- 代码/Code: None

**HairStep: Transfer Synthetic to Real Using Strand and Depth Maps for Single-View 3D Hair Modeling**

- 论文/Paper: http://arxiv.org/pdf/2303.02700
- 代码/Code: None

**NeuDA: Neural Deformable Anchor for High-Fidelity Implicit Surface Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2303.02375
- 代码/Code: None

**Structural Multiplane Image: Bridging Neural View Synthesis and 3D Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2303.05937
- 代码/Code: None



[返回目录/back](#Contents)

<a name="FrameInterpolation"></a>

## 视频插帧/Frame Interpolation

**Extracting Motion and Appearance via Inter-Frame Attention for Efficient Video Frame Interpolation**

- 论文/Paper: http://arxiv.org/pdf/2303.00440
- 代码/Code: https://github.com/MCG-NJU/EMA-VFI

[返回目录/back](#Contents)

<a name="3DPointCloud"></a>

## 3D点云/3D point cloud

**ISBNet: a 3D Point Cloud Instance Segmentation Network with Instance-aware Sampling and Box-aware Dynamic Convolution**

- 论文/Paper: http://arxiv.org/pdf/2303.00246
- 代码/Code: None

**Self-Supervised Image-to-Point Distillation via Semantically Tolerant Contrastive Loss**

- 论文/Paper: https://arxiv.org/abs/2301.05709
- 代码/Code: None

**Neural Intrinsic Embedding for Non-rigid Point Cloud Matching**

- 论文/Paper: http://arxiv.org/pdf/2303.01038
- 代码/Code: None

**ACL-SPC: Adaptive Closed-Loop system for Self-Supervised Point Cloud Completion**

- 论文/Paper: http://arxiv.org/pdf/2303.01979
- 代码/Code: https://github.com/Sangminhong/ACL-SPC_PyTorch

**PointCert: Point Cloud Classification with Deterministic Certified Robustness Guarantees**

- 论文/Paper: http://arxiv.org/pdf/2303.01959
- 代码/Code: None

**SCPNet: Semantic Scene Completion on Point Cloud**

- 论文/Paper: http://arxiv.org/pdf/2303.06884
- 代码/Code: None

**Parameter is Not All You Need: Starting from Non-Parametric Networks for 3D Point Cloud Analysis**

- 论文/Paper: http://arxiv.org/pdf/2303.08134
- 代码/Code: https://github.com/zrrskywalker/point-nn

**PiMAE: Point Cloud and Image Interactive Masked Autoencoders for 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.08129
- 代码/Code: https://github.com/blvlab/pimae

**Frequency-Modulated Point Cloud Rendering with Easy Editing**

- 论文/Paper: http://arxiv.org/pdf/2303.07596
- 代码/Code: None

**MSF: Motion-guided Sequential Fusion for Efficient 3D Object Detection from Point Cloud Sequences**

- 论文/Paper: http://arxiv.org/pdf/2303.08316
- 代码/Code: \url{https://github.com/skyhehe123/MSF}.

**Rotation-Invariant Transformer for Point Cloud Matching**

- 论文/Paper: http://arxiv.org/pdf/2303.08231
- 代码/Code: None

**Deep Graph-based Spatial Consistency for Robust Non-rigid Point Cloud Registration**

- 论文/Paper: http://arxiv.org/pdf/2303.09950
- 代码/Code: https://github.com/qinzheng93/GraphSCNet

**Less is More: Reducing Task and Model Complexity for 3D Point Cloud Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.11203
- 代码/Code: https://github.com/l1997i/lim3d

**Novel Class Discovery for 3D Point Cloud Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.11610
- 代码/Code: https://github.com/luigiriz/nops

**Unsupervised Deep Probabilistic Approach for Partial Point Cloud Registration**

- 论文/Paper: http://arxiv.org/pdf/2303.13290
- 代码/Code: https://github.com/gfmei/udpreg

[返回目录/back](#Contents)

<a name="AnomalyDetection"></a>

# Anomaly Detection

**Diversity-Measurable Anomaly Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.05047
- 代码/Code: None

[返回目录/back](#Contents)

<a name="Other"></a>

## 其他/Other

**PA&DA: Jointly Sampling PAth and DAta for Consistent NAS**

- 论文/Paper: http://arxiv.org/pdf/2302.14772
- 代码/Code: https://github.com/ShunLu91/PA-DA

**Generic-to-Specific Distillation of Masked Autoencoders**

- 论文/Paper: http://arxiv.org/pdf/2302.14771
- 代码/Code: https://github.com/pengzhiliang/G2SD.

**Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger**

- 论文/Paper: http://arxiv.org/pdf/2302.14677
- 代码/Code: None

**Turning a CLIP Model into a Scene Text Detector**

- 论文/Paper: http://arxiv.org/pdf/2302.14338
- 代码/Code: None

**Adversarial Attack with Raindrops**

- 论文/Paper: http://arxiv.org/pdf/2302.14267
- 代码/Code: None

**Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning**

- 论文/Paper: http://arxiv.org/pdf/2302.14115
- 代码/Code: None

**DART: Diversify-Aggregate-Repeat Training Improves Generalization of Neural Networks**

- 论文/Paper: http://arxiv.org/pdf/2302.14685
- 代码/Code: None

**Neural Video Compression with Diverse Contexts**

- 论文/Paper: http://arxiv.org/pdf/2302.14402
- 代码/Code: https://github.com/microsoft/DCVC

**Learning to Retain while Acquiring: Combating Distribution-Shift in Adversarial Data-Free Knowledge Distillation**

- 论文/Paper: http://arxiv.org/pdf/2302.14290
- 代码/Code: None

**Efficient and Explicit Modelling of Image Hierarchies for Image Restoration**

- 论文/Paper: http://arxiv.org/pdf/2303.00748
- 代码/Code: https://github.com/ofsoundof/grl-image-restoration

**Quality-aware Pre-trained Models for Blind Image Quality Assessment**

- 论文/Paper: http://arxiv.org/pdf/2303.00521
- 代码/Code: None

**Renderable Neural Radiance Map for Visual Navigation**

- 论文/Paper: http://arxiv.org/pdf/2303.00304
- 代码/Code: None

**Single Image Backdoor Inversion via Robust Smoothed Classifiers**

- 论文/Paper: http://arxiv.org/pdf/2303.00215
- 代码/Code: https://github.com/locuslab/smoothinv

**Towards Generalisable Video Moment Retrieval: Visual-Dynamic Injection to Image-Text Pre-Training**

- 论文/Paper: http://arxiv.org/pdf/2303.00040
- 代码/Code: None

**Zero-Shot Text-to-Parameter Translation for Game Character Auto-Creation**

- 论文/Paper: http://arxiv.org/pdf/2303.01311
- 代码/Code: None

**MixPHM: Redundancy-Aware Parameter-Efficient Tuning for Low-Resource Visual Question Answering**

- 论文/Paper: http://arxiv.org/pdf/2303.01239
- 代码/Code: https://github.com/jingjing12110/MixPHM

**Disentangling Orthogonal Planes for Indoor Panoramic Room Layout Estimation with Cross-Scale Distortion Awareness**

- 论文/Paper: http://arxiv.org/pdf/2303.00971
- 代码/Code: https://github.com/zhijieshen-bjtu/dopnet

**Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2303.00914
- 代码/Code: None

**Towards Trustable Skin Cancer Diagnosis via Rewriting Model's Decision**

- 论文/Paper: http://arxiv.org/pdf/2303.00885
- 代码/Code: None

**Geometric Visual Similarity Learning in 3D Medical Image Self-supervised Pre-training**

- 论文/Paper: http://arxiv.org/pdf/2303.00874
- 代码/Code: https://github.com/yutinghe-list/gvsl

**Demystifying Causal Features on Adversarial Examples and Causal Inoculation for Robust Network by Adversarial Instrumental Variable Regression**

- 论文/Paper: http://arxiv.org/pdf/2303.01052
- 代码/Code: None

**UniDexGrasp: Universal Robotic Dexterous Grasping via Learning Diverse Proposal Generation and Goal-Conditioned Policy**

- 论文/Paper: http://arxiv.org/pdf/2303.00938
- 代码/Code: None

**Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners**

- 论文/Paper: http://arxiv.org/pdf/2303.02151
- 代码/Code: https://github.com/ZrrSkywalker/CaFo.

**Zero-shot Object Counting**

- 论文/Paper: http://arxiv.org/pdf/2303.02001
- 代码/Code: https://github.com/cvlab-stonybrook/zero-shot-counting

**EcoTTA: Memory-Efficient Continual Test-time Adaptation via Self-distilled Regularization**

- 论文/Paper: http://arxiv.org/pdf/2303.01904
- 代码/Code: None

**Prompting Large Language Models with Answer Heuristics for Knowledge-based Visual Question Answering**

- 论文/Paper: http://arxiv.org/pdf/2303.01903
- 代码/Code: https://github.com/MILVLG/prophet

**Intrinsic Physical Concepts Discovery with Object-Centric Predictive Models**

- 论文/Paper: http://arxiv.org/pdf/2303.01869
- 代码/Code: None

**Visual Exemplar Driven Task-Prompting for Unified Perception in Autonomous Driving**

- 论文/Paper: http://arxiv.org/pdf/2303.01788
- 代码/Code: None

**Diverse 3D Hand Gesture Prediction from Body Dynamics by Bilateral Hand Disentanglement**

- 论文/Paper: http://arxiv.org/pdf/2303.01765
- 代码/Code: None

**Learning Common Rationale to Improve Self-Supervised Representation for Fine-Grained Visual Recognition Problems**

- 论文/Paper: http://arxiv.org/pdf/2303.01669
- 代码/Code: None

**Hierarchical discriminative learning improves visual representations of biomedical microscopy**

- 论文/Paper: http://arxiv.org/pdf/2303.01605
- 代码/Code: None

**A Meta-Learning Approach to Predicting Performance and Data Requirements**

- 论文/Paper: http://arxiv.org/pdf/2303.01598
- 代码/Code: None

**DejaVu: Conditional Regenerative Learning to Enhance Dense Prediction**

- 论文/Paper: http://arxiv.org/pdf/2303.01573
- 代码/Code: None

**Detecting Human-Object Contact in Images**

- 论文/Paper: http://arxiv.org/pdf/2303.03373
- 代码/Code: None

**MACARONS: Mapping And Coverage Anticipation with RGB Online Self-Supervision**

- 论文/Paper: http://arxiv.org/pdf/2303.03315
- 代码/Code: None

**Masked Images Are Counterfactual Samples for Robust Fine-tuning**

- 论文/Paper: http://arxiv.org/pdf/2303.03052
- 代码/Code: None

**UniHCP: A Unified Model for Human-Centric Perceptions**

- 论文/Paper: http://arxiv.org/pdf/2303.02936
- 代码/Code: None

**PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow**

- 论文/Paper: http://arxiv.org/pdf/2303.02595
- 代码/Code: None

**CapDet: Unifying Dense Captioning and Open-World Detection Pretraining**

- 论文/Paper: http://arxiv.org/pdf/2303.02489
- 代码/Code: None

**DistilPose: Tokenized Pose Regression with Heatmap Distillation**

- 论文/Paper: http://arxiv.org/pdf/2303.02455
- 代码/Code: None

**DeepMAD: Mathematical Architecture Design for Deep Convolutional Neural Network**

- 论文/Paper: http://arxiv.org/pdf/2303.02165
- 代码/Code: https://github.com/alibaba/lightweight-neural-architecture-search

**Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization**

- 论文/Paper: http://arxiv.org/pdf/2303.03108
- 代码/Code: None

**Meta-Explore: Exploratory Hierarchical Vision-and-Language Navigation Using Scene Object Spectrum Grounding**

- 论文/Paper: http://arxiv.org/pdf/2303.04077
- 代码/Code: None

**Guiding Pseudo-labels with Uncertainty Estimation for Test-Time Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2303.03770
- 代码/Code: None

**Learning Discriminative Representations for Skeleton Based Action Recognition**

- 论文/Paper: http://arxiv.org/pdf/2303.03729
- 代码/Code: None

**MOSO: Decomposing MOtion, Scene and Object for Video Prediction**

- 论文/Paper: http://arxiv.org/pdf/2303.03684
- 代码/Code: None

**RM-Depth: Unsupervised Learning of Recurrent Monocular Depth in Dynamic Scenes**

- 论文/Paper: http://arxiv.org/pdf/2303.04456
- 代码/Code: https://github.com/twhui/rm-depth

**A Light Weight Model for Active Speaker Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.04439
- 代码/Code: https://github.com/junhua-liao/light-asd

**Where We Are and What We're Looking At: Query Based Worldwide Image Geo-localization Using Hierarchies and Scenes**

- 论文/Paper: http://arxiv.org/pdf/2303.04249
- 代码/Code: None

**CUDA: Convolution-based Unlearnable Datasets**

- 论文/Paper: http://arxiv.org/pdf/2303.04278
- 代码/Code: None

**Masked Image Modeling with Local Multi-Scale Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2303.05251
- 代码/Code: None

**Revisiting Rotation Averaging: Uncertainties and Robust Losses**

- 论文/Paper: http://arxiv.org/pdf/2303.05195
- 代码/Code: https://github.com/zhangganlin/globalsfmpy

**Text-Visual Prompting for Efficient 2D Temporal Video Grounding**

- 论文/Paper: http://arxiv.org/pdf/2303.04995
- 代码/Code: None

**MVImgNet: A Large-scale Dataset of Multi-view Images**

- 论文/Paper: http://arxiv.org/pdf/2303.06042
- 代码/Code: None

**Neuron Structure Modeling for Generalizable Remote Physiological Measurement**

- 论文/Paper: http://arxiv.org/pdf/2303.05955
- 代码/Code: https://github.com/lupaopao/nest

**3D Cinemagraphy from a Single Image**

- 论文/Paper: http://arxiv.org/pdf/2303.05724
- 代码/Code: None

**HumanBench: Towards General Human-centric Perception with Projector Assisted Pretraining**

- 论文/Paper: http://arxiv.org/pdf/2303.05675
- 代码/Code: https://github.com/OpenGVLab/HumanBench

**TrojDiff: Trojan Attacks on Diffusion Models with Diverse Targets**

- 论文/Paper: http://arxiv.org/pdf/2303.05762
- 代码/Code: https://github.com/chenweixin107/trojdiff

**Modality-Agnostic Debiasing for Single Domain Generalization**

- 论文/Paper: http://arxiv.org/pdf/2303.07123
- 代码/Code: None

**Upcycling Models under Domain and Category Shift**

- 论文/Paper: http://arxiv.org/pdf/2303.07110
- 代码/Code: https://github.com/ispc-lab/glc

**Prototype-based Embedding Network for Scene Graph Generation**

- 论文/Paper: http://arxiv.org/pdf/2303.07096
- 代码/Code: None

**MSINet: Twins Contrastive Search of Multi-Scale Interaction for Object ReID**

- 论文/Paper: http://arxiv.org/pdf/2303.07065
- 代码/Code: https://github.com/vimar-gu/MSINet

**Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling**

- 论文/Paper: http://arxiv.org/pdf/2303.06949
- 代码/Code: None

**Progressive Open Space Expansion for Open-Set Model Attribution**

- 论文/Paper: http://arxiv.org/pdf/2303.06877
- 代码/Code: https://github.com/tianyunyoung/pose

**Interventional Bag Multi-Instance Learning On Whole-Slide Pathological Images**

- 论文/Paper: http://arxiv.org/pdf/2303.06873
- 代码/Code: https://github.com/HHHedo/IBMIL

**Three Guidelines You Should Know for Universally Slimmable Self-Supervised Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.06870
- 代码/Code: https://github.com/megvii-research/US3L-CVPR2023.

**Adaptive Data-Free Quantization**

- 论文/Paper: http://arxiv.org/pdf/2303.06869
- 代码/Code: https://github.com/hfutqian/adadfq

**Learning Distortion Invariant Representation for Image Restoration from A Causality Perspective**

- 论文/Paper: http://arxiv.org/pdf/2303.06859
- 代码/Code: https://github.com/lixinustc/casual-ir-dil

**Dynamic Neural Network for Multi-Task Learning Searching across Diverse Network Topologies**

- 论文/Paper: http://arxiv.org/pdf/2303.06856
- 代码/Code: None

**Universal Instance Perception as Object Discovery and Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2303.06674
- 代码/Code: https://github.com/MasterBin-IIAU/UNINEXT

**Iterative Geometry Encoding Volume for Stereo Matching**

- 论文/Paper: http://arxiv.org/pdf/2303.06615
- 代码/Code: https://github.com/gangweix/igev

**Regularized Vector Quantization for Tokenized Image Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2303.06424
- 代码/Code: None

**Semi-supervised Hand Appearance Recovery via Structure Disentanglement and Dual Adversarial Discrimination**

- 论文/Paper: http://arxiv.org/pdf/2303.06380
- 代码/Code: None

**CASP-Net: Rethinking Video Saliency Prediction from an Audio-VisualConsistency Perceptual Perspective**

- 论文/Paper: http://arxiv.org/pdf/2303.06357
- 代码/Code: None

**DeltaEdit: Exploring Text-free Training for Text-Driven Image Manipulation**

- 论文/Paper: http://arxiv.org/pdf/2303.06285
- 代码/Code: https://github.com/yueming6568/deltaedit

**Diversity-Aware Meta Visual Prompting**

- 论文/Paper: http://arxiv.org/pdf/2303.08138
- 代码/Code: https://github.com/shikiw/dam-vp

**Blind Video Deflickering by Neural Filtering with a Flawed Atlas**

- 论文/Paper: http://arxiv.org/pdf/2303.08120
- 代码/Code: https://github.com/chenyanglei/all-in-one-deflicker

**Non-Contrastive Unsupervised Learning of Physiological Signals from Video**

- 论文/Paper: http://arxiv.org/pdf/2303.07944
- 代码/Code: None

**DAA: A Delta Age AdaIN operation for age estimation via binary code transformer**

- 论文/Paper: http://arxiv.org/pdf/2303.07929
- 代码/Code: None

**You Can Ground Earlier than See: An Effective and Efficient Pipeline for Temporal Sentence Grounding in Compressed Videos**

- 论文/Paper: http://arxiv.org/pdf/2303.07863
- 代码/Code: None

**NEF: Neural Edge Fields for 3D Parametric Curve Reconstruction from Multi-view Images**

- 论文/Paper: http://arxiv.org/pdf/2303.07653
- 代码/Code: None

**I$^2$-SDF: Intrinsic Indoor Scene Reconstruction and Editing via Raytracing in Neural SDFs**

- 论文/Paper: http://arxiv.org/pdf/2303.07634
- 代码/Code: None

**V2V4Real: A Real-world Large-scale Dataset for Vehicle-to-Vehicle Cooperative Perception**

- 论文/Paper: http://arxiv.org/pdf/2303.07601
- 代码/Code: https://github.com/ucla-mobility/V2V4Real

**Bi-directional Distribution Alignment for Transductive Zero-Shot Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.08698
- 代码/Code: https://github.com/Zhicaiwww/Bi-VAEGAN

**Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry**

- 论文/Paper: http://arxiv.org/pdf/2303.08658
- 代码/Code: https://github.com/Kebii/R2ET.

**Lana: A Language-Capable Navigator for Instruction Following and Generation**

- 论文/Paper: http://arxiv.org/pdf/2303.08409
- 代码/Code: https://github.com/wxh1996/lana-vln

**Rethinking Optical Flow from Geometric Matching Consistent Perspective**

- 论文/Paper: http://arxiv.org/pdf/2303.08384
- 代码/Code: https://github.com/dqiaole/matchflow

**Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring**

- 论文/Paper: http://arxiv.org/pdf/2303.08536
- 代码/Code: https://github.com/joannahong/av-relscore

**Hubs and Hyperspheres: Reducing Hubness and Improving Transductive Few-shot Learning with Hyperspherical Embeddings**

- 论文/Paper: http://arxiv.org/pdf/2303.09352
- 代码/Code: https://github.com/uitml/nohub

**A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2303.09165
- 代码/Code: https://github.com/huitangtang/on_the_utility_of_synthetic_data

**Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.09483
- 代码/Code: https://github.com/kim-sanghwan/ancl

**TBP-Former: Learning Temporal Bird's-Eye-View Pyramid for Joint Perception and Prediction in Vision-Centric Autonomous Driving**

- 论文/Paper: http://arxiv.org/pdf/2303.09998
- 代码/Code: None

**Adversarial Counterfactual Visual Explanations**

- 论文/Paper: http://arxiv.org/pdf/2303.09962
- 代码/Code: None

**A Dynamic Multi-Scale Voxel Flow Network for Video Prediction**

- 论文/Paper: http://arxiv.org/pdf/2303.09875
- 代码/Code: None

**TeSLA: Test-Time Self-Learning With Automatic Adversarial Augmentation**

- 论文/Paper: http://arxiv.org/pdf/2303.09870
- 代码/Code: None

**Video Dehazing via a Multi-Range Temporal Alignment Network with Physical Prior**

- 论文/Paper: http://arxiv.org/pdf/2303.09757
- 代码/Code: https://github.com/jiaqixuac/MAP-Net

**LOCATE: Localize and Transfer Object Parts for Weakly Supervised Affordance Grounding**

- 论文/Paper: http://arxiv.org/pdf/2303.09665
- 代码/Code: None

**On the Effects of Self-supervision and Contrastive Alignment in Deep Multi-view Clustering**

- 论文/Paper: http://arxiv.org/pdf/2303.09877
- 代码/Code: https://github.com/DanielTrosten/DeepMVC

**3D Concept Learning and Reasoning from Multi-View Images**

- 论文/Paper: http://arxiv.org/pdf/2303.11327
- 代码/Code: None

**Picture that Sketch: Photorealistic Image Generation from Abstract Sketches**

- 论文/Paper: http://arxiv.org/pdf/2303.11162
- 代码/Code: None

**Coreset Sampling from Open-Set for Fine-Grained Self-Supervised Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.11101
- 代码/Code: None

**Boosting Semi-Supervised Learning by Exploiting All Unlabeled Data**

- 论文/Paper: http://arxiv.org/pdf/2303.11066
- 代码/Code: https://github.com/megvii-research/FullMatch.

**Feature Alignment and Uniformity for Test Time Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2303.10902
- 代码/Code: None

**EqMotion: Equivariant Multi-agent Motion Prediction with Invariant Interaction Reasoning**

- 论文/Paper: http://arxiv.org/pdf/2303.10876
- 代码/Code: https://github.com/mediabrain-sjtu/eqmotion

**Trainable Projected Gradient Method for Robust Fine-tuning**

- 论文/Paper: http://arxiv.org/pdf/2303.10720
- 代码/Code: \url{https://github.com/PotatoTian/TPGM}.

**Partial Network Cloning**

- 论文/Paper: http://arxiv.org/pdf/2303.10597
- 代码/Code: https://github.com/jngwenye/pncloning

**Divide and Conquer: Answering Questions with Object Factorization and Compositional Reasoning**

- 论文/Paper: http://arxiv.org/pdf/2303.10482
- 代码/Code: https://github.com/szzexpoi/poem

**Uncertainty-Aware Optimal Transport for Semantically Coherent Out-of-Distribution Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.10449
- 代码/Code: https://github.com/lufan31/et-ood

**DeAR: Debiasing Vision-Language Models with Additive Residuals**

- 论文/Paper: http://arxiv.org/pdf/2303.10431
- 代码/Code: None

**3DQD: Generalized Deep 3D Shape Prior via Part-Discretized Diffusion Process**

- 论文/Paper: http://arxiv.org/pdf/2303.10406
- 代码/Code: https://github.com/colorful-liyu/3dqd

**Sharpness-Aware Gradient Matching for Domain Generalization**

- 论文/Paper: http://arxiv.org/pdf/2303.10353
- 代码/Code: https://github.com/wang-pengfei/sagm

**Extracting Class Activation Maps from Non-Discriminative Features as well**

- 论文/Paper: http://arxiv.org/pdf/2303.10334
- 代码/Code: https://github.com/zhaozhengchen/lpcam

**Make Landscape Flatter in Differentially Private Federated Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.11242
- 代码/Code: None

**Computationally Budgeted Continual Learning: What Does Matter?**

- 论文/Paper: http://arxiv.org/pdf/2303.11165
- 代码/Code: https://github.com/drimpossible/BudgetCL.

**TWINS: A Fine-Tuning Framework for Improved Transferability of Adversarial Robustness and Generalization**

- 论文/Paper: http://arxiv.org/pdf/2303.11135
- 代码/Code: https://github.com/ziquanliu/cvpr2023-twins

**Efficient Map Sparsification Based on 2D and 3D Discretized Grids**

- 论文/Paper: http://arxiv.org/pdf/2303.10882
- 代码/Code: https://github.com/fishmarch/SLAM_Map_Compression.

**ProphNet: Efficient Agent-Centric Motion Forecasting with Anchor-Informed Proposals**

- 论文/Paper: http://arxiv.org/pdf/2303.12071
- 代码/Code: None

**Joint Visual Grounding and Tracking with Natural Language Specification**

- 论文/Paper: http://arxiv.org/pdf/2303.12027
- 代码/Code: https://github.com/lizhou-cs/JointNLT.

**Automatic evaluation of herding behavior in towed fishing gear using end-to-end training of CNN and attention-based networks**

- 论文/Paper: http://arxiv.org/pdf/2303.12016
- 代码/Code: None

**Learning A Sparse Transformer Network for Effective Image Deraining**

- 论文/Paper: http://arxiv.org/pdf/2303.11950
- 代码/Code: https://github.com/cschenxiang/drsformer

**Context De-confounded Emotion Recognition**

- 论文/Paper: http://arxiv.org/pdf/2303.11921
- 代码/Code: None

**Solving Oscillation Problem in Post-Training Quantization Through a Theoretical Perspective**

- 论文/Paper: http://arxiv.org/pdf/2303.11906
- 代码/Code: None

**The Treasure Beneath Multiple Annotations: An Uncertainty-aware Edge Detector**

- 论文/Paper: http://arxiv.org/pdf/2303.11828
- 代码/Code: https://github.com/zhoucx117/uaed

**Propagate And Calibrate: Real-time Passive Non-line-of-sight Tracking**

- 论文/Paper: http://arxiv.org/pdf/2303.11791
- 代码/Code: None

**Detecting Everything in the Open World: Towards Universal Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.11749
- 代码/Code: None

**Data-efficient Large Scale Place Recognition with Graded Similarity Supervision**

- 论文/Paper: http://arxiv.org/pdf/2303.11739
- 代码/Code: https://github.com/marialeyvallina/generalized_contrastive_loss

**Abstract Visual Reasoning: An Algebraic Approach for Solving Raven's Progressive Matrices**

- 论文/Paper: http://arxiv.org/pdf/2303.11730
- 代码/Code: https://github.com/xu-jingyi/algebraicmr

**Learning a 3D Morphable Face Reflectance Model from Low-cost Data**

- 论文/Paper: http://arxiv.org/pdf/2303.11686
- 代码/Code: https://github.com/yxuhan/reflectancemm

**Full or Weak annotations? An adaptive strategy for budget-constrained annotation campaigns**

- 论文/Paper: http://arxiv.org/pdf/2303.11678
- 代码/Code: None

**ALOFT: A Lightweight MLP-like Architecture with Dynamic Low-frequency Transform for Domain Generalization**

- 论文/Paper: http://arxiv.org/pdf/2303.11674
- 代码/Code: https://github.com/lingeringlight/aloft

**Visibility Constrained Wide-band Illumination Spectrum Design for Seeing-in-the-Dark**

- 论文/Paper: http://arxiv.org/pdf/2303.11642
- 代码/Code: Available:https://github.com/MyNiuuu/VCSD.

**Human Pose as Compositional Tokens**

- 论文/Paper: http://arxiv.org/pdf/2303.11638
- 代码/Code: https://github.com/Gengzigang/PCT

**Equiangular Basis Vectors**

- 论文/Paper: http://arxiv.org/pdf/2303.11637
- 代码/Code: https://github.com/njust-vipgroup/equiangular-basis-vectors

**HRDFuse: Monocular 360°Depth Estimation by Collaboratively Learning Holistic-with-Regional Depth Distributions**

- 论文/Paper: http://arxiv.org/pdf/2303.11616
- 代码/Code: None

**Boundary Unlearning**

- 论文/Paper: http://arxiv.org/pdf/2303.11570
- 代码/Code: None

**One-to-Few Label Assignment for End-to-End Dense Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.11567
- 代码/Code: https://github.com/strongwolf/o2f.

**Fix the Noise: Disentangling Source Feature for Controllable Domain Translation**

- 论文/Paper: http://arxiv.org/pdf/2303.11545
- 代码/Code: https://github.com/LeeDongYeun/FixNoise

**PRISE: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment**

- 论文/Paper: http://arxiv.org/pdf/2303.11526
- 代码/Code: fromhttps://github.com/Zhang-VISLab.

**Sketch2Saliency: Learning to Detect Salient Objects from Human Drawings**

- 论文/Paper: http://arxiv.org/pdf/2303.11502
- 代码/Code: None

**Polynomial Implicit Neural Representations For Large Diverse Datasets**

- 论文/Paper: http://arxiv.org/pdf/2303.11424
- 代码/Code: https://github.com/rajhans0/poly_inr

**Persistent Nature: A Generative Model of Unbounded 3D Worlds**

- 论文/Paper: http://arxiv.org/pdf/2303.13515
- 代码/Code: None

**MV-JAR: Masked Voxel Jigsaw and Reconstruction for LiDAR-Based Self-Supervised Pre-Training**

- 论文/Paper: http://arxiv.org/pdf/2303.13510
- 代码/Code: https://github.com/smartbot-pjlab/mv-jar

**NS3D: Neuro-Symbolic Grounding of 3D Objects and Relations**

- 论文/Paper: http://arxiv.org/pdf/2303.13483
- 代码/Code: None

**Egocentric Audio-Visual Object Localization**

- 论文/Paper: http://arxiv.org/pdf/2303.13471
- 代码/Code: https://github.com/wikichao/ego-av-loc

**Improving Generalization with Domain Convex Game**

- 论文/Paper: http://arxiv.org/pdf/2303.13297
- 代码/Code: None

**Visual-Language Prompt Tuning with Knowledge-guided Context Optimization**

- 论文/Paper: http://arxiv.org/pdf/2303.13283
- 代码/Code: https://github.com/htyao89/kgcoop

**TAPS3D: Text-Guided 3D Textured Shape Generation from Pseudo Supervision**

- 论文/Paper: http://arxiv.org/pdf/2303.13273
- 代码/Code: https://github.com/plusmultiply/taps3d

**A Bag-of-Prototypes Representation for Dataset-Level Applications**

- 论文/Paper: http://arxiv.org/pdf/2303.13251
- 代码/Code: None

**CrOC: Cross-View Online Clustering for Dense Visual Representation Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.13245
- 代码/Code: https://github.com/stegmuel/croc

**Transforming Radiance Field with Lipschitz Network for Photorealistic 3D Scene Stylization**

- 论文/Paper: http://arxiv.org/pdf/2303.13232
- 代码/Code: None

**Exploring Structured Semantic Prior for Multi Label Recognition with Incomplete Labels**

- 论文/Paper: http://arxiv.org/pdf/2303.13223
- 代码/Code: https://github.com/jameslahm/SCPNet.

**Marching-Primitives: Shape Abstraction from Signed Distance Function**

- 论文/Paper: http://arxiv.org/pdf/2303.13190
- 代码/Code: https://github.com/ChirikjianLab/Marching-Primitives.git.

**CP$^3$: Channel Pruning Plug-in for Point-based Networks**

- 论文/Paper: http://arxiv.org/pdf/2303.13097
- 代码/Code: None

**Box-Level Active Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.13089
- 代码/Code: https://github.com/lyumengyao/blad.

**Robust Generalization against Photon-Limited Corruptions via Worst-Case Sharpness Minimization**

- 论文/Paper: http://arxiv.org/pdf/2303.13087
- 代码/Code: https://github.com/zhuohuangai/sharpdro

**CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching**

- 论文/Paper: http://arxiv.org/pdf/2303.13076
- 代码/Code: https://github.com/tgxs002/cora

**PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360$^{\circ}$**

- 论文/Paper: http://arxiv.org/pdf/2303.13071
- 代码/Code: None

**Human Guided Ground-truth Generation for Realistic Image Super-resolution**

- 论文/Paper: http://arxiv.org/pdf/2303.13069
- 代码/Code: https://github.com/chrisdud0257/hggt

**SIEDOB: Semantic Image Editing by Disentangling Object and Background**

- 论文/Paper: http://arxiv.org/pdf/2303.13062
- 代码/Code: https://github.com/wuyangluo/siedob

**Hierarchical Semantic Contrast for Scene-aware Video Anomaly Detection**

- 论文/Paper: http://arxiv.org/pdf/2303.13051
- 代码/Code: None

**Top-Down Visual Attention from Analysis by Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2303.13043
- 代码/Code: None

**Semantic Ray: Learning a Generalizable Semantic Field with Cross-Reprojection Attention**

- 论文/Paper: http://arxiv.org/pdf/2303.13014
- 代码/Code: None

**Backdoor Defense via Adaptively Splitting Poisoned Dataset**

- 论文/Paper: http://arxiv.org/pdf/2303.12993
- 代码/Code: https://github.com/kuofenggao/asd

**LightPainter: Interactive Portrait Relighting with Freehand Scribble**

- 论文/Paper: http://arxiv.org/pdf/2303.12950
- 代码/Code: None

**Dense-Localizing Audio-Visual Events in Untrimmed Videos: A Large-Scale Benchmark and Baseline**

- 论文/Paper: http://arxiv.org/pdf/2303.12930
- 代码/Code: None

**Don't FREAK Out: A Frequency-Inspired Approach to Detecting Backdoor Poisoned Samples in DNNs**

- 论文/Paper: http://arxiv.org/pdf/2303.13211
- 代码/Code: None

**Learning a Practical SDR-to-HDRTV Up-conversion using New Dataset and Degradation Models**

- 论文/Paper: http://arxiv.org/pdf/2303.13031
- 代码/Code: https://github.com/andreguo/hdrtvdm

**Open Set Action Recognition via Multi-Label Evidential Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.12698
- 代码/Code: None

**Dense Network Expansion for Class Incremental Learning**

- 论文/Paper: http://arxiv.org/pdf/2303.12696
- 代码/Code: None

**VecFontSDF: Learning to Reconstruct and Synthesize High-quality Vector Fonts via Signed Distance Functions**

- 论文/Paper: http://arxiv.org/pdf/2303.12675
- 代码/Code: None

**Correlational Image Modeling for Self-Supervised Visual Pre-Training**

- 论文/Paper: http://arxiv.org/pdf/2303.12670
- 代码/Code: None

**An Extended Study of Human-like Behavior under Adversarial Training**

- 论文/Paper: http://arxiv.org/pdf/2303.12669
- 代码/Code: None


**RaBit: Parametric Modeling of 3D Biped Cartoon Characters with a Topological-consistent Dataset**

- 论文/Paper: http://arxiv.org/pdf/2303.12564
- 代码/Code: None

**Is BERT Blind? Exploring the Effect of Vision-and-Language Pretraining on Visual Language Understanding**

- 论文/Paper: http://arxiv.org/pdf/2303.12513
- 代码/Code: None




**BiCro: Noisy Correspondence Rectification for Multi-modality Data via Bi-directional Cross-modal Similarity Consistency**

- 论文/Paper: http://arxiv.org/pdf/2303.12419
- 代码/Code: https://github.com/xu5zhao/bicro

**Balanced Spherical Grid for Egocentric View Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2303.12408
- 代码/Code: https://github.com/changwoonchoi/EgoNeRF


**Weakly Supervised Video Representation Learning with Unaligned Text for Sequential Videos**

- 论文/Paper: http://arxiv.org/pdf/2303.12370
- 代码/Code: https://github.com/svip-lab/weaksvr

**Re-thinking Federated Active Learning based on Inter-class Diversity**

- 论文/Paper: http://arxiv.org/pdf/2303.12317
- 代码/Code: https://github.com/raymin0223/logo




**Learning a Depth Covariance Function**

- 论文/Paper: http://arxiv.org/pdf/2303.12157
- 代码/Code: None

**Positive-Augmented Constrastive Learning for Image and Video Captioning Evaluation**

- 论文/Paper: http://arxiv.org/pdf/2303.12112
- 代码/Code: https://github.com/aimagelab/pacscore

**Music-Driven Group Choreography**

- 论文/Paper: http://arxiv.org/pdf/2303.12337
- 代码/Code: None

[返回目录/back](#Contents)