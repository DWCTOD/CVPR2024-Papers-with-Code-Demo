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
- [视频插帧/Frame Interpolation](#FrameInterpolation)
- [视频超分 / Video Super-Resolution](#VideoSuper-Resolution)
- [ 3D点云/3D point cloud](#3DPointCloud)
- [标签噪声 / Label-Noise](#Label-Noise)
- [对抗样本/Adversarial Examples](#AdversarialExamples)
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



[返回目录/back](#Contents)

<a name="NAS"></a> 

## NAS



[返回目录/back](#Contents)

<a name="KnowledgeDistillation"></a> 

## Knowledge Distillation

**Paper title: Generic-to-Specific Distillation of Masked Autoencoders**

- 论文/Paper: https://arxiv.org/abs/2302.14771
- 代码/Code: https://github.com/pengzhiliang/G2SD

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

[返回目录/back](#Contents)

<a name="ContrastiveLearning"></a> 

## Contrastive Learning



[返回目录/back](#Contents)

<a name="CapsuleNetwork"></a> 

# 胶囊网络 / Capsule Network



[返回目录/back](#Contents)

<a name="ImageClassification"></a> 

# 图像分类 / Image Classification



[返回目录/back](#Contents)

<a name="ObjectDetection"></a> 

## 目标检测/Object Detection



[返回目录/back](#Contents)



<a name="ObjectTracking"></a> 

## 目标跟踪/Object Tracking



# 3D Object Tracking



[返回目录/back](#Contents)

<a name="TrajectoryPrediction"></a> 

## 轨迹预测/Trajectory Prediction

**IPCC-TP: Utilizing Incremental Pearson Correlation Coefficient for Joint Multi-Agent Trajectory Prediction**

- 论文/Paper: http://arxiv.org/pdf/2303.00575
- 代码/Code: None

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

[返回目录/back](#Contents)

<a name="DepthEstimation"></a>

## 深度估计/Depth Estimation

**Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation**

- 论文/Paper: https://arxiv.org/abs/2211.13202
- 代码/Code:https://github.com/noahzn/Lite-Mono

[返回目录/back](#Contents)



<a name="SuperResolution"></a>

## 超分辨率/Super Resolution

**OPE-SR: Orthogonal Position Encoding for Designing a Parameter-free Upsampling Module in Arbitrary-scale Image Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2303.01091
- 代码/Code: None

[返回目录/back](#Contents)

<a name="FaceRecognition"></a>

# 人脸识别/Face Recognition



[返回目录/back](#Contents)

<a name="FaceDetection"></a>

# 人脸检测/Face Detection



[返回目录/back](#Contents)

<a name="FaceAnti-Spoofing"></a>

# 人脸活体检测/Face Anti-Spoofing



[返回目录/back](#Contents)

<a name="FaceReconstruction"></a>

## 人脸重建/Face Reconstruction

**ProxyFormer: Proxy Alignment Assisted Point Cloud Completion with Missing Part Sensitive Transformer**

- 论文/Paper: http://arxiv.org/pdf/2302.14435
- 代码/Code: https://github.com/I2-Multimedia-Lab/ProxyFormer.

[返回目录/back](#Contents)

<a name="TalkingFace"></a>

## Talking Face

**SadTalker： Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation**

- 论文/Paper: https://arxiv.org/abs/2211.12194
- 代码/Code: https://github.com/Winfredy/SadTalker

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