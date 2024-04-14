# CVPR2024-Papers-with-Code-Demo

 :star_and_crescent:**添加微信: nvshenj125, 备注方向，进交流学习群**


欢迎关注公众号：AI算法与图像处理

:star2: [CVPR 2024](https://cvpr.thecvf.com/Conferences/2024) 持续更新最新论文/paper和相应的开源代码/code！



B站demo：https://space.bilibili.com/288489574

> :hand: ​注：欢迎各位大佬提交issue，分享CVPR 2022论文/paper和开源项目！共同完善这个项目
>
> 往年顶会论文汇总：
>
> [CVPR2021](https://github.com/DWCTOD/CVPR2023-Papers-with-Code-Demo/blob/main/CVPR2021.md)
>
> [CVPR2022](https://github.com/DWCTOD/CVPR2023-Papers-with-Code-Demo/blob/main/CVPR2022.md)
>
> [CVPR2023](https://github.com/DWCTOD/CVPR2024-Papers-with-Code-Demo/blob/main/CVPR2023.md)
>
> [ICCV2021](https://github.com/DWCTOD/ICCV2021-Papers-with-Code-Demo)
>
> [ECCV2022](https://github.com/DWCTOD/ECCV2022-Papers-with-Code-Demo)

### **:fireworks: 欢迎进群** | Welcome

CVPR 2024 论文/paper交流群已成立！已经收录的同学，可以添加微信：**nvshenj125**，请备注：**CVPR+姓名+学校/公司名称**！一定要根据格式申请，可以拉你进群。

<a name="Contents"></a>



### :hammer: **目录 |Table of Contents（点击直接跳转）**

<details open>
<summary> 目录（右侧点击可折叠）</summary>

- [Backbone](#Backbone)
- [数据集/Dataset](#Dataset)
- [Diffusion Model](#DiffusionModel)
- [Text-to-Image](#T2I)
- [NAS](#NAS)
- [NeRF](#NeRF)
- [Knowledge Distillation](#KnowledgeDistillation)
- [多模态 / Multimodal ](#Multimodal)
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
- [姿态估计/Pose Estimation](#HumanPoseEstimation)
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
- [图像去模糊/Image Deblur](#ImageDeblur)
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



[返回目录/back](#Contents)

<a name="Dataset"></a> 

## 数据集/Dataset

**HoloVIC: Large-scale Dataset and Benchmark for Multi-Sensor Holographic Intersection and Vehicle-Infrastructure Cooperative**

- 论文/Paper: http://arxiv.org/pdf/2403.02640
- 代码/Code: None



[返回目录/back](#Contents)

<a name="DiffusionModel"></a> 

# Diffusion Model

**Balancing Act: Distribution-Guided Debiasing in Diffusion Models**

- 论文/Paper: http://arxiv.org/pdf/2402.18206
- 代码/Code: None

**DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models**

- 论文/Paper: http://arxiv.org/pdf/2402.19481
- 代码/Code: https://github.com/mit-han-lab/distrifuser

**DiffAssemble: A Unified Graph-Diffusion Model for 2D and 3D Reassembly**

- 论文/Paper: http://arxiv.org/pdf/2402.19302
- 代码/Code: https://github.com/iit-pavis/diffassemble

**Diff-Plugin: Revitalizing Details for Diffusion-based Low-level Tasks**

- 论文/Paper: http://arxiv.org/pdf/2403.00644
- 代码/Code: None

**Few-shot Learner Parameterization by Diffusion Time-steps**

- 论文/Paper: http://arxiv.org/pdf/2403.02649
- 代码/Code: https://github.com/yue-zhongqi/tif

**MedM2G: Unifying Medical Multi-Modal Generation via Cross-Guided Diffusion with Visual Invariant**

- 论文/Paper: http://arxiv.org/pdf/2403.04290
- 代码/Code: None

**DEADiff: An Efficient Stylization Diffusion Model with Disentangled Representations**

- 论文/Paper: https://arxiv.org/abs/2403.06951
- 代码/Code: https://github.com/Tianhao-Qi/DEADiff_code

**Face2Diffusion for Fast and Editable Face Personalization**

- 论文/Paper: http://arxiv.org/pdf/2403.05094
- 代码/Code: https://github.com/mapooon/Face2Diffusion

[返回目录/back](#Contents)

<a name="T2I"></a> 

## Text-to-Image

**RealCustom: Narrowing Real Text Word for Real-Time Open-Domain Text-to-Image Customization**

- 论文/Paper: http://arxiv.org/pdf/2403.00483
- 代码/Code: None

**NoiseCollage: A Layout-Aware Text-to-Image Diffusion Model Based on Noise Cropping and Merging**

- 论文/Paper: http://arxiv.org/pdf/2403.03485
- 代码/Code: https://github.com/univ-esuty/noisecollage

**Discriminative Probing and Tuning for Text-to-Image Generation**

- 论文/Paper: http://arxiv.org/pdf/2403.04321
- 代码/Code: None

**Towards Effective Usage of Human-Centric Priors in Diffusion Models for Text-based Human Image Generation**

- 论文/Paper: http://arxiv.org/pdf/2403.05239
- 代码/Code: None

[返回目录/back](#Contents)

<a name="NAS"></a> 

## NAS



[返回目录/back](#Contents)

<a name="NeRF"></a> 

# NeRF

**GSNeRF: Generalizable Semantic Neural Radiance Fields with Enhanced 3D Scene Understanding**

- 论文/Paper: http://arxiv.org/pdf/2403.03608
- 代码/Code: None

[返回目录/back](#Contents)

<a name="KnowledgeDistillation"></a> 

## Knowledge Distillation

**PromptKD: Unsupervised Prompt Distillation for Vision-Language Models**

- 论文/Paper: http://arxiv.org/pdf/2403.02781
- 代码/Code: https://github.com/zhengli97/PromptKD

**Logit Standardization in Knowledge Distillation**

- 论文/Paper: https://arxiv.org/abs/2403.01427
- 代码/Code: https://github.com/sunshangquan/logit-standardization-KD

**RadarDistill: Boosting Radar-based Object Detection Performance via Knowledge Distillation from LiDAR Features**

- 论文/Paper: http://arxiv.org/pdf/2403.05061
- 代码/Code: None

[返回目录/back](#Contents)

<a name="Multimodal"></a> 

## 多模态 / Multimodal

**MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception**

- 论文/Paper: https://arxiv.org/abs/2312.07472
- 代码/Code: https://github.com/IranQin/MP5
- 主页/Website：https://iranqin.github.io/MP5.github.io/

**Polos: Multimodal Metric Learning from Human Feedback for Image Captioning**

- 论文/Paper: http://arxiv.org/pdf/2402.18091
- 代码/Code: None

**MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer**

- 论文/Paper: http://arxiv.org/pdf/2403.02991
- 代码/Code: None

**Learning to Rematch Mismatched Pairs for Robust Cross-Modal Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2403.05105
- 代码/Code: https://github.com/hhc1997/L2RM

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

**UniMODE: Unified Monocular 3D Object Detection**

- 论文/Paper: http://arxiv.org/pdf/2402.18573
- 代码/Code: None

**CN-RMA: Combined Network with Ray Marching Aggregation for 3D Indoors Object Detection from Multi-view Images**

- 论文/Paper: http://arxiv.org/pdf/2403.04198
- 代码/Code: https://github.com/SerCharles/CN-RMA

**Memory-based Adapters for Online 3D Scene Perception**

- 论文/Paper: https://arxiv.org/abs/2403.06974
- 代码/Code:https://github.com/xuxw98/Online3D

 **Salience DETR: Enhancing Detection Transformer with Hierarchical Salience Filtering Refinement**

- 论文/Paper: https://arxiv.org/abs/2403.16131

- 代码/Code:https://github.com/xiuqhou/Salience-DETR

[返回目录/back](#Contents)



<a name="ObjectTracking"></a> 

# 目标跟踪/Object Tracking

**DeconfuseTrack:Dealing with Confusion for Multi-Object Tracking**

- 论文/Paper: http://arxiv.org/pdf/2403.02767
- 代码/Code: None

**Delving into the Trajectory Long-tail Distribution for Muti-object Tracking**

- 论文/Paper: http://arxiv.org/pdf/2403.04700
- 代码/Code: https://github.com/chen-si-jia/Trajectory-Long-tail-Distribution-for-MOT

[返回目录/back](#Contents)

# 3D Object Tracking



[返回目录/back](#Contents)

<a name="TrajectoryPrediction"></a> 

## 轨迹预测/Trajectory Prediction



[返回目录/back](#Contents)

<a name="Segmentation"></a> 

## 语义分割/Segmentation

**PEM: Prototype-based Efficient MaskFormer for Image Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2402.19422
- 代码/Code: https://github.com/niccolocavagnero/pem
- 

[返回目录/back](#Contents)

<a name="WSSS"></a>

## 弱监督语义分割/Weakly Supervised Semantic Segmentation



[返回目录/back](#Contents)

<a name="MedicalImageSegmentation"></a>

# 医学图像/Medical Image

**Modality-Agnostic Structural Image Representation Learning for Deformable Multi-Modality Medical Image Registration**

- 论文/Paper: http://arxiv.org/pdf/2402.18933
- 代码/Code: None

[返回目录/back](#Contents)

<a name="VideoObjectSegmentation"></a>

# 视频目标分割/Video Object Segmentation

**Depth-aware Test-Time Training for Zero-shot Video Object Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2403.04258
- 代码/Code: None

[返回目录/back](#Contents)

<a name="InteractiveVideoObjectSegmentation"></a>

# 交互式视频目标分割/Interactive Video Object Segmentation



[返回目录/back](#Contents)

<a name="VisualTransformer"></a>

# Visual Transformer

**Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery**

- 论文/Paper: http://arxiv.org/pdf/2403.05419
- 代码/Code: https://github.com/techmn/satmae_pp

[返回目录/back](#Contents)

<a name="DepthEstimation"></a>

## 深度估计/Depth Estimation

**Representations for Recognition and Retrieval**

- 论文/Paper: https://arxiv.org/pdf/2403.07535.pdf
- 代码/Code: https://github.com/Junda24/AFNet

[返回目录/back](#Contents)

<a name="Retrieval"></a>

# 图像、视频检索 / Image Retrieval/Video retrieval

**Dual Pose-invariant Embeddings: Learning Category and Object-specific Discriminative Representations for Recognition and Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2403.00272
- 代码/Code: None

**Learning to Rematch Mismatched Pairs for Robust Cross-Modal Retrieval**

- 论文/Paper: http://arxiv.org/pdf/2403.05105
- 代码/Code: https://github.com/hhc1997/L2RM

[返回目录/back](#Contents)

<a name="SuperResolution"></a>

## 超分辨率/Super Resolution

**SeD: Semantic-Aware Discriminator for Image Super-Resolution**

- 论文/Paper: http://arxiv.org/pdf/2402.19387
- 代码/Code: None

**Training Generative Image Super-Resolution Models by Wavelet-Domain Losses Enables Better Control of Artifacts**

- 论文/Paper: http://arxiv.org/pdf/2402.19215
- 代码/Code: https://github.com/mandalinadagi/wgsr

**CAMixerSR: Only Details Need More "Attention"**

- 论文/Paper: http://arxiv.org/pdf/2402.19289
- 代码/Code: https://github.com/icandle/camixersr

**Low-Res Leads the Way: Improving Generalization for Super-Resolution by Self-Supervised Learning**

- 论文/Paper: http://arxiv.org/pdf/2403.02601
- 代码/Code: None

[返回目录/back](#Contents)

<a name="ImageDenoising"></a>

## 图像去噪/Image Denoising - 1 篇 



[返回目录/back](#Contents)

<a name="ImageEditing"></a>

# 图像编辑/Image Editing

**Doubly Abductive Counterfactual Inference for Text-based Image Editing**

- 论文/Paper: http://arxiv.org/pdf/2403.02981
- 代码/Code: https://github.com/xuesong39/DAC

[返回目录/back](#Contents)

<a name="ImageCompression"></a>

# 图像压缩/Image Compression



[返回目录/back](#Contents)

<a name="ImageDeblur"></a>

## 图像去模糊/Image Deblur

**A Unified Framework for Microscopy Defocus Deblur with Multi-Pyramid Transformer and Contrastive Learning**

- 论文/Paper: http://arxiv.org/pdf/2403.02611
- 代码/Code: https://github.com/PieceZhang/MPT-CataBlur

[返回目录/back](#Contents)

<a name="AutonomousDriving"></a>

## 自动驾驶 / Autonomous Driving

**Abductive Ego-View Accident Video Understanding for Safe Driving Perception**

- 论文/Paper: http://arxiv.org/pdf/2403.00436
- 代码/Code: None

返回目录/back

<a name="FaceRecognition"></a>

# 人脸识别/Face Recognition



[返回目录/back](#Contents)

<a name="FaceDetection"></a>

# 人脸检测/Face Detection



[返回目录/back](#Contents)

<a name="FaceAnti-Spoofing"></a>

# 人脸活体检测/Face Anti-Spoofing

**Suppress and Rebalance: Towards Generalized Multi-Modal Face Anti-Spoofing**

- 论文/Paper: http://arxiv.org/pdf/2402.19298
- 代码/Code: https://github.com/omggggg/mmdg

[返回目录/back](#Contents)

<a name="FaceReconstruction"></a>

## 人脸重建/Face Reconstruction



[返回目录/back](#Contents)

<a name="VideoActionDetection"></a>

# 视频动作检测/Video Action Detection





[返回目录/back](#Contents)

<a name="SignLanguageTranslation"></a>

# 手语翻译/Sign Language Translation



[返回目录/back](#Contents)

<a name="PersonRe-identification"></a>

# 行人重识别/Person Re-identification





[返回目录/back](#Contents)

<a name="TalkingFace"></a>

# Talking Face



[返回目录/back](#Contents)

<a name="HumanPoseEstimation"></a>

# 姿态估计/Pose Estimation

**FAR: Flexible, Accurate and Robust 6DoF Relative Camera Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2403.03221
- 代码/Code: None

**Single-to-Dual-View Adaptation for Egocentric 3D Hand Pose Estimation**

- 论文/Paper: http://arxiv.org/pdf/2403.04381
- 代码/Code: https://github.com/MickeyLLG/S2DHand

**Hourglass Tokenizer for Efficient Transformer-Based 3D Human Pose Estimation**

- 论文/Paper: https://arxiv.org/pdf/2311.12028.pdf
- 代码/Code: https://github.com/NationalGAILab/HoT

[返回目录/back](#Contents)



<a name="GAN"></a>

# GAN



[返回目录/back](#Contents)

<a name="AgeEstimation"></a>

# 人脸年龄估计/Age Estimation



[返回目录/back](#Contents)

<a name="FacialExpressionRecognition"></a>

# 人脸表情识别/Facial Expression Recognition



[返回目录/back](#Contents)

<a name="HandPoseEstimation"></a>

## 手势姿态估计（重建）/Hand Pose Estimation( Hand Mesh Recovery)



[返回目录/back](#Contents)

<a name="3DReconstruction"></a>

## 3D Reconstruction

**UFORecon: Generalizable Sparse-View Surface Reconstruction from Arbitrary and UnFavOrable Data Sets**

- 论文/Paper: http://arxiv.org/pdf/2403.05086
- 代码/Code: https://github.com/Youngju-Na/UFORecon

**DITTO: Dual and Integrated Latent Topologies for Implicit 3D Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2403.05005
- 代码/Code: None



[返回目录/back](#Contents)

<a name="FrameInterpolation"></a>

## 视频插帧/Frame Interpolation



[返回目录/back](#Contents)

<a name="3DPointCloud"></a>

## 3D点云/3D point cloud

**Rethinking Few-shot 3D Point Cloud Semantic Segmentation**

- 论文/Paper: http://arxiv.org/pdf/2403.00592
- 代码/Code: https://github.com/ZhaochongAn/COSeg

**Extend Your Own Correspondences: Unsupervised Distant Point Cloud Registration by Progressive Distance Extension**

- 论文/Paper: http://arxiv.org/pdf/2403.03532
- 代码/Code: https://github.com/liuquan98/eyoc

**Hide in Thicket: Generating Imperceptible and Rational Adversarial Perturbations on 3D Point Clouds**

- 论文/Paper: http://arxiv.org/pdf/2403.05247
- 代码/Code: https://github.com/TRLou/HiT-ADV

[返回目录/back](#Contents)

<a name="AnomalyDetection"></a>

# Anomaly Detection



[返回目录/back](#Contents)

<a name="Other"></a>

## 其他/Other

**DisCo: Disentangled Control for Realistic Human Dance Generation**

- 论文/Paper: https://arxiv.org/abs/2307.00040
- 代码/Code: https://github.com/Wangt-CN/DisCo

**Gradient Reweighting: Towards Imbalanced Class-Incremental Learning**

- 论文/Paper: http://arxiv.org/pdf/2402.18528
- 代码/Code: None

**TAMM: TriAdapter Multi-Modal Learning for 3D Shape Understanding**

- 论文/Paper: http://arxiv.org/pdf/2402.18490
- 代码/Code: None

**Attention-Propagation Network for Egocentric Heatmap to 3D Pose Lifting**

- 论文/Paper: http://arxiv.org/pdf/2402.18330
- 代码/Code: https://github.com/tho-kn/egotap

**Attentive Illumination Decomposition Model for Multi-Illuminant White Balancing**

- 论文/Paper: http://arxiv.org/pdf/2402.18277
- 代码/Code: None

**Misalignment-Robust Frequency Distribution Loss for Image Transformation**

- 论文/Paper: http://arxiv.org/pdf/2402.18192
- 代码/Code: https://github.com/eezkni/FDL

**3DSFLabelling: Boosting 3D Scene Flow Estimation by Pseudo Auto-labelling**

- 论文/Paper: http://arxiv.org/pdf/2402.18146
- 代码/Code: https://github.com/jiangchaokang/3dsflabelling

**OccTransformer: Improving BEVFormer for 3D camera-only occupancy prediction**

- 论文/Paper: http://arxiv.org/pdf/2402.18140
- 代码/Code: None

**UniVS: Unified and Universal Video Segmentation with Prompts as Queries**

- 论文/Paper: http://arxiv.org/pdf/2402.18115
- 代码/Code: https://github.com/minghanli/univs

**Coarse-to-Fine Latent Diffusion for Pose-Guided Person Image Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2402.18078
- 代码/Code: https://github.com/YanzuoLu/CFLD

**Boosting Neural Representations for Videos with a Conditional Decoder**

- 论文/Paper: http://arxiv.org/pdf/2402.18152
- 代码/Code: None

**Classes Are Not Equal: An Empirical Study on Image Recognition Fairness**

- 论文/Paper: http://arxiv.org/pdf/2402.18133
- 代码/Code: None

**QN-Mixer: A Quasi-Newton MLP-Mixer Model for Sparse-View CT Reconstruction**

- 论文/Paper: http://arxiv.org/pdf/2402.17951
- 代码/Code: None

**Panda-70M: Captioning 70M Videos with Multiple Cross-Modality Teachers**

- 论文/Paper: http://arxiv.org/pdf/2402.19479
- 代码/Code: None

**SeMoLi: What Moves Together Belongs Together**

- 论文/Paper: http://arxiv.org/pdf/2402.19463
- 代码/Code: None

**Generalizable Whole Slide Image Classification with Fine-Grained Visual-Semantic Interaction**

- 论文/Paper: http://arxiv.org/pdf/2402.19326
- 代码/Code: None

**CricaVPR: Cross-image Correlation-aware Representation Learning for Visual Place Recognition**

- 论文/Paper: http://arxiv.org/pdf/2402.19231
- 代码/Code: https://github.com/lu-feng/cricavpr

**MemoNav: Working Memory Model for Visual Navigation**

- 论文/Paper: http://arxiv.org/pdf/2402.19161
- 代码/Code: None

**VideoMAC: Video Masked Autoencoders Meet ConvNets**

- 论文/Paper: http://arxiv.org/pdf/2402.19082
- 代码/Code: https://github.com/nust-machine-intelligence-laboratory/videomac

**Theoretically Achieving Continuous Representation of Oriented Bounding Boxes**

- 论文/Paper: http://arxiv.org/pdf/2402.18975
- 代码/Code: https://github.com/Jittor/JDet

**OHTA: One-shot Hand Avatar via Data-driven Implicit Priors**

- 论文/Paper: http://arxiv.org/pdf/2402.18969
- 代码/Code: None

**WWW: A Unified Framework for Explaining What, Where and Why of Neural Networks by Interpretation of Neuron Concepts**

- 论文/Paper: http://arxiv.org/pdf/2402.18956
- 代码/Code: None

**Spectral Meets Spatial: Harmonising 3D Shape Matching and Interpolation**

- 论文/Paper: http://arxiv.org/pdf/2402.18920
- 代码/Code: None

**SwitchLight: Co-design of Physics-driven Architecture and Pre-training Framework for Human Portrait Relighting**

- 论文/Paper: http://arxiv.org/pdf/2402.18848
- 代码/Code: None

**ViewFusion: Towards Multi-View Consistency via Interpolated Denoising**

- 论文/Paper: http://arxiv.org/pdf/2402.18842
- 代码/Code: None

**OpticalDR: A Deep Optical Imaging Model for Privacy-Protective Depression Recognition**

- 论文/Paper: http://arxiv.org/pdf/2402.18786
- 代码/Code: None

**NARUTO: Neural Active Reconstruction from Uncertain Target Observations**

- 论文/Paper: http://arxiv.org/pdf/2402.18771
- 代码/Code: None

**Towards Generalizable Tumor Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2402.19470
- 代码/Code: None

**Rethinking Multi-domain Generalization with A General Learning Objective**

- 论文/Paper: http://arxiv.org/pdf/2402.18853
- 代码/Code: None

**Rethinking Inductive Biases for Surface Normal Estimation**

- 论文/Paper: http://arxiv.org/pdf/2403.00712
- 代码/Code: https://github.com/baegwangbin/DSINE

**SURE: SUrvey REcipes for building reliable and robust deep networks**

- 论文/Paper: http://arxiv.org/pdf/2403.00543
- 代码/Code: https://github.com/YutingLi0606/SURE

**Selective-Stereo: Adaptive Frequency Information Selection for Stereo Matching**

- 论文/Paper: http://arxiv.org/pdf/2403.00486
- 代码/Code: https://github.com/Windsrain/Selective-Stereo.

**Deformable One-shot Face Stylization via DINO Semantic Guidance**

- 论文/Paper: http://arxiv.org/pdf/2403.00459
- 代码/Code: https://github.com/zichongc/DoesFS

**CustomListener: Text-guided Responsive Interaction for User-friendly Listening Head Generation**

- 论文/Paper: http://arxiv.org/pdf/2403.00274
- 代码/Code: None

**NRDF: Neural Riemannian Distance Fields for Learning Articulated Pose Priors**

- 论文/Paper: http://arxiv.org/pdf/2403.03122
- 代码/Code: None

**Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning of Instructional Videos**

- 论文/Paper: http://arxiv.org/pdf/2403.02782
- 代码/Code: None

**HUNTER: Unsupervised Human-centric 3D Detection via Transferring Knowledge from Synthetic Instances to Real Scenes**

- 论文/Paper: http://arxiv.org/pdf/2403.02769
- 代码/Code: None

**Learning Group Activity Features Through Person Attribute Prediction**

- 论文/Paper: http://arxiv.org/pdf/2403.02753
- 代码/Code: https://github.com/chihina/GAFL-CVPR2024.

**Interactive Continual Learning: Fast and Slow Thinking**

- 论文/Paper: http://arxiv.org/pdf/2403.02628
- 代码/Code: None

**NRDF: Neural Riemannian Distance Fields for Learning Articulated Pose Priors**

- 论文/Paper: http://arxiv.org/pdf/2403.03122
- 代码/Code: None

**Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning of Instructional Videos**

- 论文/Paper: http://arxiv.org/pdf/2403.02782
- 代码/Code: None

**HUNTER: Unsupervised Human-centric 3D Detection via Transferring Knowledge from Synthetic Instances to Real Scenes**

- 论文/Paper: http://arxiv.org/pdf/2403.02769
- 代码/Code: None

**Learning Group Activity Features Through Person Attribute Prediction**

- 论文/Paper: http://arxiv.org/pdf/2403.02753
- 代码/Code: https://github.com/chihina/GAFL-CVPR2024.

**Interactive Continual Learning: Fast and Slow Thinking**

- 论文/Paper: http://arxiv.org/pdf/2403.02628
- 代码/Code: None

**Hierarchical Diffusion Policy for Kinematics-Aware Multi-Task Robotic Manipulation**

- 论文/Paper: http://arxiv.org/pdf/2403.03890
- 代码/Code: None

**DART: Implicit Doppler Tomography for Radar Novel View Synthesis**

- 论文/Paper: http://arxiv.org/pdf/2403.03896
- 代码/Code: None

**MeaCap: Memory-Augmented Zero-shot Image Captioning**

- 论文/Paper: http://arxiv.org/pdf/2403.03715
- 代码/Code: https://github.com/joeyz0z/MeaCap

**HMD-Poser: On-Device Real-time Human Motion Tracking from Scalable Sparse Observations**

- 论文/Paper: http://arxiv.org/pdf/2403.03561
- 代码/Code: None

**Continual Segmentation with Disentangled Objectness Learning and Class Recognition**

- 论文/Paper: http://arxiv.org/pdf/2403.03477
- 代码/Code: https://github.com/jordangong/CoMasTRe

**HDRFlow: Real-Time HDR Video Reconstruction with Large Motions**

- 论文/Paper: http://arxiv.org/pdf/2403.03447
- 代码/Code: None

**LEAD: Learning Decomposition for Source-free Universal Domain Adaptation**

- 论文/Paper: http://arxiv.org/pdf/2403.03421
- 代码/Code: https://github.com/ispc-lab/lead

**F$^3$Loc: Fusion and Filtering for Floorplan Localization**

- 论文/Paper: http://arxiv.org/pdf/2403.03370
- 代码/Code: None

**Enhancing Vision-Language Pre-training with Rich Supervisions**

- 论文/Paper: http://arxiv.org/pdf/2403.03346
- 代码/Code: None

**Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed**

- 论文/Paper: http://arxiv.org/pdf/2403.04765
- 代码/Code: None

**Discriminative Sample-Guided and Parameter-Efficient Feature Space Adaptation for Cross-Domain Few-Shot Learning**

- 论文/Paper: http://arxiv.org/pdf/2403.04492
- 代码/Code: https://github.com/rashindrie/dipa

**Learning to Remove Wrinkled Transparent Film with Polarized Prior**

- 论文/Paper: http://arxiv.org/pdf/2403.04368
- 代码/Code: https://github.com/jqtangust/filmremoval

**LORS: Low-rank Residual Structure for Parameter-Efficient Network Stacking**

- 论文/Paper: http://arxiv.org/pdf/2403.04303
- 代码/Code: None

**Active Generalized Category Discovery**

- 论文/Paper: http://arxiv.org/pdf/2403.04272
- 代码/Code: https://github.com/mashijie1028/activegcd

**MAP: MAsk-Pruning for Source-Free Model Intellectual Property Protection**

- 论文/Paper: http://arxiv.org/pdf/2403.04149
- 代码/Code: https://github.com/ispc-lab/map

**A Study of Dropout-Induced Modality Bias on Robustness to Missing Video Frames for Audio-Visual Speech Recognition**

- 论文/Paper: http://arxiv.org/pdf/2403.04245
- 代码/Code: https://github.com/dalision/modalbiasavsr

**Seamless Human Motion Composition with Blended Positional Encodings**

- 论文/Paper: https://arxiv.org/abs/2402.15509
- 代码/Code:https://github.com/BarqueroGerman/FlowMDM

**DiffusionLight: Light Probes for Free by Painting a Chrome Ball**

- 论文/Paper: https://arxiv.org/abs/2312.09168
- 代码/Code:https://github.com/DiffusionLight/DiffusionLight

**SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting**

- 论文/Paper: http://arxiv.org/pdf/2403.05087
- 代码/Code: https://github.com/initialneil/SplattingAvatar

[返回目录/back](#Contents)

