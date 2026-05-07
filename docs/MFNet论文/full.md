# A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation

Xianping Ma , Member, IEEE, Xiaokang Zhang , Senior Member, IEEE, Man-On Pun , Senior Member, IEEE, and Bo Huang

Abstract— Multimodal remote sensing data, acquired from diverse sensors, offer a comprehensive and integrated perspective of the Earth’s surface. Leveraging multimodal fusion techniques, semantic segmentation enables detailed and accurate analysis of geographic scenes, surpassing single-modality approaches. Building on advancements in vision foundation models, particularly the segment anything model (SAM), this study proposes a unified framework incorporating a novel multimodal fine-tuning network (MFNet) for remote sensing semantic segmentation. The proposed framework is designed to seamlessly integrate with various fine-tuning mechanisms, demonstrated through the inclusion of Adapter and low-rank adaptation (LoRA) as representative examples. This extensibility ensures the framework’s adaptability to other emerging fine-tuning strategies, allowing models to retain SAM’s general knowledge while effectively leveraging multimodal data. Additionally, a pyramid-based deep fusion module (DFM) is introduced to integrate high-level geographic features across multiple scales, enhancing feature representation prior to decoding. This work also highlights SAM’s robust generalization capabilities with digital surface model (DSM) data, a novel application. Extensive experiments on three benchmark multimodal remote sensing datasets, ISPRS Vaihingen, ISPRS Potsdam, and MMHunan, demonstrate that the proposed MFNet significantly outperforms existing methods in multimodal semantic segmentation, setting a new standard in the field while offering a versatile foundation for future research and applications. The source code for this work is accessible at https://github.com/sstary/SSRS.

Index Terms— Adapter, low-rank adaptation (LoRA), multimodal fine-tuning, remote sensing, segment anything model (SAM), semantic segmentation.

# I. INTRODUCTION

MULTIMODAL remote sensing semantic segmentationinvolves the process of classifying each pixel in ground images using data from multiple sources or modalities, such

Received 21 December 2024; revised 31 May 2025; accepted 28 June 2025. Date of publication 2 July 2025; date of current version 9 July 2025. This work was supported in part by the National Natural Science Foundation of China under Grant 42371374 and Grant 41801323, in part by China Postdoctoral Science Foundation under Grant 2020M682038, and in part by Hong Kong Research Grants Council through the General Research Fund under Grant 17617024. (Corresponding authors: Man-On Pun; Xiaokang Zhang.)

Xianping Ma and Man-On Pun are with the School of Science and Engineering, The Chinese University of Hong Kong-Shenzhen, Shenzhen 518172, China (e-mail: xianpingma@link.cuhk.edu.cn; SimonPun@cuhk.edu.cn).

Xiaokang Zhang is with the School of Information Science and Engineering, Wuhan University of Science and Technology, Wuhan 430081, China (e-mail: natezhangxk@gmail.com).

Bo Huang is with the Department of Geography, The University of Hong Kong, Hong Kong, SAR, China (e-mail: hbcuhk@gmail.com).

Digital Object Identifier 10.1109/TGRS.2025.3585238

as optical images, multispectral images, hyperspectral images, and LiDAR. By integrating diverse types of information, multimodal approaches enhance the accuracy and robustness of segmentation, particularly in complex environments [1], [2]. This technique leverages the complementary strengths of different data types to improve the identification of land objects, making it crucial for applications such as land use and land cover [3], [4], environmental monitoring [5], and disaster management [6], [7]. In recent years, deep learning technologies have greatly prompted the development of multimodal fusion methods in remote sensing.

Initially, convolutional neural networks (CNNs) were the dominant architecture, known for their ability to extract local spatial features from different modalities with the encoderdecoder framework [8], [9]. These early CNN-based methods stack multimodal data and fuse features at various stages for improved segmentation performance [10], [11]. With the introduction of the self-attention-based Transformer [12], which excels at modeling global context and long-range dependencies, many hybrid methods integrating Transformers into CNN-based methods have emerged. In particular, Vision Transformer (ViT) [13] and Swin Transformer [14] further introduced Transformer to the computer vision community, greatly improving the ability of image feature extraction. Combining CNNs for detailed feature extraction with Transformers for capturing global relationships between different data sources marked a new stage in segmentation models [15], [16], [17]. This hybrid approach enhances the ability to fuse information across modalities and scales, leading to more accurate, robust segmentation results in complex scenes. Despite their good performance, the aforementioned models were trained solely on narrowly scoped task-specific data, limiting their acquisition of general visual knowledge. In contrast, foundation models benefit from large-scale, diverse pretraining data, enabling them to develop broad visual representations that extend beyond single-task constraints.

Interestingly, the recent emergence of large foundation models offers a solution to this challenge. In particular, segment anything model (SAM) [18] is a cutting-edge segmentation model designed to tackle a wide range of segmentation tasks across diverse datasets. It is comprised of three components, including a ViT-based image encoder, a prompt encoder, and a mask decoder. Developed by Meta AI, SAM benefits from large-scale training on a vast visual corpus of natural images, enabling it to generalize effectively to unseen objects. Its

![](images/4de02b52bd59234a75ae0d4e5e187ae02a851faf6ebdb6536d1a851b3d1c9d67.jpg)  
Fig. 1. Illustration of the key challenges: traditional task-specific remote sensing models are limited in general visual knowledge. In contrast, SAM contains general knowledge from large-scale natural image corpora, but it lacks the framework to support multimodal remote sensing tasks.

versatility and robust performance position it as a valuable tool for various applications [19], [20]. However, compared to natural images, remote sensing images exhibit significant differences due to variations in sensors, resolution, spectral ranges, and so on [21], [22], [23]. For multimodal tasks, nonoptical data such as digital surface model (DSM) further increase the discrepancy. This raises a new challenge: how can the capabilities of SAM, acquired from massive natural visual corpora, be leveraged to enhance multimodal remote sensing tasks? Fig. 1 illustrates the key challenges discussed above, providing a clearer depiction of the problem addressed in this work.

Existing fine-tuning techniques, especially Adapter [24], [25], [26] and low-rank adaptation (LoRA) [27], [28], address part of this challenge. Compared to training and full finetuning, these methods fix most of the parameters and learn task-specific information with very few parameters. This approach enables parameter-efficient learning and the successful migration of large foundation models to a wider range of specific downstream tasks, even in more constrained hardware environments. The concept of Adapter was first introduced in the natural language processing community [24] as a method to fine-tune large pretrained models for specific downstream tasks. The core idea behind Adapter is to introduce a parallel, compact, and scalable adaptation module that learns task-specific knowledge during training, while the original model branch remains fixed, retaining task-agnostic knowledge. Similarly, LoRA [27] introduced trainable rank decomposition matrices to learn task-specific knowledge. This approach leverages the synergy between task-specific and taskagnostic knowledge, enabling efficient fine-tuning of the large model.

In the field of remote sensing, fine-tuning techniques have focused on adapting SAM for single-modality tasks [29], [30], [31]. For instance, CWSAM [29] and MeSAM [30] adapted SAM’s image encoder and introduced custom mask decoders for remote sensing data. However, an analysis of the SAM parameter distribution across three scales reveals that the majority of SAM’s parameters are concentrated in its image encoder, suggesting that most of its general knowledge

![](images/4f8ff677928fa784525eb925fd9139965f1ae9462237dba033ab7fa174200cef.jpg)  
Fig. 2. (a) Current multimodal fusion methods and (b) proposed unified multimodal fine-tuning framework. In our method, single-modal encoders are used to learn general knowledge and frozen, enabling multimodal tasks to benefit from the vision foundational model.

is encapsulated within this component. Therefore, we consider it unnecessary to modify SAM’s prompt encoder and mask decoder for remote sensing-specific tasks, as such modifications increase model adaptation complexity while hindering seamless integration with existing segmentation models.

To address the aforementioned challenges with fine-tuning framework, we present a unified multimodal fine-tuning framework for multimodal data feature learning and fusion. Specifically, as illustrated in Fig. 2(a), current multimodal fusion methods typically assign separate encoders for each modality and perform feature fusion through a multimodal fusion module [32], [33], [34]. When additional modalities are introduced, more encoders need to be added accordingly. During training, all encoders and the fusion module need to be trained simultaneously. In contrast, the unified multimodal fine-tuning framework proposed in this work revolutionizes this structure. As depicted in Fig. 2(b), it capitalizes on the general knowledge embedded in the foundational model and facilitates modality learning and fusion through the multimodal fine-tuning module. During training, the encoders remain fixed, requiring optimization only for the multimodal fine-tuning module. This approach not only fully leverages the vision foundational model but also offers seamless scalability to incorporate additional modalities.

After that, to thoroughly investigate the impact of various fine-tuning mechanisms in remote sensing, we introduce two multimodal fine-tuning techniques, namely Multimodal Adapter (MMAdapter) and Multimodal LoRA (MMLoRA), for multimodal fusion remote sensing tasks. It is worth noting that our multimodal fine-tuning framework represents a novel feature learning fusion strategy that operates independently of the specific fine-tuning module used. Leveraging MMAdapter and MMLoRA, we propose a novel multimodal fusion method, namely multimodal fine-tuning network (MFNet) with different image encoder architectures. Furthermore, MFNet utilizes a deep fusion module (DFM) to perform multiscale processing and fusion of the deep high-level features from SAM’s image

encoder. This is an effective solution for the complex characteristics of remote sensing data. Additionally, we employ a universal semantic segmentation decoder [35] that does not require additional task-specific design efforts, allowing for easy integration with decoders from other tasks. The fourfold contributions are summarized in the following.

1) A unified multimodal fine-tuning framework is proposed to adapt SAM for multimodal feature learning, independent of the specific design of the fine-tuning module and the number of data modalities.   
2) A scalable multimodal fusion network, namely MFNet, is proposed by capitalizing on SAM’s image encoder and the proposed multimodal fine-tuning framework to perform remote sensing semantic segmentation. This is a streamlined and flexible adaptation network that eliminates most of the redundant modules in SAM.   
3) Two most representative fine-tuning architectures from the literature, namely Adapter and LoRA, are utilized to validate the framework’s effectiveness by showing SAM’s robust generalization capability on DSM for the first time. Extensive experiments on three well-known multimodal remote sensing datasets, ISPRS Vaihingen, ISPRS Potsdam, and MMHunan, confirm that the proposed MFNet substantially outperforms existing methods in terms of semantic segmentation performance.   
4) To the best of our knowledge, as pioneers in exploring the multimodal fine-tuning approach based on SAM, this work thoroughly investigates the role of the two most widely used fine-tuning mechanisms in the field of remote sensing. It establishes a robust foundation for related studies and provides clear directions for future research.

The remainder of this article is structured as follows. Section II provides a review of related works on multimodal fusion and SAM. Section III presents the unified multimodal fine-tuning framework and MFNet, followed by a detailed description and analysis of the extensive experiments and a discussion in Section IV. Finally, the conclusion is given in Section V.

# II. RELATED WORKS

# A. Multimodal Remote Sensing Semantic Segmentation

Semantic segmentation is a critical preprocessing step in remote sensing image processing, and leveraging multimodal information often yields better results than relying on a single modality. Recently, the advent of deep learning has revolutionized the entire field of remote sensing, including semantic segmentation. Based on the classical encoder-decoder framework [8], [9], numerous multimodal fusion approaches based on CNNs and Transformers have driven significant advancements in the field [15], [16], [33], [36]. ResUNet-a [32], an early CNN-based architecture, simply stacked multimodal data into four channels. Furthermore, vFuseNet [36] introduced a two-branch encoder to separately extract multimodal features, enabling deeper multimodal fusion through element-wise operations at the feature level. Recently, the introduction of Transformers [12], [13] has further enriched multimodal networks. For instance, CMFNet [15] employed

CNNs for feature extraction and uses a Transformer structure to connect multimodal features across scales, emphasizing the importance of scale in multimodal fusion. Similarly, MFTransNet [33] used CNNs for feature extraction while enhancing the self-attention module with spatial and channel attention for better feature fusion. FTransUNet [33] presented a multilevel fusion approach to refine the fusion of shallowand deep-level remote sensing semantic features. Despite the great performance they achieved, we argue that existing models lack sufficient general knowledge, which poses a fundamental limitation to the advancement of multimodal fusion methods.

# B. SAM in Remote Sensing

The SAM [18] holds a unique position as a general-purpose image segmentation model. This vision foundation model was trained on a very large visual corpus. It endows SAM with a remarkable ability to generalize to unseen objects, making it well-suited for applications in diverse scenarios. Nowadays, SAM has already been applied across various fields, such as autonomous driving [38], medical image processing [39], and remote sensing [40], [41], [42], [43]. In remote sensing, SAMRS [40] leveraged SAM to integrate numerous existing remote sensing datasets, using a new prompt named rotated bounding box. Furthermore, some recent works have considered fine-tuning SAM for remote sensing tasks such as semantic segmentation [29], [30], [44] and change detection [45], [46].

For single-modality tasks, CWSAM [29] adapted SAM to synthetic aperture radar (SAR) images by introducing a task-specific input module and a class-wise mask decoder. MeSAM [30] incorporated an inception mixer into the SAM’s image encoder to retain high-frequency features and introduced a multiscale-connected mask decoder for optical images. SAM_MLoRA [31] employed multiple LoRA modules in parallel to enhance LoRA’s decomposition capability. For multimodal tasks, RingMo-SAM [44] introduced a prompt encoder tailored to multimodal remote sensing data, along with a category-decoupling mask decoder. It is observed that these methods focus on refining the fine-tuning mechanisms and designing task-specific prompts or mask decoders. They have preliminarily explored the generalization ability of SAM in remote sensing tasks. However, as discussed in Section I, the general knowledge in SAM is primarily centered on the image encoder. While these approaches successfully utilize SAM’s general knowledge in remote sensing, their complex architectures greatly hinder their adaptation to existing general semantic segmentation networks. Furthermore, there is currently no SAM-based multimodal approach designed for DSM data.

# III. PROPOSED METHOD

We first introduce the unified multimodal fine-tuning framework by elaborating the MMAdapter and MMLoRA. Specifically, we review the conventional single-modal finetuning strategy Adapter and the proposed MMAdapter in

![](images/5cd46fd127f38d1bb43f888ac18e39258ad8d1e7272ea9a582fd79bee3b90376.jpg)

![](images/a380406abc97cb70341b9c4d855ce0802e7a7ab7b228b350a1939f784de71978.jpg)

![](images/218acdcb6cfdd66397715fb58cfc852f959d102cfde1660ce425962e80442740.jpg)  
Fig. 3. (a) ViT block without Adapter in SAM’s image encoder, (b) ViT block equipped with standard Adapter [37], (c) ViT block endowed with the proposed MMAdapter, and (d) detailed structure of the MMAdapter. The Adapter facilitates the efficient utilization of general knowledge in specific tasks. Compared to the standard Adapter, MMAadapter is characterized by two branches of shared weights for multimodal feature extraction. The standard Adapter and the proposed MMAdapter are used to fine-tune and fuse features, respectively.

Section III-A. After that, we present another classic singlemodal fine-tuning strategy LoRA and the proposed MMLoRA in Section III-B. Building upon the proposed multimodal finetuning mechanisms, the proposed MFNet is elaborated in Section III-C. Notably, MFNet has two distinct architectures based on the selection between MMAdapter and MMLoRA. Finally, to provide a clear explanation, we use the example of two modalities for illustration.

# A. Standard Adapter and the Proposed MMAdapter

They are given as follows.

1) Standard Adapter: As discussed in Section I, the general knowledge is confined to SAM’s image encoders, specifically the ViT blocks, whose structure is shown in Fig. 3(a). In [37], the Adapter was proposed to enhance the capabilities of the ViT blocks for medical tasks through fine-tuning, as illustrated in Fig. 3(b). Instead of adjusting all parameters, the pretrained SAM parameters remain frozen, while two Adapter modules are introduced to learn task-specific knowledge. Each Adapter consists of a downprojection, a ReLU activation, and an upprojection. The downprojection compresses the input embedding to a lower dimension using a simple MLP layer, and the upprojection restores the compressed embedding to its original dimension using another MLP layer. For a specific input feature $\pmb { x } _ { i } ~ \in ~ \mathbb { R } ^ { h \times w \times c }$ , where h, w, and $c$ represent the height, width, and channels of the input feature, respectively, the Adapter’s process for generating the adapted feature can be expressed as

$$
\boldsymbol {x} _ {a} ^ {\mathrm {S A}} = \operatorname {R e L U} \left(\operatorname {L N} \left(\boldsymbol {x} _ {i}\right) \cdot \boldsymbol {W} _ {d}\right) \cdot \boldsymbol {W} _ {u} \tag {1}
$$

where $\pmb { W } _ { d } \in \mathbb { R } ^ { c \times \hat { c } }$ and $\boldsymbol { W } _ { u } \in \mathbb { R } ^ { \hat { c } \times c }$ are the downprojection and upprojection, respectively, and $\hat { c } \ll c$ is the compressed middle dimension of the Adapter. After that, both the adapted feature $x _ { a }$ and the output of the original MLP branch are fused with $\boldsymbol { x } _ { i }$ by residual connection to generate

the output feature $x _ { o }$

$$
\boldsymbol {x} _ {o} ^ {\mathrm {S A}} = \mathcal {F} (\boldsymbol {x} _ {i}) + s \cdot \boldsymbol {x} _ {a} ^ {\mathrm {S A}} + \boldsymbol {x} _ {i} \tag {2}
$$

where $\mathcal { F } ( \cdot )$ denotes the MLP operation and $s$ is a scaling factor to weight the task-specific and task-agnostic knowledge. Since the Adapter proposed in [37] was designed for single-modal data, it is referred to as the standard Adapter in the sequel.

2) Proposed MMAdapter: Next, we extend the standard Adapter to multimodal tasks. The proposed MMAdapter is a core component in the proposed multimodal finetuning framework. As illustrated in Fig. 3(c), we employ dual branches with shared weights to process multimodal information. The Adapter after the multihead attention is retained to extract features from each modality independently, while the Adapter during the MLP stage is replaced with the proposed MMAdapter. The detail of MMAdapter is presented in Fig. 3(d). While preserving the core structure of the Adapter, the MMAdapter enables modality interaction through a fusion module. Notably, this design accommodates arbitrary feature fusion strategies. To emphasize the effectiveness of the multimodal fine-tuning framework, we adopt the simplest method of element-wise addition based on two weighting factors, $\lambda _ { 1 }$ and $\lambda _ { 2 }$ . For two specific multimodal input features, $\pmb { x } _ { i } \in \mathbb { R } ^ { h \times w \times c }$ and $\mathbf { y } _ { i } \in \mathbb { R } ^ { h \times w \times c }$ , the process of generating the adapted features with MMAdapter can be described as

$$
\boldsymbol {x} _ {a} ^ {\mathrm {M M A}} = \operatorname {R e L U} \left(\ln \left(\boldsymbol {x} _ {i}\right) \cdot \boldsymbol {W} _ {d x}\right) \cdot \boldsymbol {W} _ {u x} \tag {3}
$$

$$
\mathbf {y} _ {a} ^ {\mathrm {M M A}} = \operatorname {R e L U} \left(\operatorname {L N} \left(\mathbf {y} _ {i}\right) \cdot \mathbf {W} _ {d y}\right) \cdot \mathbf {W} _ {u y} \tag {4}
$$

where $\pmb { W } _ { d x } , \pmb { W } _ { d y } \in \mathbb { R } ^ { c \times \hat { c } }$ and $\boldsymbol { W } _ { u x } , \boldsymbol { W } _ { u y } \in \mathbb { R } ^ { \hat { c } \times c }$ are the downprojections and upprojections, respectively. After that, the multimodal output features $\pmb { x } _ { o } ^ { \mathrm { M M A } }$ and ${ \mathbf { y } } _ { o } ^ { \mathrm { { M M A } } }$ are generated using $\lambda _ { 1 }$ and $\lambda _ { 2 }$ as follows:

$$
\boldsymbol {x} _ {o} ^ {\mathrm {M M A}} = \mathcal {F} (\boldsymbol {x} _ {i}) + \lambda_ {1} \cdot \boldsymbol {x} _ {a} ^ {\mathrm {M M A}} + (1 - \lambda_ {1}) \cdot \boldsymbol {y} _ {a} ^ {\mathrm {M M A}} + \boldsymbol {x} _ {i} \tag {5}
$$

$$
\mathbf {y} _ {o} ^ {\mathrm {M M A}} = \mathcal {F} \left(\mathbf {y} _ {i}\right) + \lambda_ {2} \cdot \mathbf {y} _ {a} ^ {\mathrm {M M A}} + (1 - \lambda_ {2}) \cdot \mathbf {x} _ {a} ^ {\mathrm {M M A}} + \mathbf {y} _ {i}. \tag {6}
$$

![](images/32d017882ba97fdcaf8ecafca07fd70cc890f1c0f6379c706546ec5d9dfcd552.jpg)

![](images/2ef6875aa55d03ba456669b7a4f570659ddb5808887d8475c2e9f08b5b42fa64.jpg)  
(b)   
Fig. 4. (a) Detailed structure of standard LoRA [27] and (b) proposed MMLoRA. It is observed that MMLoRA adopts the design principles of MMAdapter, which not only reduces the module’s complexity but also highlights the versatility of this strategy.

Only newly added parameters are optimized, while other parameters remain fixed during fine-tuning. Detailed annotations are provided in Fig. 3(b), while other subfigures omit these annotations for clarity and conciseness.

# B. Standard LoRA and the Proposed MMLoRA

They are given as follows.

1) Standard LoRA: Foundation models are composed of numerous dense layers, typically utilizing full-rank matrix multiplication. To adapt these pretrained models for specific tasks, LoRA [27] assumes that during the adaptation process, the updates to the weights have a lower “intrinsic rank” [47]. This mechanism can be applied to any linear layer. For a pretrained weight matrix $W _ { 0 } ~ \in ~ \mathbb { R } ^ { d \times d }$ , the update is expressed with a low-rank decomposition

$$
\boldsymbol {W} _ {0} + \Delta \boldsymbol {W} = \boldsymbol {W} _ {0} + \boldsymbol {B} \boldsymbol {A} \tag {7}
$$

where $\pmb { { B } } \in \mathbb { R } ^ { d \times r }$ , $\pmb { A } \in \mathbb { R } ^ { r \times d }$ , and the rank $r \ll d$

During training, $W _ { 0 }$ remains fixed and does not receive gradient updates, while the trainable parameters are contained in $\pmb { A }$ and $\pmb { B }$ . Given an input $\mathbf { \lambda } _ { x _ { i } }$ , the forward computation for the adapted module is represented as

$$
\boldsymbol {x} _ {o} ^ {\mathrm {S L}} = \left(\boldsymbol {W} _ {0} + \Delta \boldsymbol {W}\right) \boldsymbol {x} _ {i} = \boldsymbol {W} _ {0} \boldsymbol {x} _ {i} + \boldsymbol {x} _ {a} ^ {\mathrm {S L}}. \tag {8}
$$

The matrix $\pmb { A }$ is initialized with a random Gaussian distribution, while $\pmb { B }$ is initialized to 0, resulting in $\Delta W = 0$ at the start of training. The architecture of LoRA is depicted in Fig. 4(a). Throughout this work, the single-modal implementation of LoRA is referred to as the standard LoRA.

2) Proposed MMLoRA: Similar to MMAdapter, we extend standard LoRA to handle multimodal tasks. As shown in Fig. 4(b), a dual-branch structure with shared weights is employed to process multimodal information. This design enables the learning of task-specific knowledge both within individual modalities and across modalities, facilitated by the fusion module. Given inputs $\boldsymbol { x } _ { i }$ and ${ \bf \nabla } _ { \bf { y } } \cdot \mathrm { ~  ~ \ l ~ } \cdot \mathrm { ~  ~ \Omega ~ } ^ { \mathrm { ~ ~ } }$ , the process of generating the adapted features using MMLoRA can be described as

$$
\boldsymbol {x} _ {o} ^ {\mathrm {M M L}} = \boldsymbol {W} _ {x 0} \boldsymbol {x} _ {i} + \lambda_ {1} \cdot \boldsymbol {x} _ {a} ^ {\mathrm {M M L}} + (1 - \lambda_ {1}) \boldsymbol {y} _ {a} ^ {\mathrm {M M L}} \tag {9}
$$

$$
\boldsymbol {y} _ {o} ^ {\mathrm {M M L}} = \boldsymbol {W} _ {y 0} \boldsymbol {y} _ {i} + \lambda_ {2} \cdot \boldsymbol {y} _ {a} ^ {\mathrm {M M L}} + (1 - \lambda_ {2}) \boldsymbol {x} _ {a} ^ {\mathrm {M M L}} \tag {10}
$$

where

$$
\boldsymbol {x} _ {a} ^ {\text {M M L}} = \boldsymbol {B} _ {x} \boldsymbol {A} _ {x} \boldsymbol {x} _ {i} \tag {11}
$$

$$
\boldsymbol {y} _ {a} ^ {\mathrm {M M L}} = \boldsymbol {B} _ {y} \boldsymbol {A} _ {y} \boldsymbol {y} _ {i}. \tag {12}
$$

Finally, it is worth mentioning that the design shown in Figs. 3 and 4 can be generalized to more than two modalities in a straightforward manner.

# C. Proposed MFNet

Fig. 5 shows an overview of the proposed MFNet and two different multimodal fune-tuning strategies. The input to MFNet is first processed by SAM’s image encoder endowed with MMAdapter or MMLoRA, which is responsible for extracting and fusing multimodal remote sensing features using the multimodal fine-tuning mechanism. The output is then fed into DFM, which receives two single-scale multimodal outputs from the encoder and expands them into two sets of multiscale multimodal features using pyramid modules. These high-level abstract features are then fused by four squeeze-and-excitation (SE) fusion modules to generate a group of multiscale features. Finally, the outputs of DFM are passed to the decoder to produce segmentation prediction maps. In this section, we introduce the key components of the proposed MFNet in detail.

1) SAM’s Image Encoder: We denote the optical images and their corresponding DSM data as $\pmb { X } \in \bar { \mathbb { R } } ^ { H \times W \times 3 }$ and $\textbf { \textit { Y } } \in$ $\mathbb { R } ^ { H \times W \times 1 }$ , respectively, where $H$ and $W$ represent the height and width of the inputs, respectively. The SAM’s image encoder, employing a nonhierarchical ViT backbone, first embeds the input into a size of $\mathbb { R } ^ { h \times w \times c }$ , where $h = ( H / 1 6 )$ , $w = ( W / 1 6 )$ , and $c$ is the embedding dimension. Next, stacked ViT blocks are used to extract features whose size is maintained throughout the encoding process [48]. As illustrated in Fig. 5(a), both $X$ and Y are input into the SAM’s image encoder. It is important to note that the same SAM encoder is used for DSM data, which demonstrates that SAM can be utilized to extract features from nonimage data. The SAM’s image encoder extracts and fuses multimodal features, generating high-level abstract features $\pmb { F } _ { x } \in \mathbb { R } ^ { h \times w \times c }$ and $\pmb { F } _ { y } \in \mathbb { R } ^ { h \times w \times c }$ through the multimodal fine-tuning modules.

![](images/4c3d6e1f124dc847dceaf610b66d2a06b8bd64213ce3db93e535c311c03db1e9.jpg)  
(a)

![](images/cf8d1f76cd9c859c4cb66028433dcf93d0f96fdb46f9b78f4f2ea64eff88496d.jpg)  
(b)

![](images/3f88be0e54bdc6269193e34362ca93711e2cf117d3bb3d7b6d047f209d5eceb2.jpg)  
  
Fig. 5. (a) Overview of the proposed MFNet consisting of an SAM’s image encoder with multimodal fine-tuning, a DFM, and a general decoder. The structure of (b) ViT block with MMAdapter and (c) ViT block with MMLoRA. These constitute the two distinct architectures of MFNet.

2) ViT Block With MMLoRA: Fig. 5(b) depicts the architecture of MMAdapter within the ViT block. Conversely, MMLoRA serves as a multimodal fine-tuning method applied in parallel with linear layers. For greater clarity, the structure of the ViT block incorporating MMLoRA is presented in Fig. 5(c). In the multihead attention module, the LoRA module is applied to the $q$ and $v$ projection layers [49]. At this stage, multimodal interaction is excluded to concentrate on capturing task-specific information within each modality. In the subsequent MLP layer, the MMLoRA module is applied to the linear layers in MLP, facilitating the fusion of multimodal information.

3) DFM: Multiscale features play a critical role in semantic segmentation tasks since dense predicting requires both global information and local information. As shown in Fig. 6(a), two pyramid modules, each consisting of a set of parallel convolutions or deconvolutions, are used to generate multiscale feature maps. Starting with the plain ViT feature map at a scale of (1/16), we produce feature maps at scales of {(1/4), (1/8), (1/16), (1/32)} using convolutions with strides of $\{ ( 1 / 4 ) , ( 1 / 2 ) , 1 , 2 \}$ , where fractional strides indicate deconvolutions [48]. These simple pyramid modules generate two sets of multimodal multiscale features, denoted as $F _ { x } ^ { i }$ and $F _ { y } ^ { i }$ , where $i = \{ 1 , 2 , 3 , 4 \}$ represents the scale index. Subsequently, four SE fusion modules [33] are employed to further integrate the multimodal features. It is worth mentioning that more advanced fusion modules can yield further improved segmentation performance.

As illustrated in Fig. 6(b), the SE fusion module begins by aggregating the global information from the multimodal features. For the ith fusion module, with an input channel size of $C _ { i }$ , the squeeze-and-excitation process is performed through two convolutional operations with a kernel size of $1 \times 1$ , followed by the ReLU and sigmoid activations. The multimodal outputs are then weighted and combined element-wise, producing the enhanced fused features denoted by $F _ { \ f } ^ { i }$ . The outputs from the four SE

![](images/23edea327a9eb4137d32b1fab5da979300f7667eba2be7672b8ed606ca34d2b6.jpg)

![](images/4c924366a07acbef862a77f5afd0648cae1738e9eeade99163df150b26144758.jpg)  
(b)   
Fig. 6. (a) Schematic of the DFM. There are two pyramid modules that expand multimodal features for multiscale features before they are fused by four SE fusion modules. (b) Schematic of the SE fusion module [33]. Notably, we just employ the existing simple fusion module and do not design this structure specifically, which proves that our primary gains are derived from the multimodal fine-tuning strategy based on the vision foundation model.

fusion modules form the multiscale fusion features, denoted as $F _ { f } ^ { I - 4 }$ , which are fed into the decoder for processing. The decoder introduced in UNetformer [35] is employed in this work, which reconstructs abstract semantic information into a segmented map by focusing on both global and local information.

# IV. EXPERIMENTS AND DISCUSSION

# A. Datasets

1) Vaihingen: It contains 16 fine-resolution orthophotos, each averaging $2 5 0 0 \times 2 0 0 0$ pixels. These orthophotos consist of three channels: near infrared, red, and green (NIRRG), and come with a normalized DSM at a 9-cm ground sampling distance. The 16 orthophotos are split into a training set of 12 patches and a test set of four patches. To improve the storage and reading efficiency of large patches, a sliding window of size $2 5 6 ~ \times ~ 2 5 6$ is utilized rather than cropping the patches into smaller images in both training and test stages, which results in about 960 training images and 320 test images.

![](images/177ab86e6618777466cbf61b5d10704436c24dc3af145d9a84e0416d2b478fa7.jpg)  
Fig. 7. Here, we present (a) and (b) two samples of size $2 0 4 8 \ \times$ 2048 from Vaihingen, (c) and (d) two samples of size $2 0 4 8 \times 2 0 4 8 \mathrm { ~ f t }$ rom Potsdam (last two columns), and (e) and (f) two samples of size $2 5 6 \ \times$ 256 from MMHunan. The first row displays the orthophotos with NIRRG channels for Vaihingen, RGB channels for Potsdam, and RGB channels for MMHunan. The second and third rows present the corresponding pixel-wise depth information and ground truth labels. They show the individual and complementary characteristics of remote sensing data from different sources.

2) Potsdam: This dataset is much larger than the Vaihingen dataset, which consists of 24 high-resolution orthophotos, each with a size of $6 0 0 0 ~ \times ~ 6 0 0 0$ pixels. It includes four multispectral bands: infrared, red, green, and blue (IRRGB), along with a normalized DSM of $5 \ \mathrm { c m }$ . The last three bands (RGB) in this dataset are utilized to diversify our experiments. The 24 orthophotos are split into 18 patches for training and 6 for testing. Using the same sliding window approach, this dataset contains 10 368 training samples and 3456 test samples.

The Vaihingen and Potsdam datasets classify five main foreground categories: Building (Bui.), Tree (Tre.), Low Vegetation (Low.), Car, and Impervious Surface (Imp.). Additionally, a background class labeled as clutter contains indistinguishable debris and water surfaces. Notably, the sliding window to collect training samples moves with a smaller step size, and the overlapping areas are averaged to the reduced boundary effect during the test stage.

3) MMHunan: This dataset [50] differs significantly in spatial resolution from that of the ISPRS datasets, with a value of $1 0 \mathrm { ~ m ~ }$ . It contains 500 Sentinel-2 image patches, each of size $2 5 6 \times 2 5 6$ pixels, accompanied by corresponding digital elevation data. We selected the red, green, and blue bands to form the visible imagery. The dataset includes annotations for seven land cover types: Cropland, Forest, Grassland, Wetland, Water, Unused Land, and Built-up Area.

Visual examples from all three datasets are presented in Fig. 7. The substantial differences in both data characteristics and land cover categories greatly enhance the diversity of our experiments.

# B. Implementation Details

All experiments were conducted using PyTorch on a single NVIDIA GeForce RTX 3090 GPU with 24 GB of RAM. The stochastic gradient descent algorithm was used to train all the models under consideration with a learning rate of 0.01, a momentum of 0.9, a weight decay of 0.0005, and a batch

size of 10. The batch size is reduced to 4 when ViT-H is applied to meet the memory limit. All models were trained for a total of 50 epochs, with each epoch comprising 1000 batches. Basic data augmentation techniques, including random rotation and flipping, are applied after sample collection with the sliding window of size $2 5 6 ~ \times ~ 2 5 6$ . For MMAdapter, the downprojection ratio is set to 0.25. For MMLoRA, we follow the low-rank value in [27] as 4. More details can be found in https://github.com/sstary/SSRS

To assess the semantic segmentation performance on multimodal remote sensing data, we use overall accuracy (OA), mean $F 1$ score (mF1), and mean intersection over union (mIoU). These standard metrics enable a fair comparison between the proposed MFNet and other state-of-the-art methods. Specifically, OA evaluates both foreground classes and the background class, while mF1 and mIoU are calculated for the five foreground classes.

# C. Performance Comparison

We benchmarked the proposed MFNet against 15 stateof-the-art methods, including PSPNet [54], MAResU-Net [51], vFuseNet [36], FuseNet [10], ESANet [52], SA-GATE [55], CMGFNet [53], TransUNet [56], CMFNet [15], UNetFormer [35], MFTransNet [16], FTransUNet [33], RS3Mamba [57], FTransDeepLab [58], and MultiSenseSeg [59], most of which were specifically designed for remote sensing tasks. In our experiments, PSPNet, MAResU-Net, UNetFormer, and ${ \mathsf { R } } { \mathsf { S } } ^ { 3 }$ Mamba utilized only the optical images, to highlight the impact of DSM data and demonstrate the advantages of multimodal methods over single-modal ones. The other methods are state-of-the-art multimodal models based on different network architectures, including CNN, Transformer, and Mamba. Taking into account the three different backbones provided by SAM and the two multimodal fine-tuning architectures proposed in this work, we present six sets of experimental results for each dataset. The comparative quantitative results are presented in Tables I and II.

1) Performance Comparison on the Vaihingen Dataset: As presented in Table I, the proposed MFNet demonstrated substantial improvements in terms of OA, mF1, and mIoU metrics compared to existing segmentation methods. These results confirmed that our MFNet can effectively leverage the extensive general knowledge embedded in SAM. In particular, MFNet outperformed state-of-the-art models across four specific classes, namely Building, Tree, Low Vegetation, and Impervious surface. In terms of the overall performance, the proposed MFNet (MMAdapter) with ViT-H achieved an OA of $9 2 . 9 7 \%$ , an mF1 score of $9 1 . 7 1 \%$ , and an mIoU of $8 5 . 0 3 \%$ , reflecting gains of $0 . 2 4 \%$ , $0 . 2 9 \%$ , and $0 . 5 0 \%$ over the second-best method MultiSenseSeg, respectively. Moreover, each of the three MFNet backbone variants offers unique advantages. Even the smallest variant, ViT-B, is comparable to most methods, further validating that our multimodal fine-tuning framework can efficiently utilize general knowledge from SAM to assist with the semantic segmentation of multimodal remote sensing data. This result demonstrated the practical value of the proposed MFNet and MMAdapter or MMLoRA in guiding the introduction of foundation models,

TABLE I QUANTITATIVE RESULTS ON THE VAIHINGEN DATASET. THE BEST RESULTS ARE IN BOLD. THE SECOND-BEST RESULTS ARE UNDERLINED $( \% )$   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="6">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Bui.</td><td>Tre.</td><td>Low.</td><td>Car</td><td>Imp.</td><td>Total</td></tr><tr><td>FuseNet [10]</td><td>VGG16</td><td>96.28</td><td>90.28</td><td>78.98</td><td>81.37</td><td>91.66</td><td>90.51</td><td>87.71</td><td>78.71</td></tr><tr><td>vFuseNet [36]</td><td>VGG16</td><td>95.92</td><td>91.36</td><td>77.64</td><td>76.06</td><td>91.85</td><td>90.49</td><td>87.89</td><td>78.92</td></tr><tr><td>MAResU-Net [51]</td><td>ResNet18</td><td>94.84</td><td>89.99</td><td>79.09</td><td>85.89</td><td>92.19</td><td>90.17</td><td>88.54</td><td>79.89</td></tr><tr><td>ESANet [52]</td><td>ResNet34</td><td>95.69</td><td>90.50</td><td>77.16</td><td>85.46</td><td>91.39</td><td>90.61</td><td>88.18</td><td>79.42</td></tr><tr><td>CMGFNet [53]</td><td>ResNet34</td><td>97.75</td><td>91.60</td><td>80.03</td><td>87.28</td><td>92.35</td><td>91.72</td><td>90.00</td><td>82.26</td></tr><tr><td>PSPNet [54]</td><td>ResNet101</td><td>94.52</td><td>90.17</td><td>78.84</td><td>79.22</td><td>92.03</td><td>89.94</td><td>86.55</td><td>76.96</td></tr><tr><td>SA-GATE [55]</td><td>ResNet101</td><td>94.84</td><td>92.56</td><td>81.29</td><td>87.79</td><td>91.69</td><td>91.10</td><td>89.81</td><td>81.27</td></tr><tr><td>CMFNet [15]</td><td>VGG16</td><td>97.17</td><td>90.82</td><td>80.37</td><td>85.47</td><td>92.36</td><td>91.40</td><td>89.48</td><td>81.44</td></tr><tr><td>UNetFormer [35]</td><td>ResNet18</td><td>96.23</td><td>91.85</td><td>79.95</td><td>86.99</td><td>91.85</td><td>91.17</td><td>89.48</td><td>81.97</td></tr><tr><td>MFTransNet [16]</td><td>ResNet34</td><td>96.41</td><td>91.48</td><td>80.09</td><td>86.52</td><td>92.11</td><td>91.22</td><td>89.62</td><td>81.61</td></tr><tr><td>TransUNet [56]</td><td>R50-ViT-B</td><td>96.48</td><td>92.77</td><td>76.14</td><td>69.56</td><td>91.66</td><td>90.96</td><td>87.34</td><td>78.26</td></tr><tr><td>FTransUNet [33]</td><td>R50-ViT-B</td><td>98.20</td><td>91.94</td><td>81.49</td><td>91.27</td><td>93.01</td><td>92.40</td><td>91.21</td><td>84.23</td></tr><tr><td>RS3Mamba [57]</td><td>R18-Mamba-T</td><td>97.40</td><td>92.14</td><td>79.56</td><td>88.15</td><td>92.19</td><td>91.64</td><td>90.34</td><td>82.78</td></tr><tr><td>FTransDeepLab [58]</td><td>ResNet101</td><td>98.11</td><td>93.45</td><td>80.35</td><td>89.98</td><td>93.23</td><td>92.61</td><td>91.00</td><td>83.87</td></tr><tr><td>MultiSenseSeg [59]</td><td>Segformer-B2</td><td>97.91</td><td>93.04</td><td>81.58</td><td>89.06</td><td>93.56</td><td>92.73</td><td>91.42</td><td>84.53</td></tr><tr><td rowspan="3">MFNet (MMLoRA)</td><td>ViT-B</td><td>97.83</td><td>94.26</td><td>77.82</td><td>85.43</td><td>91.98</td><td>91.93</td><td>89.89</td><td>82.09</td></tr><tr><td>ViT-L</td><td>96.85</td><td>92.89</td><td>81.09</td><td>89.95</td><td>93.28</td><td>92.22</td><td>91.09</td><td>83.96</td></tr><tr><td>ViT-H</td><td>97.98</td><td>92.35</td><td>82.96</td><td>90.09</td><td>93.25</td><td>92.73</td><td>91.50</td><td>84.66</td></tr><tr><td rowspan="3">MFNet (MMAdapter)</td><td>ViT-B</td><td>98.73</td><td>91.41</td><td>83.09</td><td>85.63</td><td>92.91</td><td>92.62</td><td>90.60</td><td>83.24</td></tr><tr><td>ViT-L</td><td>98.84</td><td>93.17</td><td>81.16</td><td>89.23</td><td>93.39</td><td>92.93</td><td>91.51</td><td>84.72</td></tr><tr><td>ViT-H</td><td>98.38</td><td>93.94</td><td>80.70</td><td>90.47</td><td>93.59</td><td>92.97</td><td>91.71</td><td>85.03</td></tr></table>

TABLE II QUANTITATIVE RESULTS ON THE POTSDAM DATASET. THE BEST RESULTS ARE IN BOLD. THE SECOND-BEST RESULTS ARE UNDERLINED (%)   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="6">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Bui.</td><td>Tre.</td><td>Low.</td><td>Car</td><td>Imp.</td><td>Total</td></tr><tr><td>FuseNet [10]</td><td>VGG16</td><td>97.48</td><td>85.14</td><td>87.31</td><td>96.10</td><td>92.64</td><td>90.58</td><td>91.60</td><td>84.86</td></tr><tr><td>vFuseNet [36]</td><td>VGG16</td><td>97.23</td><td>84.29</td><td>89.03</td><td>95.49</td><td>91.62</td><td>90.22</td><td>91.26</td><td>84.26</td></tr><tr><td>MAResU-Net [51]</td><td>ResNet18</td><td>96.82</td><td>83.97</td><td>87.70</td><td>95.88</td><td>92.19</td><td>89.82</td><td>90.86</td><td>83.61</td></tr><tr><td>ESANet [52]</td><td>ResNet34</td><td>97.10</td><td>85.31</td><td>87.81</td><td>94.08</td><td>92.76</td><td>89.74</td><td>91.22</td><td>84.15</td></tr><tr><td>CMGFNet [53]</td><td>ResNet34</td><td>97.41</td><td>86.80</td><td>86.68</td><td>95.68</td><td>92.60</td><td>90.21</td><td>91.40</td><td>84.53</td></tr><tr><td>PSPNet [54]</td><td>ResNet101</td><td>97.03</td><td>83.13</td><td>85.67</td><td>88.81</td><td>90.91</td><td>88.67</td><td>88.92</td><td>80.36</td></tr><tr><td>SA-GATE [55]</td><td>ResNet101</td><td>96.54</td><td>81.18</td><td>85.35</td><td>96.63</td><td>90.77</td><td>87.91</td><td>90.26</td><td>82.53</td></tr><tr><td>CMFNet [15]</td><td>VGG16</td><td>97.63</td><td>87.40</td><td>88.00</td><td>95.68</td><td>92.84</td><td>91.16</td><td>92.10</td><td>85.63</td></tr><tr><td>UNetFormer [35]</td><td>ResNet18</td><td>97.69</td><td>86.47</td><td>87.93</td><td>95.91</td><td>92.27</td><td>90.65</td><td>91.71</td><td>85.05</td></tr><tr><td>MFTransNet [16]</td><td>ResNet34</td><td>97.37</td><td>85.71</td><td>86.92</td><td>96.05</td><td>92.45</td><td>89.96</td><td>91.11</td><td>84.04</td></tr><tr><td>TransUNet [56]</td><td>R50-ViT-B</td><td>96.63</td><td>82.65</td><td>89.98</td><td>93.17</td><td>91.93</td><td>90.01</td><td>90.97</td><td>83.74</td></tr><tr><td>FTransUNet [33]</td><td>R50-ViT-B</td><td>97.78</td><td>88.27</td><td>88.48</td><td>96.31</td><td>93.17</td><td>91.34</td><td>92.41</td><td>86.20</td></tr><tr><td>RS3Mamba [57]</td><td>R18-Mamba-T</td><td>97.70</td><td>86.11</td><td>89.53</td><td>96.23</td><td>91.36</td><td>90.49</td><td>91.69</td><td>85.01</td></tr><tr><td>FTransDeepLab [58]</td><td>ResNet101</td><td>97.58</td><td>85.87</td><td>90.08</td><td>96.94</td><td>92.81</td><td>90.97</td><td>92.08</td><td>85.62</td></tr><tr><td>MultiSenseSeg [59]</td><td>Segformer-B2</td><td>98.32</td><td>87.65</td><td>89.54</td><td>96.27</td><td>92.46</td><td>91.30</td><td>92.35</td><td>86.10</td></tr><tr><td rowspan="3">MFNet (MMLoRA)</td><td>ViT-B</td><td>97.60</td><td>86.45</td><td>87.87</td><td>94.39</td><td>92.44</td><td>90.57</td><td>91.48</td><td>84.61</td></tr><tr><td>ViT-L</td><td>97.59</td><td>88.57</td><td>88.34</td><td>96.35</td><td>92.68</td><td>90.99</td><td>92.13</td><td>85.71</td></tr><tr><td>ViT-H</td><td>98.19</td><td>87.30</td><td>89.89</td><td>96.27</td><td>92.80</td><td>91.43</td><td>92.49</td><td>86.34</td></tr><tr><td rowspan="3">MFNet (MMAdapter)</td><td>ViT-B</td><td>97.93</td><td>87.13</td><td>87.72</td><td>95.68</td><td>92.68</td><td>90.89</td><td>91.79</td><td>85.14</td></tr><tr><td>ViT-L</td><td>98.31</td><td>88.78</td><td>87.27</td><td>96.29</td><td>93.69</td><td>91.62</td><td>92.51</td><td>86.37</td></tr><tr><td>ViT-H</td><td>98.44</td><td>87.37</td><td>90.36</td><td>96.24</td><td>93.17</td><td>91.71</td><td>92.70</td><td>86.69</td></tr></table>

like SAM, into multimodal remote sensing tasks. On the other hand, we observed that MFNet based on MMLoRA performed less effectively than the MMAdapter-based MFNet, which will be analyzed in Section IV-F.

Fig. 8 presents a visual comparison of the results produced by various methods and the best MFNet (MMAdapter) with ViT-H. MFNet demonstrates superior performance in generating sharper and more precise boundaries for ground objects, such as trees, cars, and buildings, resulting in clearer separations. This also helps preserve the integrity of the ground objects. Overall, the visualizations produced by MFNet exhibit a cleaner and more organized appearance. We attribute these improvements primarily to SAM’s powerful feature extraction capabilities. By applying the multimodal fine-tuning mechanism, SAM’s ability to segment every natural element is effectively extended to ground objects.

2) Performance Comparison on the Potsdam Dataset: Experiments on the Potsdam dataset yielded results consistent with those from the Vaihingen dataset. As shown in

Table II, the corresponding OA, mF1, and mIoU values of MFNet (MMAdapter) with ViT-H were $9 1 . 7 1 \%$ , $9 2 . 7 0 \%$ , and $8 6 . 6 9 \%$ , which corresponds to increases of $0 . 3 7 \%$ , $0 . 2 9 \%$ , and $0 . 4 9 \%$ , respectively, over FTransUNet. Notably, significant gains were observed for Building, Tree, Low Vegetation, and Impervious Surface compared to other state-of-the-art methods. The MFNet with smaller backbones also shows great performance. This flexibility allows MFNet to balance hardware requirements and performance needs across various application scenarios.

Fig. 9 shows a visualization example from Potsdam, where we observed more defined boundaries and intact object representations. These visual improvements are consistent with the mF1 and mIoU metrics shown in Table II. Undoubtedly, it further validates the practical applicability of the proposed MFNet and MMAdapter or MMLoRA.

Additionally, it is observed that MFNet struggles to achieve optimal performance in identifying both Tree and Low Vegetation simultaneously. This challenge arises because both

![](images/25451308ea0009ce17ac99f3889a31f77551d6c7dae0c2f8effe1212dfee1580.jpg)

![](images/715649efb7a81dc35f1ba23223594672424eeceee5f3b16e19cef45077ee74d6.jpg)

![](images/780433a89266815f9434ed1221ca62bae6f9c2eac621d86b72a7cdcb81ef551f.jpg)

![](images/f596cfd9661db92217028847fd88e8d147e03d211974eb741f82a5e7078da445.jpg)

![](images/b1897aaa28f7c4ff71f31e25f5a590d79cf1908b9b03586be0eb95cb34faf2b1.jpg)

![](images/4cf6830eb74edc3df815c51e89027dc2951ec17cd849da94c0d00d721f705ef6.jpg)  
(f)

![](images/bbaff57a4f8640d32e79102537af91b87cbaf4eef6ee70867b2ae92ff0dfb412.jpg)  
(g1)

![](images/1d1adda566269947cd10dec8c7cd9008f99ca848c025522fba92d56510bf9fb4.jpg)  
(h)

![](images/4f6541ff460b1a106e34b616dfa60e9700bbfe3a20f6e77e52d76961bc8fa138.jpg)  
(i)

![](images/817cfe0779c20e8e2cfc1fcf2354be5ef84e2aae485349e714b5d8940c64f1c8.jpg)  
(j)

![](images/ef4ab4101ccddee4fba381a5e0b18fa85e95951ef2cef4960f235df27a91a4ed.jpg)

![](images/288f4fb62ddfc02d335c98bd15a56ca86e0ea670c847e494507229c38b16f42e.jpg)

![](images/5cfca96052c4b2d00de897b2282d675395ea5c28c463131c57b7dcbd7e980dd9.jpg)

![](images/109a8ef0c97bdac1c7d1c22ff8900f3a5f17cb8f1c6d8225c062e5e9b46f3a6a.jpg)

![](images/eaa34f6987f7a5903ee70eeb9ce32e9cced283c1c0571a5e414f7e2a8262c793.jpg)

![](images/30bd13390a05174b3af6139d65ce61af826d1c1bb034abcaed0b67642348f8cd.jpg)  
  
building

![](images/3a696f14d83af7ecad865c6716f103a1afb489f297c1d25cdabe54f442f084ed.jpg)

![](images/327654a322bb4b527a6f50616aee42db42327cff3cf51e977eb1f4c95f280d50.jpg)

![](images/b5ae4e96449803df732073df404278766118473a05385d92a1cc14b662dd734b.jpg)

![](images/829e6737f2d8b8ef137c0a35a75e17ddfe35ed75e25f5dd227d58062501b13c0.jpg)  
  
mparisons on the Vaihingen test set with the size of $5 1 2 \times 5 1 2$ . (a) IRRG images, (b) DSM, (c) ground truth, (d) CMFNet, (e) FTransUNet, MGFNet, (h) FTransDeepLab, (i) MultiSenseSeg, and (j) proposed MFNet. To highlight the differences, some purple boxes are added cripts $( = 1 , 2 )$ represent the serial number of the samples displayed.

TABLE III QUANTITATIVE RESULTS ON THE MMHUNAN DATASET. THE BEST RESULTS ARE IN BOLD. THE SECOND-BEST RESULTS ARE UNDERLINED (%)   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="8">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Cropland</td><td>Forest</td><td>Grassland</td><td>Wetland</td><td>Water</td><td>Unused Land</td><td>Built-up Area</td><td>Total</td></tr><tr><td>FuseNet [10]</td><td>VGG16</td><td>76.10</td><td>89.21</td><td>37.11</td><td>8.37</td><td>70.12</td><td>72.75</td><td>16.99</td><td>76.35</td><td>54.80</td><td>42.30</td></tr><tr><td>CMFNet [15]</td><td>VGG16</td><td>83.83</td><td>82.27</td><td>43.88</td><td>39.57</td><td>70.14</td><td>81.72</td><td>23.66</td><td>77.41</td><td>59.63</td><td>46.52</td></tr><tr><td>MFTransNet [16]</td><td>ResNet34</td><td>78.99</td><td>91.30</td><td>33.80</td><td>26.80</td><td>77.38</td><td>78.29</td><td>27.17</td><td>80.07</td><td>61.45</td><td>48.25</td></tr><tr><td>FTransUNet [33]</td><td>R50-ViT-B</td><td>78.75</td><td>90.54</td><td>32.13</td><td>27.51</td><td>76.64</td><td>75.59</td><td>48.63</td><td>79.47</td><td>61.95</td><td>48.78</td></tr><tr><td>CMGFNet [53]</td><td>ResNet34</td><td>82.08</td><td>87.90</td><td>37.50</td><td>26.48</td><td>79.82</td><td>74.64</td><td>41.15</td><td>79.85</td><td>62.69</td><td>49.44</td></tr><tr><td>FTTransDeepLab [58]</td><td>ResNet101</td><td>79.39</td><td>88.89</td><td>35.71</td><td>30.88</td><td>83.95</td><td>78.14</td><td>32.60</td><td>80.62</td><td>62.51</td><td>49.66</td></tr><tr><td>MultiSenseSeg [59]</td><td>Segformer-B2</td><td>78.03</td><td>90.93</td><td>40.47</td><td>38.16</td><td>80.19</td><td>81.03</td><td>38.14</td><td>80.51</td><td>63.74</td><td>50.76</td></tr><tr><td rowspan="3">MFNet (MMLoRA)</td><td>ViT-B</td><td>76.83</td><td>90.69</td><td>24.16</td><td>22.03</td><td>80.17</td><td>78.12</td><td>29.66</td><td>79.35</td><td>58.79</td><td>46.54</td></tr><tr><td>ViT-L</td><td>79.39</td><td>88.89</td><td>35.71</td><td>30.88</td><td>83.95</td><td>78.14</td><td>32.60</td><td>80.62</td><td>62.51</td><td>49.66</td></tr><tr><td>ViT-H</td><td>76.42</td><td>90.65</td><td>38.08</td><td>20.78</td><td>74.48</td><td>77.93</td><td>38.69</td><td>78.87</td><td>60.38</td><td>47.63</td></tr><tr><td rowspan="3">MFNet (MMAdapter)</td><td>ViT-B</td><td>74.81</td><td>91.47</td><td>27.59</td><td>25.00</td><td>86.47</td><td>78.92</td><td>37.46</td><td>80.68</td><td>62.10</td><td>49.20</td></tr><tr><td>ViT-L</td><td>79.19</td><td>89.52</td><td>46.07</td><td>42.23</td><td>81.15</td><td>77.58</td><td>39.65</td><td>80.93</td><td>65.33</td><td>51.82</td></tr><tr><td>ViT-H</td><td>79.66</td><td>90.06</td><td>42.61</td><td>38.81</td><td>80.31</td><td>78.92</td><td>40.04</td><td>81.07</td><td>64.13</td><td>51.08</td></tr></table>

categories are characterized by irregular boundaries. Furthermore, their similarity, combined with their staggered or overlapping distribution, makes it challenging to distinguish between them. Combining SAM with more specialized designs to tackle the identification of these challenging categories presents an interesting future direction.

3) Performance Comparison on the MMHunan Dataset: The experimental results on the MMHunan dataset are presented in Table III. MFNet (MMAdapter) with a ViT-L backbone achieved OA, mF1, and mIoU scores of $8 0 . 9 3 \%$ , $6 5 . 3 3 \%$ , and $5 1 . 8 2 \%$ , respectively, representing improvements of $0 . 4 2 \%$ , $1 . 5 9 \%$ , and $1 . 0 6 \%$ over MultiSenseSeg.

These results validate the consistent overall performance gains brought by our method. In addition, we observed an interesting phenomenon: across both fine-tuning strategies and ViT-H underperformed compared to ViT-L on this dataset. This suggests that in scenarios with small datasets, larger backbones may be more prone to overfitting, highlighting the importance of backbone selection for different remote sensing scenes. Fig. 10 shows a visualization example from MMHunan. In large-scale scenarios, the domain gap between remote sensing and natural images is more significant. However, our proposed approach successfully bridges this gap, allowing the vision foundation model’s general understanding to be

![](images/de95b225ecf9ecf10fbf8cc83fdc512c916a6f99b7c6ef5d7eb44d06e7dd2eba.jpg)

![](images/b4be6fae6bb7b4e55ed4fa1c740b416e4f822e321dba16cd4cd1201884baa1a7.jpg)

![](images/f69e956353867be1b3aa25d108e45ae5e398a7a28af967f38ddf2e68cb48411c.jpg)

![](images/f815c60aa03d73ab943a5f81a1b6fa2fe1ff0ae87e85f5dbb23007e5dbf3fce8.jpg)

![](images/eb520f3c74ecae7fe6e42ae63af03f024a4a4deaf38a9100aace802529a1bab1.jpg)  
(e1)

![](images/b9eb23cc1be0bf7432034560c2d4bb586dcf36a46fa459093be5b17ef7f8d4f9.jpg)

![](images/2a8ca939c6abbc6328c9ee8009161c268e89f414e670ed4947e5ff9854f92f3b.jpg)

![](images/c04e6c87affae234395ac6b2aa0c2d2e12693aff3441c8dcbf3ab97d063f82a6.jpg)

![](images/567ae42469f82628899f79e4904cf2a27a3cc33f76cede9540ee2dc39765d282.jpg)

![](images/ca00c7470f9bcf761e35ee927ffc0c51b5053c8127c26cf73445f8b65862939e.jpg)

![](images/3e8481c40a6b3053e2ff4c0af9647978735ae9b1b55a6c370c820d720f5e65af.jpg)

![](images/b16f2f944b506cf16aa27f66dfc67514ac0eee63d6ef73a9603b3ab3b7df858a.jpg)

![](images/e949e911940d0ef6fc534049b8d12b9f5a3e924cdccd008f1485106f4a0a8fa8.jpg)

![](images/830356697997f1cc94f2b7a004d3d0d643700828ea7a8df91cc50fe7569e7bf9.jpg)

![](images/ec16017f5a15f6e32eeaa6a76645dd24e40429ee6443dc29268b8eec084816c7.jpg)

![](images/1ad71b759595b67f50086cb399e43eeea715ae225396d72097b6250353a11d4d.jpg)  
  
building building

![](images/d753c9c42ac3c69c578d692028590d3671c1cc92192fcfcc8fd4c56816d6b14e.jpg)

![](images/80317206084a9ebd3b814c5f9c8e450d7514d65a36634505376e213fd520546f.jpg)

![](images/e7280bd453f81b89a7e7260de674644df4c194410a3a76402191b7839cdb730d.jpg)  
(i2)

![](images/d58a2e326687464781d5380d97b75ac921e99dd21a68db1add58c5dec6a8a994.jpg)  
tree   
low vegetation   
car   
impervious surface   
clutter   
Fig. 9. Visualized comparisons on the Potsdam test set with the size of $1 0 2 4 \times 1 0 2 4$ . (a) IRRG images, (b) DSM, (c) ground truth, (d) CMFNet, (e) FTransUNet, (f) MFTransNet, (g) CMGFNet, (h) FTransDeepLab, (i) MultiSenseSeg, and (j) proposed MFNet. To highlight the differences, some purple boxes are added to all subfigures. Subscripts $( = 1 , 2 )$ represent the serial number of the samples displayed.

(2)

effectively transferred to remote sensing tasks, resulting in consistent performance improvements.

# D. Modality and Fine-Tuning Analysis

To illustrate the necessity of multimodal fine-tuning framework, we conducted the modality and fine-tuning analysis, with the results presented in Table IV. In the first experiment, we only used single-modality data and did not apply any finetuning mechanism, while in the second and fifth experiments, we applied standard Adapter/LoRA mechanisms to fine-tune the SAM’s image encoder. These experiments highlight the importance and need for multimodal information and finetuning mechanisms. In the third and sixth experiments, the SAM’s image encoder retained standard Adapter/LoRA but excluded the proposed MMAdapter/MMLoR. Therefore, these experiments could still extract remote sensing multimodal features independently, but they lacked crucial information fusion during the encoding process. The fourth and seventh experiments contain multimodal information and the proposed MMAdapter/MMLoR.

Inspection of Table IV first highlights the need for finetuning mechanisms. Without Adapter or LoRA, SAM struggles to effectively extract remote sensing features, leading to a significant decline in performance. Furthermore, the results reveal the great effectiveness of incorporating multimodal

information. The improvement is particularly pronounced in the categories of Building and Impervious surface, which tend to have significant and stable surface elevation information. After that, this enhancement strengthens the model’s ability to distinguish other categories. Additionally, we observe that the performance in the third experiment is lower than in the second experiment. This is attributed to low-rank decomposition significantly reducing the dimensionality of task-specific information. Hereby, the heterogeneity between modalities complicates their fusion after the encoder. It highlights the challenge that improper fine-tuning poses in leveraging multimodal information. Our MMLoRA effectively addresses this challenge through a gradual feature fusion approach in the image encoder.

Overall, the integration of multimodal information provides comprehensive benefits across the board. The introduction of the MMAdapter and MMLoRA enables more effective utilization of DSM information, significantly enhancing the model’s ability to extract and fuse multimodal information. As a result, the semantic segmentation performance is further improved.

# E. Ablation Study

1) Component Ablation: The proposed MFNet consists of two core components: the SAM’s image encoder with

![](images/f03af6bcc22e1f3a8178898146af3687a7b6ad5152fcfd3da9e5e61be4281b2e.jpg)

![](images/80a9eae3a4a9bf7c4f9401947252da04775e6d67892bbc70ca87b292dd0c0a1b.jpg)

![](images/44d817a4664d804a646219a0d9c108c9ae1bc79ca3decd7bcdc0910c927fa319.jpg)

![](images/8eff5287cf299965de97a4e48e503b68383898fc424290cc3bf3e31ad1138f92.jpg)

![](images/a9fb2f956fdd14c66750e7c6302c619a86d9e7f5a058cc079800d2850c10979b.jpg)

![](images/d0842435bbf32559701c2c86a8f905da90fd5538b28cf839432d227d2e31269d.jpg)

![](images/0c273d136f820fc9defd386a2813f8f29d2aa00c2c0a93ef9da3881c99bff6db.jpg)

![](images/1fad03dfe140ddbb8834178ddcb2fefd913b177e4284691c02f376095666038b.jpg)

![](images/1a4e47045f66753023d3eab7bc2619b20ecb589ff58e7fbe16ee4288103ad673.jpg)

![](images/d7a9515aca417f2626a57b136d44dd4f6bbdafed9e714e99a388aab19ebb8af1.jpg)

![](images/acf5fc0555f89e0c9ddf061aa0ce63013443c9408920671a5035af8c13c1e6f7.jpg)

![](images/49a13d7689d3406317d78cfb82fc41756434aac3f33d3dd111133e1d5a488d26.jpg)

![](images/7153effd72400f4996d1ccd025bc6187cc3e1993e592e6c46de024f1621c9080.jpg)

![](images/81f8fa52f60803a687e2f36b9fbecdfc881c2231ea7cc1718af6a4f319546966.jpg)

![](images/92bec820c5d61933a289b4ae32ad5fbee1b1214c812cd6ea5228d0e21d656a16.jpg)

![](images/8dbb0f8c44f5e93f0d3c1dc1e77c46b9ccd8fb8255935941c66466c1a262afc2.jpg)

![](images/95e706d1e837128c4dcf928594cfb6dda00df917837e13d041bb597cb1e19114.jpg)

![](images/81a7401dc3600ede8a041a6486b246b2200fede925c81fc61138a47ed865a852.jpg)

![](images/4b99bf8655404459df0ea550c8d0d039596e61c1d41a0fbe474816552166b33c.jpg)

![](images/587277303e3edf538eea03c243a8c85d554d6ed3253ec59547fcfb2f1899f9f4.jpg)

![](images/b1a6f7d2732b53e71d8d46974792824ccb200c9dee73247ac21cf83a1a57f94f.jpg)  
forest

![](images/037ea5a3a78fa8317fb6270fede63de381ed7bca38629268461391676a471cbd.jpg)  
cropland

![](images/2e5762a02bf9779bc86214fe8053e8223c92f31179025e64ad6ed72e78bdca3c.jpg)  
water

![](images/db13f5a2fa27e63ec82d4dccdd550d2b8e010cc14336b6229de1e67769367129.jpg)  
build-up area

![](images/14401166a1683432be4bd5d23db146187bf28e74fcf77c0dc5901848f6afef06.jpg)  
grassland

![](images/d8d6b64d04f689174a8f5a7be6bde4a5e60579c10319444fc5b78ebd641c9add.jpg)  
wetland

![](images/c890a2db20854444c74a0f3df4f61543f1f1b8068f3df46aeeae404ab3bb82b3.jpg)  
unused land   
Fig. 10. Visualized comparisons on the MMHunan test set with the size of $2 5 6 \times 2 5 6$ and $1 0 2 4 \times 1 0 2 4$ . (a) IRRG images, (b) DSM, (c) ground truth, (d) CMFNet, (e) FTransUNet, (f) MFTransNet, (g) CMGFNet, (h) FTransDeepLab, (i) MultiSenseSeg, and (j) proposed MFNet. To highlight the differences, some purple boxes are added to all subfigures. Subscripts (=1, 2) represent the serial number of the samples displayed.

TABLE IV   
QUANTITATIVE RESULTS ON THE VAIHINGEN DATASET WITH DIFFERENT MODALITIES AND FINE-TUNING MECHANISMS (%)   

<table><tr><td rowspan="2">Modality</td><td rowspan="2">Fine-tuning</td><td colspan="6">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Bui.</td><td>Tre.</td><td>Low.</td><td>Car</td><td>Imp.</td><td>Total</td></tr><tr><td>NIIRG</td><td>Without Adapter</td><td>94.64</td><td>89.47</td><td>71.71</td><td>76.83</td><td>89.51</td><td>88.01</td><td>85.34</td><td>75.11</td></tr><tr><td>NIIRG</td><td>Standard LoRA</td><td>96.50</td><td>93.62</td><td>80.35</td><td>86.32</td><td>92.78</td><td>92.00</td><td>90.35</td><td>82.74</td></tr><tr><td>NIIRG + DSM</td><td>Standard LoRA</td><td>97.26</td><td>92.61</td><td>81.58</td><td>86.53</td><td>91.58</td><td>91.86</td><td>89.90</td><td>82.06</td></tr><tr><td>NIIRG + DSM</td><td>MFNet with MMLoRA</td><td>96.85</td><td>92.89</td><td>81.09</td><td>89.95</td><td>93.28</td><td>92.22</td><td>91.09</td><td>83.96</td></tr><tr><td>NIIRG</td><td>Standard Adapter</td><td>96.29</td><td>93.09</td><td>80.15</td><td>89.08</td><td>92.59</td><td>92.02</td><td>90.94</td><td>83.69</td></tr><tr><td>NIIRG + DSM</td><td>Standard Adapter</td><td>99.02</td><td>91.68</td><td>83.04</td><td>89.71</td><td>92.90</td><td>92.80</td><td>91.30</td><td>84.35</td></tr><tr><td>NIIRG + DSM</td><td>MFNet with MMApapter</td><td>98.84</td><td>93.17</td><td>81.16</td><td>89.23</td><td>93.39</td><td>92.93</td><td>91.51</td><td>84.72</td></tr></table>

MMAdapter or MMLoRA and the DFM. To validate their effectiveness, ablation experiments were conducted by systematically removing specific components. As shown in Table V, two ablation experiments were designed. In the first experiment, the DFM was removed from MFNet, leading to a lack of deep analysis and integration of high-level abstract remote sensing semantic features. The second experiment follows the setup of the third and sixth experiments in Table IV.

Before analyzing the ablation experiments, it is important to note that removing all Adapters or LoRAs from the SAM’s image encoder severely degrades the model’s performance, as confirmed in Fig. 11, which also illustrates the effectiveness of the multimodal fine-tuning mechanisms. Inspection of Fig. 11(c) and (d) reveals that SAM, without fine-tuning, cannot extract meaningful features from remote sensing data,

rendering it unsuitable for semantic segmentation tasks. However, inspection of Fig. 11(d) and (f), or (g) and (h) shows that, after fine-tuning with the MMAdapter or MMLoRA, the heatmaps change dramatically. Furthermore, Fig. 11(f) and (h) clearly demonstrates that SAM, despite being trained on RGB optical images, is also effective when applied to nonoptical DSM data. It is observed that DSM can effectively provide supplementary information. Therefore, the fine-tuned SAM’s image encoder is capable of recognizing and segmenting remote sensing objects effectively in multimodal tasks.

Inspection of Table V indicates that both the multimodal fine-tuning and DFM are essential for enhancing the performance of the proposed MFNet. Specifically, the MMAdapter and MMLoRA facilitate continuous information fusion, allowing for extraction and fusion of multimodal information as the

![](images/a9e1f9fe11424111b293986c6df03a9523f9a137e1603ffb708a823d17d7cf08.jpg)

![](images/875427a0bcdf7af1e67224259307947c374bcca7e970cb8a1e08c3e5439afeae.jpg)  
(b)

![](images/b81d3ce79649792c172f816b733998c7224e919a386c8773d153b9ce353cab9e.jpg)  
(C)

![](images/b2c61b3354e1d9aa647e73ebf1bbb08f6da01a8df3961ead9008abc6cfa02af7.jpg)  
(d)

![](images/fc6cc4edbc0fb6d7f8aa2207c4f578cb85dfbc6cfe9360bc013c54138c24fe16.jpg)  
(e)

![](images/69d9ea9bbf4fcd1a825605beef669ba93ca361bc3f3ff46c5ace890ba85adee6.jpg)  
(f)

![](images/4ad57db848aaeb2ad6de475229076f7599c6a0edb4f6739ac4f2e48d7116bb1c.jpg)  
(g)

![](images/b0126d1df4b1e3af6b97561b45bb4b9c178ba32bcb3def2f8a5f81dba7a9b8c6.jpg)  
(h)   
Fig. 11. Four groups of heatmaps. (a) NIIRG images, (b) DSM, (c) heatmaps from NIIRG images, (d) heatmaps from DSM both generated by the original SAM, (e) heatmaps from NIIRG images, (f) heatmaps from DSM both generated by the proposed MMAdapter, (g) heatmaps from NIIRG images, and (h) heatmaps from DSM both generated by the proposed MMLoRA. The high-value areas in the heatmaps indicate objects identified as buildings by the methods. The effectiveness of our MMAdapter and MMLoRA can be clearly observable.

TABLE V ABLATION STUDY OF THE PROPOSED MFNET. THE BEST RESULTS ARE IN BOLD   

<table><tr><td>MMAdapter</td><td>DFM</td><td>OA(%)</td><td>mF1(%)</td><td>mIoU(%)</td><td>MMLoRA</td><td>DFM</td><td>OA(%)</td><td>mF1(%)</td><td>mIoU(%)</td></tr><tr><td>✓</td><td></td><td>92.73</td><td>91.23</td><td>84.25</td><td>✓</td><td></td><td>92.02</td><td>90.64</td><td>83.24</td></tr><tr><td></td><td>✓</td><td>92.80</td><td>91.30</td><td>84.35</td><td></td><td>✓</td><td>91.86</td><td>89.90</td><td>82.06</td></tr><tr><td>✓</td><td>✓</td><td>92.93</td><td>91.51</td><td>84.72</td><td>✓</td><td>✓</td><td>92.22</td><td>91.09</td><td>83.96</td></tr></table>

![](images/83f5afac6beb273f0597b274f6b2d8e3b5a536fdf61e1979c1d634b05b2c6cd9.jpg)  
Fig. 12. Relationship between training data volume and model performance during the training stage. As the model is unable to generate predictions in the absence of training data, we designed and carried out experiments with three levels of training data availability: $2 5 \%$ , $50 \%$ , and $7 5 \%$ , in addition to using the complete training set $( 1 0 0 \% )$ .

encoding depth increases. The DFM verifies the importance of high-level features in the semantic segmentation of remote sensing data. In this work, we primarily introduce a new framework for leveraging SAM, rather than emphasizing the high-level feature fusion techniques. Replacing DFM with a more advanced fusion model is expected to result in further performance improvement.

2) Data Amount Ablation: To investigate the fine-tuning efficiency of SAM on remote sensing tasks, we conducted experiments using varying proportions of the training data to

explore the relationship between training data volume and model performance. Specifically, we fine-tuned the model using only $2 5 \%$ , $50 \%$ , and $7 5 \%$ of the training set while evaluating on the full test set. The results, shown in Fig. 12, reveal a phenomenon where data quantity plays a crucial role between $2 5 \%$ and $50 \%$ , yet the performance gains tend to saturate beyond the $50 \%$ threshold. This suggests that SAM is capable of rapidly acquiring task-specific knowledge through fine-tuning, making further increases in training data yield diminishing returns in downstream performance. This finding offers valuable insights into the data requirements for related tasks, providing guidance on the efficient use of training data.

# F. Model Scale Analysis

The improved performance of MFNet is largely attributable to the general knowledge provided by the vision foundation model, SAM. However, SAM is also a large model, and the large model does not have advantages in terms of computational complexity or inference speed compared to existing general methods. Consequently, we focus on reporting the model’s trainable parameter number and memory footprint to measure its hardware requirements.

Table VI presents the results of the model scale for all methods compared in this work. As indicated in Table VI, the proposed multimodal fine-tuning techniques allow the large foundation models to be used on a single GPU while maintaining a manageable number of trainable parameters and memory costs. The parameter statistics of MFNet are divided into two parts: the fine-tuning parameters in SAM’s image encoder $+$ the parameters in DFM and the decoder. The parameters in the latter remain consistent across different MFNet configurations.

TABLE VI MODEL SCALE ANALYSIS MEASURED BY A $2 5 6 ~ \times ~ 2 5 6$ IMAGE ON A SINGLE NVIDIA GEFORCE RTX 3090 GPU. FOR DIFFERENT MFNET CONFIGURATIONS, THE PARAMETER STATISTICS ARE: THE FINE-TUNING PARAMETERS IN SAM’S IMAGE ENCODER $^ +$ THE PARAMETERS IN DFM AND THE DECODER. MIOU VALUES ARE THE RESULTS OF THE VAIHINGEN DATASET. THE BEST RESULTS ARE IN BOLD   

<table><tr><td>Method</td><td>Parameter (M)</td><td>Memory (MB)</td><td>MIoU (%)</td></tr><tr><td>PSPNet [54]</td><td>46.72</td><td>3124</td><td>76.96</td></tr><tr><td>MAResU-Net [51]</td><td>26.27</td><td>1908</td><td>79.89</td></tr><tr><td>UNetFormer [35]</td><td>24.20</td><td>1980</td><td>81.97</td></tr><tr><td>RS3Mamba [57]</td><td>43.32</td><td>1548</td><td>82.78</td></tr><tr><td>TransUNet [56]</td><td>93.23</td><td>3028</td><td>78.26</td></tr><tr><td>FuseNet [10]</td><td>42.08</td><td>2284</td><td>78.71</td></tr><tr><td>vFuseNet [36]</td><td>44.17</td><td>2618</td><td>78.92</td></tr><tr><td>ESANet [52]</td><td>34.03</td><td>1914</td><td>79.42</td></tr><tr><td>SA-GATE [55]</td><td>110.85</td><td>3174</td><td>81.27</td></tr><tr><td>CMFNet [15]</td><td>123.63</td><td>4058</td><td>81.44</td></tr><tr><td>MFTransUNet [16]</td><td>43.77</td><td>1549</td><td>81.61</td></tr><tr><td>CMGFNet [53]</td><td>64.20</td><td>2463</td><td>82.26</td></tr><tr><td>FTransUNet [33]</td><td>160.88</td><td>3463</td><td>84.23</td></tr><tr><td>FTransDeepLab [58]</td><td>69.86</td><td>1624</td><td>83.87</td></tr><tr><td>MultiSenseSeg [59]</td><td>60.46</td><td>2264</td><td>84.53</td></tr><tr><td>MFNet (MMLoRA) (ViT-B)</td><td>1.03+6.22</td><td>1924</td><td>82.09</td></tr><tr><td>MFNet (MMLoRA) (ViT-L)</td><td>2.75+6.22</td><td>4158</td><td>83.96</td></tr><tr><td>MFNet (MMLoRA) (ViT-H)</td><td>4.59+6.22</td><td>6520</td><td>84.66</td></tr><tr><td>MFNet (MMAdapter) (ViT-B)</td><td>14.20+6.22</td><td>1872</td><td>83.24</td></tr><tr><td>MFNet (MMAdapter) (ViT-L)</td><td>50.45+6.22</td><td>4242</td><td>84.72</td></tr><tr><td>MFNet (MMAdapter) (ViT-H)</td><td>105.06+6.22</td><td>6854</td><td>85.03</td></tr></table>

A comparison between MMLoRA and MMAdapter shows that MMLoRA significantly reduces the number of parameters by compressing thousands of dimensional spaces into a rank of four through low-rank decomposition. While this approach is generally efficient, it may result in the loss of some essential information, especially when processing complex remote sensing data. Consequently, MMAdapter outperforms MMLoRA in terms of performance.

In our experiments, we successfully fine-tuned the ViT-L backbone on the same hardware with the same hyperparameters, achieving results that surpassed all existing methods. For the ViT-H backbone, we adjusted the batch size from 10 to 4 due to the GPU memory limitation. This reduction in batch size did not degrade performance but further improved performance. These results prove the powerful feature extraction and fusion capabilities of the large vision foundation model. This work also offers valuable insights for exploring multimodal tasks with large models under constrained hardware conditions.

# G. Discussion

This work introduces a unified multimodal fine-tuning framework with two SAM-based multimodal fine-tuning mechanisms. As an early exploration of this field, we thoroughly investigate the performance of the vision foundation model on remote sensing multimodal tasks by developing two classical fine-tuning approaches: Adapter and LoRA. Comprehensive analytical experiments are conducted to evaluate these methods. Additionally, MFNet offers a straightforward multimodal fusion network, paving the way for future research in the following directions.

1) Improving Fine-Tuning Modules: This work employs two representative fine-tuning techniques, namely LoRA and Adapter, to demonstrate the framework’s effectiveness. Future research is encouraged to apply more advanced variants, such as [60], [61], [62], and [63] to diverse multimodal remote sensing tasks. In particular, more efficient fine-tuning strategies for large-scale models are worth exploring, as foundation models typically require substantial memory resources.

2) Improving Fusion Modules: The work employs an adaptive weight-based method for feature fusion during the encoding stage. Future work could explore more advanced and effective fusion strategies tailored to MMAdapter and MMLoRA. Similarly, the fusion of deep high-level features can be enhanced with other fusion mechanisms, such as cross-attention, to improve the performance.

3) Addressing Challenging Categories: SAM’s performance on challenging categories, such as distinguishing between very similar trees and low vegetation, or accurately detecting small objects such as cars, demands further investigation. To improve the accuracy, it may be necessary to develop specialized object recognition modules that focus on these specific tasks. This could involve category-specific feature extraction techniques.

4) Exploration of Other Remote Sensing Modalities: This study demonstrates the superiority of multimodal fine-tuning framework using optical images and DSM data as examples and also provides valuable insights into the potential of combining these modalities. However, the performance of SAM on other remote sensing modalities, such as multispectral, LiDAR, and SAR, offers an exciting avenue for future investigation. Exploring these modalities could further enhance SAM’s capabilities and broaden its applicability in diverse remote sensing tasks.

Overall, this work serves as a foundational framework, with several aspects left open for further exploration. We hope that it can be broadly extended to various types of multimodal remote sensing tasks.

# V. CONCLUSION

In this study, we proposed a unified multimodal fusion framework with multimodal fine-tuning for remote sensing semantic segmentation, leveraging the general knowledge embedded in the vision foundation model, SAM. Using two representative single-modal fine-tuning mechanisms, namely Adapter and LoRA, we demonstrated the seamless integration of existing mechanisms into the proposed unified framework for extracting and fusing multimodal features from remote sensing data. The fused deep features were further refined using a pyramid-based DFM and reconstructed into segmentation maps. Comprehensive experiments on three benchmark multimodal datasets, ISPRS Vaihingen, ISPRS Potsdam, and MMHunan, confirmed that MFNet achieves superior performance compared to current state-of-the-art segmentation methods. This research represents the first validation of SAM’s reliability on DSM data and offers a promising pathway for leveraging vision foundation models in multimodal remote

sensing tasks. Moreover, the proposed framework has the potential to be extended to other remote sensing applications, including semi-supervised and unsupervised learning tasks.

# REFERENCES

[1] L. Gómez-Chova, D. Tuia, G. Moser, and G. Camps-Valls, “Multimodal classification of remote sensing images: A review and future directions,” Proc. IEEE, vol. 103, no. 9, pp. 1560–1584, Sep. 2015.   
[2] J. Li et al., “Deep learning in multimodal remote sensing data fusion: A comprehensive review,” Int. J. Appl. Earth Observ. Geoinf., vol. 112, Aug. 2022, Art. no. 102926.   
[3] J. Yao, B. Zhang, C. Li, D. Hong, and J. Chanussot, “Extended vision transformer (ExViT) for land use and land cover classification: A multimodal deep learning framework,” IEEE Trans. Geosci. Remote Sens., vol. 61, 2023, Art. no. 5514415.   
[4] D. Hong, J. Hu, J. Yao, J. Chanussot, and X. X. Zhu, “Multimodal remote sensing benchmark datasets for land cover classification with a shared and specific feature learning model,” ISPRS J. Photogramm. Remote Sens., vol. 178, pp. 68–80, Aug. 2021.   
[5] P. Karmakar, S. W. Teng, M. Murshed, S. Pang, Y. Li, and H. Lin, “Crop monitoring by multimodal remote sensing: A review,” Remote Sens. Appl., Soc. Environ., vol. 33, Jan. 2024, Art. no. 101093.   
[6] N. Algiriyage, R. Prasanna, K. Stock, E. E. H. Doyle, and D. Johnston, “Multi-source multimodal data and deep learning for disaster response: A systematic review,” Social Netw. Comput. Sci., vol. 3, no. 1, pp. 1–29, Jan. 2022.   
[7] X. Zhang, W. Yu, M.-O. Pun, and W. Shi, “Cross-domain landslide mapping from large-scale remote sensing images using prototype-guided domain-aware progressive representation learning,” ISPRS J. Photogramm. Remote Sens., vol. 197, pp. 1–17, Mar. 2023.   
[8] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in Proc. Int. Conf. Med. Image Comput. Comput.-Assist. Intervent., 2015, pp. 234–241.   
[9] R. Li, S. Zheng, C. Zhang, C. Duan, L. Wang, and P. M. Atkinson, “ABCNet: Attentive bilateral contextual network for efficient semantic segmentation of fine-resolution remotely sensed imagery,” ISPRS J. Photogramm. Remote Sens., vol. 181, pp. 84–98, Nov. 2021.   
[10] C. Hazirbas, L. Ma, C. Domokos, and D. Cremers, “FuseNet: Incorporating depth into semantic segmentation via fusion-based CNN architecture,” in Proc. Asian Conf. Comput. Vis., 2016, pp. 213–228.   
[11] X. Zhang, W. Yu, and M.-O. Pun, “Multilevel deformable attentionaggregated networks for change detection in bitemporal remote sensing imagery,” IEEE Trans. Geosci. Remote Sens., vol. 60, 2022, Art. no. 5621518.   
[12] A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Inf. Process. Syst., vol. 30, 2017, pp. 1–11.   
[13] A. Dosovitskiy et al., “An image is worth $1 6 \times 1 6$ words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Represent., 2021, pp. 1–22.   
[14] Z. Liu et al., “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Oct. 2021, pp. 10012–10022.   
[15] X. Ma, X. Zhang, and M.-O. Pun, “A crossmodal multiscale fusion network for semantic segmentation of remote sensing data,” IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., vol. 15, pp. 3463–3474, 2022.   
[16] S. He, H. Yang, X. Zhang, and X. Li, “MFTransNet: A multi-modal fusion with CNN-transformer network for semantic segmentation of HSR remote sensing images,” Mathematics, vol. 11, no. 3, p. 722, Feb. 2023.   
[17] X. Ma, X. Zhang, X. Ding, M.-O. Pun, and S. Ma, “Decompositionbased unsupervised domain adaptation for remote sensing image semantic segmentation,” IEEE Trans. Geosci. Remote Sens., vol. 62, 2024, Art. no. 5645118.   
[18] A. Kirillov et al., “Segment anything,” in Proc. IEEE/CVF Int. Conf. Comput. Vis., Oct. 2023, pp. 4015–4026.   
[19] J. Ma, Y. He, F. Li, L. Han, C. You, and B. Wang, “Segment anything in medical images,” Nature Commun., vol. 15, no. 1, p. 654, Jan. 2024.   
[20] H. Wang et al., “SAM-CLIP: Merging vision foundation models towards semantic and spatial understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), Jun. 2024, pp. 3635–3647.   
[21] Y. Li, H. Zhang, X. Xue, Y. Jiang, and Q. Shen, “Deep learning for remote sensing image classification: A survey,” Wiley Interdiscipl. Rev., Data Mining Knowl. Discovery, vol. 8, no. 6, p. e1264, May 2018.

[22] Z. Zheng, Y. Zhong, J. Wang, and A. Ma, “Foreground-aware relation network for geospatial object segmentation in high spatial resolution remote sensing imagery,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2020, pp. 4096–4105.   
[23] X. Ma, X. Zhang, Z. Wang, and M.-O. Pun, “Unsupervised domain adaptation augmented by mutually boosted attention for semantic segmentation of VHR remote sensing images,” IEEE Trans. Geosci. Remote Sens., vol. 61, 2023, Art. no. 5400515.   
[24] N. Houlsby et al., “Parameter-efficient transfer learning for NLP,” in Proc. Int. Conf. Mach. Learn., 2019, pp. 2790–2799.   
[25] S. Chen et al., “AdaptFormer: Adapting vision transformers for scalable visual recognition,” in Proc. NIPS, vol. 35, 2022, pp. 16664–16678.   
[26] X. He, C. Li, P. Zhang, J. Yang, and X. E. Wang, “Parameter-efficient model adaptation for vision transformers,” in Proc. AAAI Conf. Artif. Intell., 2023, vol. 37, no. 1, pp. 817–825.   
[27] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learn. Represent., 2022, pp. 1–20.   
[28] Q. Zhang et al., “Adaptive budget allocation for parameter-efficient finetuning,” in Proc. 11th Int. Conf. Learn. Represent., 2023, pp. 1–17.   
[29] X. Pu, H. Jia, L. Zheng, F. Wang, and F. Xu, “ClassWise-SAM-adapter: Parameter efficient fine-tuning adapts segment anything to SAR domain for semantic segmentation,” 2024, arXiv:2401.02326.   
[30] X. Zhou et al., “MeSAM: Multiscale enhanced segment anything model for optical remote sensing images,” IEEE Trans. Geosci. Remote Sens., vol. 62, 2024, Art. no. 5623515.   
[31] X. Lu and Q. Weng, “Multi-LoRA fine-tuned segment anything model for urban man-made object extraction,” IEEE Trans. Geosci. Remote Sens., vol. 62, 2024, Art. no. 5637519.   
[32] F. I. Diakogiannis, F. Waldner, P. Caccetta, and C. Wu, “ResUNeta: A deep learning framework for semantic segmentation of remotely sensed data,” ISPRS J. Photogramm. Remote Sens., vol. 162, pp. 94–114, Apr. 2020.   
[33] X. Ma, X. Zhang, M.-O. Pun, and M. Liu, “A multilevel multimodal fusion transformer for remote sensing semantic segmentation,” IEEE Trans. Geosci. Remote Sens., vol. 62, 2024, Art. no. 5403215.   
[34] P. Zhang, B. Peng, C. Lu, Q. Huang, and D. Liu, “ASANet: Asymmetric semantic aligning network for RGB and SAR image land cover classification,” ISPRS J. Photogramm. Remote Sens., vol. 218, pp. 574–587, Dec. 2024.   
[35] L. Wang et al., “UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery,” ISPRS J. Photogramm. Remote Sens., vol. 190, pp. 196–214, Aug. 2022.   
[36] N. Audebert, B. Le Saux, and S. Lefèvre, “Beyond RGB: Very high resolution urban remote sensing with multimodal deep networks,” ISPRS J. Photogramm. Remote Sens., vol. 140, pp. 20–32, Jun. 2018.   
[37] J. Wu et al., “Medical SAM adapter: Adapting segment anything model for medical image segmentation,” 2023, arXiv:2304.12620.   
[38] D. Li et al., “FusionSAM: Visual multi-modal learning with segment anything,” 2024, arXiv:2408.13980.   
[39] M. A. Mazurowski, H. Dong, H. Gu, J. Yang, N. Konz, and Y. Zhang, “Segment anything model for medical image analysis: An experimental study,” Med. Image Anal., vol. 89, Oct. 2023, Art. no. 102918.   
[40] D. Wang et al., “SAMRS: Scaling-up remote sensing segmentation dataset with segment anything model,” in Proc. 37th Conf. Neural Inf. Process. Syst. Datasets Benchmarks Track, 2023, pp. 1–13.   
[41] Z. Qi et al., “Multi-view remote sensing image segmentation with sam priors,” in Proc. IEEE Int. Geosci. Remote Sens. Symp. (IGARSS), Jul. 2024, pp. 8446–8449.   
[42] H. Chen, J. Song, and N. Yokoya, “Change detection between optical remote sensing imagery and map data via segment anything model (SAM),” 2024, arXiv:2401.09019.   
[43] X. Ma, Q. Wu, X. Zhao, X. Zhang, M.-O. Pun, and B. Huang, “SAMassisted remote sensing imagery semantic segmentation with object and boundary constraints,” IEEE Trans. Geosci. Remote Sens., vol. 62, 2024, Art. no. 5636916.   
[44] Z. Yan et al., “RingMo-SAM: A foundation model for segment anything in multimodal remote-sensing images,” IEEE Trans. Geosci. Remote Sens., vol. 61, 2023, Art. no. 5625716.   
[45] L. Ding, K. Zhu, D. Peng, H. Tang, K. Yang, and L. Bruzzone, “Adapting segment anything model for change detection in VHR remote sensing images,” IEEE Trans. Geosci. Remote Sens., vol. 62, 2024, Art. no. 5611711.   
[46] L. Mei et al., “SCD-SAM: Adapting segment anything model for semantic change detection in remote sensing imagery,” IEEE Trans. Geosci. Remote Sens., 2024, Art. no. 5626713.

[47] A. Aghajanyan, L. Zettlemoyer, and S. Gupta, “Intrinsic dimensionality explains the effectiveness of language model fine-tuning,” 2020, arXiv:2012.13255.   
[48] Y. Li, H. Mao, R. Girshick, and K. He, “Exploring plain vision transformer backbones for object detection,” in Proc. Eur. Conf. Comput. Vis. Cham, Switzerland: Springer, 2022, pp. 280–296.   
[49] K. Zhang and D. Liu, “Customized segment anything model for medical image segmentation,” 2023, arXiv:2304.13785.   
[50] Y. Li, Y. Zhou, Y. Zhang, L. Zhong, J. Wang, and J. Chen, “DKDFN: Domain knowledge-guided deep collaborative fusion network for multimodal unitemporal remote sensing land cover classification,” ISPRS J. Photogramm. Remote Sens., vol. 186, pp. 170–189, Apr. 2022.   
[51] R. Li, S. Zheng, C. Duan, J. Su, and C. Zhang, “Multistage attention ResU-Net for semantic segmentation of fine-resolution remote sensing images,” IEEE Geosci. Remote Sens. Lett., vol. 19, pp. 1–5, 2022.   
[52] D. Seichter, M. Köhler, B. Lewandowski, T. Wengefeld, and H.-M. Gross, “Efficient RGB-D semantic segmentation for indoor scene analysis,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), May 2021, pp. 13525–13531.   
[53] H. Hosseinpour, F. Samadzadegan, and F. D. Javan, “CMGFNet: A deep cross-modal gated fusion network for building extraction from very highresolution remote sensing images,” ISPRS J. Photogramm. Remote Sens., vol. 184, pp. 96–115, Feb. 2022.   
[54] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, “Pyramid scene parsing network,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Jul. 2017, pp. 2881–2890.   
[55] X. Chen et al., “Bi-directional cross-modality feature propagation with separation-and-aggregation gate for RGB-D semantic segmentation,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 561–577.   
[56] J. Chen et al., “TransUNet: Transformers make strong encoders for medical image segmentation,” 2021, arXiv:2102.04306.   
[57] X. Ma, X. Zhang, and M.-O. Pun, “RS3Mamba: Visual state space model for remote sensing image semantic segmentation,” IEEE Geosci. Remote Sens. Lett., vol. 21, pp. 1–5, 2024.   
[58] H. Feng et al., “FTransDeepLab: Multimodal fusion transformer-based DeepLabv $^ { + }$ for remote sensing semantic segmentation,” IEEE Trans. Geosci. Remote Sens., vol. 63, 2025, Art. no. 4406618.   
[59] Q. Wang, W. Chen, Z. Huang, H. Tang, and L. Yang, “MultiSenseSeg: A cost-effective unified multimodal semantic segmentation model for remote sensing,” EEE Trans. Geosci. Remote Sens., vol. 62, 2024, Art. no. 4703724.   
[60] T. Lei et al., “Conditional adapters: Parameter-efficient transfer learning with fast inference,” in Proc. Adv. Neural Inf. Process. Syst., vol. 36, 2023, pp. 8152–8172.   
[61] Y. Chen et al., “Hadamard adapter: An extreme parameter-efficient adapter tuning method for pre-trained language models,” in Proc. 32nd ACM Int. Conf. Inf. Knowl. Manage., Oct. 2023, pp. 276–285.   
[62] S.-Y. Liu et al., “DoRA: Weight-decomposed low-rank adaptation,” in Proc. 41st Int. Conf. Mach. Learn., 2024, pp. 1–22.   
[63] S. Hayou, N. Ghosh, and B. Yu, “LoRA+: Efficient low rank adaptation of large models,” 2024, arXiv:2402.12354.

![](images/36ee4574f87efd5dd0193285ded4d47f15545f23e115ed30fedb947b44f29b2d.jpg)

Xiaokang Zhang (Senior Member, IEEE) received the Ph.D. degree in photogrammetry and remote sensing from Wuhan University, Wuhan, China, in 2018.

From 2019 to 2022, he was a Post-Doctoral Research Associate at The Hong Kong Polytechnic University, Hong Kong, and The Chinese University of Hong Kong-Shenzhen, Shenzhen, China. He is currently a specially appointed Professor with the School of Information Science and Engineering, Wuhan University of Science and Technology,

Wuhan. He has authored or co-authored more than 50 scientific publications in international journals and conferences. His research interests include remote sensing image analysis, computer vision, and machine learning.

Dr. Zhang is a reviewer for more than 40 renowned international journals and conferences.

![](images/51c7c3256223337eb164c32ff2a299a16793fdf2c8a336c0710d6cd74c4b1cd5.jpg)

Man-On Pun (Senior Member, IEEE) received the B.Eng. degree in electronic engineering from The Chinese University of Hong Kong (CUHK), Hong Kong, in 1996, the M.Eng. degree in computer science from the University of Tsukuba, Tsukuba, Japan, in 1999, and the Ph.D. degree in electrical engineering from the University of Southern California (USC), Los Angeles, CA, USA, in 2006.

He was a Post-Doctoral Research Associate at Princeton University, Princeton, NJ, USA, from 2006 to 2008. He is currently an Associate

Professor with the School of Science and Engineering, The Chinese University of Hong Kong-Shenzhen (CUHKSZ), Shenzhen, China. Prior to joining CUHKSZ in 2015, he held research positions at Huawei, Milford, NJ, USA; Mitsubishi Electric Research Labs (MERL), Boston, MA, USA; and Sony, Tokyo, Japan. His research interests include AI Internet of Things (AIoT) and applications of machine learning in communications and satellite remote sensing.

Prof. Pun received the Best Paper Awards from IEEE VTC’06 Fall, IEEE ICC’08, and IEEE Infocom’09. He served as an Associate Editor for IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS from 2010 to 2014. He is the Founding Chair of the IEEE Joint SPS-ComSoc Chapter, Shenzhen.

![](images/c84d45b9d0791a9fc49f28e3b0e821bfd5931f3e15f97e1abc652e984ac56ee9.jpg)

Xianping Ma (Member, IEEE) received the bachelor’s degree in geographical information science from Wuhan University, Wuhan, China, in 2019, and the Ph.D. degree in computer and information engineering from The Chinese University of Hong Kong, Shenzhen, China, in 2025.

Since 2025, he has been with the Faculty of Geosciences and Environmental Engineering, Southwest Jiaotong University, Chengdu, China, as an Assistant Professor. His research interests include remote sensing image processing, deep learning, multimodal

fusion, and unsupervised domain adaptation. He has authored or co-authored more than 20 scientific publications in international journals and conferences.

Dr. Ma serves as a reviewer for more than 30 renowned international journals, such as ISPRS Journal of Photogrammetry and Remote Sensing, Remote Sensing of Environment, IEEE TRANSACTIONS ON IMAGE PROCESSING, and IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING.

![](images/fac1d0bd5132c0fded819776de27cc013d16e46ae87923fcd84774f06d52bcb4.jpg)

Bo Huang received the Ph.D. degree in remote sensing and mapping from the Institute of Remote Sensing Applications, Chinese Academy of Sciences, Beijing, China, in 1997.

He is currently the Chair Professor with the Department of Geography, The University of Hong Kong, Hong Kong, China. His research interests cover most aspects of GIScience, specifically the design and development of models and algorithms for unified satellite image fusion, spatiotemporal statistics and multiobjective spatial optimization, and

their applications in environmental monitoring and sustainable spatial planning.

Dr. Huang serves as an Associate Editor for International Journal of Geographical Information Science (Taylor and Francis) and was the Editorin-Chief of Comprehensive GIS (Elsevier), a three-volume GIS sourcebook.