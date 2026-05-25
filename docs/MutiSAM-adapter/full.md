# Multimodal SAM-adapter for Semantic Segmentation

Iacopo Curti1, Pierluigi Zama Ramirez1, Alioscia Petrelli2, and Luigi Di Stefano1

1University of Bologna, Bologna 40136, Italy 2SINA, Bologna 40132, Italy

# Abstract

Semantic segmentation, a key task in computer vision with broad applications in autonomous driving, medical imaging, and robotics, has advanced substantially with deep learning. Nevertheless, current approaches remain vulnerable to challenging conditions such as poor lighting, occlusions, and adverse weather. To address these limitations, multimodal methods that integrate auxiliary sensor data (e.g., LiDAR, infrared) have recently emerged, providing complementary information that enhances robustness. In this work, we present MM SAM-adapter, a novel framework that extends the capabilities of the Segment Anything Model (SAM) for multimodal semantic segmentation. The proposed method employs an adapter network that injects fused multimodal features into SAM’s rich RGB features. This design enables the model to retain the strong generalization ability of RGB features while selectively incorporating auxiliary modalities only when they contribute additional cues. As a result, MM SAM-adapter achieves a balanced and efficient use of multimodal information. We evaluate our approach on three challenging benchmarks, DeLiVER, FMB, and MUSES, where MM SAM-adapter delivers state-of-the-art performance. To further analyze modality contributions, we partition DeLiVER and FMB into RGB-easy and RGB-hard subsets. Results consistently demonstrate that our framework outperforms competing methods in both favorable and adverse conditions, highlighting the effectiveness of multimodal adaptation for robust scene understanding.

The code is available at the following GitHub Repository.

Keywords adapter, event cameras, thermal cameras, depth, LiDAR, multimodal semantic segmentation, SAM

# 1 Introduction

Semantic segmentation is a fundamental computer vision task that assigns a category label to each image pixel, with applications in fields such as autonomous driving, medical image analysis, and robotic navigation. The advent of deep learning has revolutionized semantic segmentation, leading to remarkable improvements in accuracy, efficiency, and generalization across environments [38]. Despite advances, semantic segmentation from RGB images fails in challenging situations such as poorly lit scenes, adverse weather conditions, and motion blur. To overcome these limitations, additional sensors that provide complementary measurements, such as infrared or event cameras and LiDAR, may be deployed to enhance segmentation performance. The increasing adoption of auxiliary sensors has fostered the creation of numerous datasets and approaches [49] that use multiple input modalities to pursue semantic segmentation. Standard practice for state-of-the-art multimodal methods is to train networks on the target dataset starting from ImageNet [13] weights, i.e., building upon the foundational knowledge acquired by image classification.

![](images/8a7004da56f1067d3df93951adc1ed609427ee981d40fb23228f115626236ed5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["RGB"] --> C["Segment Anything Model (SAM)"]
    B["Auxiliary Modality"] --> C
    C --> D["adapter"]
    D --> E["Seg. Head"]
    F["MM Fusion Encoder"] --> C
```
</details>

Figure 1: We propose adapting the Segment Anything Model’s rich and general knowledge for multimodal semantic segmentation using an external adapter network. Thanks to a Multimodal Fusion Encoder, our adapter leverages both the RGB and auxiliary modality (depth map in the Figure) information to achieve optimal performance on both challenging and easy scenarios.

However, the recent availability of large-scale annotated datasets and large GPU clusters has enabled the development of foundational segmentation models capable of accurately segmenting any RGB image. Among these, the Segment Anything Model (SAM) [26], trained on 11 million images and 1 billion ground truth masks, stands out for its impressive generalization capabilities. The availability of such strong foundational models for segmentation made us question whether we could employ the segmentation-specific knowledge embedded in models like SAM to tackle multimodal semantic segmentation. Although SAM is designed to perform instance-level segmentation via point or box prompts on RGB images, we conjecture that the fine-grained features dealing with spatial and contextual relationships between objects and parts extracted by the SAM encoder may be used to address pixel-level semantic segmentation even when processing RGB images alongside other modalities. Thus, in this paper, we propose to deploy the rich and generalizable features learned by SAM to achieve multimodal semantic segmentation. In particular, our goal is to adapt SAM to multimodal semantic segmentation while retaining its previous general knowledge. To achieve our goal, we draw inspiration from the strategy proposed by ViT-adapter [7], which pertains to tailoring a pre-trained Vision Transformer (ViT) [15] for dense tasks by an external adapter network. Based on the mechanism of cross-attention, this design can adapt the foundational ViT [15] features learned through image classification to tasks such as semantic segmentation, avoiding catastrophic forgetting. In our work, we employ a similar external adapter (the pink block in Figure 1), yet our goal is not only to adapt SAM features to a new task, namely from instance segmentation via prompts to semantic segmentation, but also to integrate features extracted from an auxiliary modality, such as, e.g., depth maps, LiDAR measurements, thermal or event data. Our architecture combines SAM alongside an adapter network. This design follows our intuitions that the RGB features learned by SAM through massive pre-training on billions of images can serve as a strong foundation for multimodal semantic segmentation and that tailoring to the new task without forgetting reusable knowledge about instances and their parts may be achieved effectively through modern adaptation strategies such as [7]. To prioritize its general and rich RGB features, we combine SAM with a lighter adapter network geared to incorporate the auxiliary modality. This asymmetric design materializes our intuition that, due to the outstanding segmentation capabilities of image-based models like SAM, multimodal semantic segmentation may best be tackled by relying mainly on RGB images and utilizing other modalities to handle challenging situations where the image turns out insufficiently informative. This starkly contrasts traditional multimodal architectures, which typically include networks of similar capacity to process the different modalities, thereby implicitly weighing each modality equally. Our experiments show that the SAM adaptation strategy outlined so far can handle challenging settings effectively. However, they also reveal that introducing an auxiliary modality can sometimes degrade performance when dealing with highly informative RGB content, e.g., images acquired in perfectly lit environments. To address this issue, we propose to adapt SAM using fused features computed from both the RGB and the auxiliary modality, as illustrated by the orange block in Figure 1. This design allows the model to learn to incorporate auxiliary information only when beneficial. For instance, in RGB-LiDAR segmentation, the model can learn to exploit LiDAR information in low-light settings while relying only on RGB features in perfectly lit environments. In summary, our design harnesses the strengths of foundational RGB models and leverages multimodal information to improve performance in challenging scenarios. We evaluated our proposal on several multimodal benchmarks such as DeLiVER [48], FMB [29], and MUSES [2], achieving state-of-the-art performance in all the considered datasets. Furthermore, we observe that many existing multimodal benchmarks fail to clearly highlight the benefits of auxiliary sensing modalities, as most data samples can be effectively segmented using only RGB information. Hence, we also conducted an evaluation using manually crafted test set splits. Specifically, for both DeLiVER and FMB, we divide the test set into two splits: RGB-easy and RGB-hard (see Figure 3). The former includes samples in which the RGB image provides enough information for accurate segmentation. Conversely, in the samples assigned to the latter split, the RGB content alone is poorly informative, making it necessary to rely on the auxiliary modality. By evaluating multimodal approaches on these splits, we better assess their ability to integrate information from multiple sensing modalities synergistically. Notably, our approach consistently performs best on both the RGB-easy and RGB-hard subsets of the DeLiVER and FMB test sets. Furthermore, our multimodal adaptation method, built on the ViT-adapter architecture, outperforms alternative approaches such as LoRA [21], adapted for multimodal input, on the DeLiVER benchmark in the RGB–LiDAR scenario.

To summarize, our contributions are as follows:

• We propose to adapt SAM foundational features to pursue multimodal semantic segmentation. In particular, our multimodal adapter, outlined in Section 3.1.3, integrates fused features, obtained with a fusion module described in Section 3.1.2, with SAM’s RGB-only features.

• Our approach attains state-of-the-art performance on the DeLiVER [48], FMB [29], and MUSES [2] benchmarks.

• To highlight the effectiveness of multimodal methods in synergistically exploiting the available sensing modalities, we split the DeLiVER and FMB test sets into RGB-hard and RGB-easy samples and evaluate methods on these splits. Our method performs best in all settings.

# 2 Related work

Our work tackles multimodal semantic segmentation by proposing an adaptation strategy for SAM’s foundational features. Hence, in this section, we first review the literature dealing with segmenting images either based on the RGB content or by also deploying auxiliary modalities. Then we pinpoint some recent approaches that attempt to deploy SAM alongside other modalities, as well as previous works aimed at adapting large pre-trained models effectively.

# 2.1 Image Segmentation

Image segmentation constitutes a fundamental discipline within computer vision, seeking to partition images into coherent and semantically meaningful regions to facilitate detailed analysis. Unlike conventional image classification, which assigns a single label to an entire image, segmentation operates at the pixel level, enabling a comprehensive understanding of visual content and spatial relationships. Segmentation tasks encompass several distinct paradigms such as semantic segmentation, instance segmentation, panoptic segmentation, and salient object detection. Semantic segmentation assigns a categorical label to each pixel in an image [34]. Instance segmentation [17] distinguishes individual instances of objects belonging to the same class and panoptic segmentation integrates the principles of both semantic and instance segmentation, yielding a unified pixelwise representation wherein every pixel is attributed either to a specific object instance or a background class. Salient object detection (SOD) [25] [4] concentrates on detecting visually prominent objects within a scene, typically generating binary masks that delineate regions most likely to attract human attention. In this work, we focus on semantic segmentation, which involves predicting a category label for each pixel. Among segmentation During the last years, many semantic segmentation methods have been designed [7, 9, 10, 23, 43].

Among them, the most relevant to our work is the ViTadapter [7], which concerns tailoring a Vision Transformer trained for ImageNet classification to handle dense tasks by leveraging a lightweight adapter network.

Thanks to the availability of large annotated RGB datasets, some foundational RGB segmentation models have been recently proposed [26, 35, 51]. In particular, the Segment Anything Model (SAM) [26] is designed for prompt-guided instance segmentation, i.e., it can segment objects and their parts in RGB images using prompts such as points, bounding boxes, and masks. Our work aims to rely on SAM’s foundational knowledge to pursue semantic segmentation from multiple modalities.

# 2.2 Multimodal Semantic Segmentation

Multimodal semantic segmentation leverages multiple modalities to address situations where individual sensors may fail [49]. This section focuses on methods that include RGB images as one of the modalities. A key design choice is deciding when to fuse RGB with other modalities. Early methods [12] perform channelwise concatenation of raw input modalities, which are then processed by a single modality-agnostic encoder, while later approaches typically fuse features at multiple levels within the encoder [16, 18, 24, 37, 40, 47, 48]. Recent state-of-the-art models [3, 22, 28] adopt a single late feature fusion stage downstream of the encoder before the decoder. Some methods process the input modalities with a shared modality-agnostic encoder [3, 22, 24, 27] while others with modality-specific encoders [14,18,28,47,48]. The machinery to combine features, namely the Fusion Module, has evolved over the years, from simple summation or concatenation to attention mechanisms [24,47]. Our MM Fusion Encoder deploys modality-specific encoders and late feature fusion. However, peculiar to our design, the purpose of the extracted fused multimodal features is to adjust SAM’s RGB features through an adapter network. The adapter enables the SAM backbone to incorporate auxiliary information at the encoder level, allowing it to leverage multimodal features for semantic segmentation.

# 2.3 SAM beyond RGB

In 2024, several studies extended SAM to tasks involving input modalities beyond RGB [8, 39, 42]. Chen et al. [8] distill the SAM backbone into a student network to perform semantic segmentation from event data. Wang et al. [39] propose a salient object detection framework that uses a fusion module as a prompt generator for the SAM decoder. Xiao et al. [42] introduce a multimodal SAM tailored to prompt-based instance segmentation. Yao et al. [45] propose a SAM-adapter for the RGB-Event framework, based on cross-attention and gated blocks, where event data are processed through a domain-specific backbone. Liu et al. [30] leverage the full SAM architecture for RGB-Depth and RGB-Event frameworks in the context of mask segmentation. Differently, we employ the SAM encoder for multimodal semantic segmentation.

# 2.4 Adapters

Adapters effectively transfer knowledge from powerful backbones to various tasks. Initially introduced in NLP [20] and later adapted for Computer Vision [44], they enhance a Transformer’s general knowledge by integrating external learnable modules. Among these methods, LoRA [21] is one of the earliest approaches designed for parameter-efficient fine-tuning and has also been extended in the vision domain. In particular, the ViT-adapter [7] aims to adapt Vision Transformers (ViT) pre-trained on image classification for dense prediction tasks. This architecture utilizes a convolutional encoder to extract spatial features, which are integrated with

![](images/630ae456a9e78f0d1d7f4ea2fa7ef00659d0562a2bb0fb5a1fa6be719f97a9f3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["RGB"] --> B["Patch Embedding"]
    B --> C["Fusion Module"]
    C --> D["Injector 1"]
    D --> E["Extractor 1"]
    E --> F["Injector N"]
    F --> G["Extractor N"]
    G --> H["SH"]
    I["Auxiliary Modality"] --> J["Modality-specific Encoder"]
    J --> K["Fusion Module"]
    K --> L["Fusion Module"]
    L --> M["Cross-attention"]
    M --> N["FFN"]
    O["Positional embedding"] --> P["Modality-specific Encoder"]
    P --> Q["Fusion Module"]
    Q --> R["Fusion Module"]
    R --> S["Cross-attention"]
    S --> T["FFN"]
    U["Element-wise addition"] --> V["Modality-specific Encoder"]
    V --> W["Fusion Module"]
    W --> X["Fusion Module"]
    X --> Y["Cross-attention"]
    Y --> Z["FFN"]
    style A fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
    style U fill:#f9f,stroke:#333
```
</details>

Figure 2: Overview of the MM SAM-adapter architecture. The top row shows the four main modules: SAM Encoder, MM Fusion Encoder, adapter and Segmentation Head (SH). The bottom row details the Modality-specific Encoders and the Fusion Module utilized by the MM Fusion Encoder as well as the Injector and Extractors modules utilized by the adapter.

ViT features through a series of injectors and extractors that maintain continuous interaction with the main backbone. Unlike standard adapters, which often freeze the backbone during training, the ViT-adapter fine-tunes the main ViT encoder to achieve optimal performance. Recently, the adapter mechanism has been explored to customize SAM to various downstream tasks [5, 6, 36, 41]. However, all previous works focus on adapting SAM to perform tasks that require processing only RGB images. In contrast, we are the first to apply an adapter network to adapt SAM for processing multimodal inputs. Our adapter mitigates the risk of catastrophic forgetting and enables the integration of multimodal features into the SAM model.

# 3 Materials and Method

# 3.1 MM SAM-adapter

Our method leverages the benefits of both fusing auxiliary modality features and adaptively integrating knowledge into the SAM backbone. Fusion allows the model to harness auxiliary modality information when the primary modality alone is insufficient for semantic segmentation, while the adaptation technique enables the SAM encoder to incorporate multimodal knowledge, mitigating the risk of catastrophic forgetting in the backbone. Notably, the overall encoder architecture is asymmetric, as the SAM backbone has a larger number of parameters compared to the combined adapter module and Multimodal Fusion Encoder. The scenario in which this parameter gap is reduced is analyzed in Section 4.3.2. Thus, our approach prioritizes the foundational knowledge embedded in the SAM RGB backbone, while exploiting the multimodal fused knowledge captured by the Multimodal Fusion Encoder as well. These advantages enable the model to achieve state-of-the-art performance across various benchmarks as discussed in Section 4 and to perform best in RGB-easy and RGB-hard scenarios. As shown in Figure 2, our multimodal semantic segmentation framework, MM SAM-adapter, comprises a main branch with the SAM Image Encoder, a secondary branch featuring a Multimodal Fusion Encoder and an adapter module, and a Segmentation Head (SH). The following sections provide a detailed description of these components.

# 3.1.1 SAM Encoder

Our goal is to leverage the segmentation-specific knowledge embedded in SAM’s weights. The overall SAM architecture consists of three main modules: an image encoder, a prompt encoder, and a decoder. As the prompt encoder and the decoder are tailored specifically for the prompt-guided instance segmentation task, we deploy only the image encoder in our multimodal semantic segmentation framework. The SAM image encoder is built upon a ViT architecture, which has been pre-trained in a self-supervised manner as a Masked AutoEncoder [19] and then fine-tuned on the SA-1B dataset to perform prompt-guided instance segmentation. We utilize the SAM encoder based on the ViT-Large (ViT-L) architecture, which includes ?? = 24 layers. As presented in Figure 2 and Section 3.1.3, akin to the architecture proposed in [7], we divide the encoder into ?? = 4 Blocks consisting of $\textstyle { \frac { L } { N } }$ layers, each Block interacting with the adapter by a pair of Injector-Extractor modules.

# 3.1.2 Multimodal Fusion Encoder

As discussed in Section 1, our method relies initially on a fusion stage, where RGB features are combined with auxiliary modality features, producing multimodal representations that are subsequently fed into the adapter module. To achieve this, we employ a Multimodal Fusion Encoder that processes the RGB image and the auxiliary measurements as inputs to generate fused features. Thereby, the Fusion Encoder can learn to weigh the contribution of the different modalities within the signal provided to the adapter, which, in turn, can learn how to integrate it with SAM’s features. Our Multimodal Fusion Encoder comprises three main modules: two modality-specific encoders and one fusion module, as described below.

Modality-specific Encoders The modality-specific encoders extract features independently from each of the two modalities to better handle their different nature: e.g., RGB images and LiDAR measurements are dense and sparse signals, respectively. Inspired by Vitadapter [7], we employ convolutional networks as they yield spatial features at multiple resolutions that help transformers better capture local spatial information. In particular, we employ ConvNext Small [32] pre-trained on ImageNet-22k [13] for both modality-specific encoders. To ensure compatibility, when the auxiliary signal has a single channel, we replicated it three times before feeding it to ConvNext. Hence, given an RGB image $I ^ { \mathrm { R G B } } \in \mathbb { R } ^ { \breve { H } \times W \times 3 }$ and a pixel-aligned auxiliary signal $\boldsymbol { I } ^ { \breve { \Chi } } \in \mathbb { R } ^ { H \times W \times 3 }$ , both encoders produce four different spatial features, namely $F _ { i } ^ { X }$ and $\dot { F } _ { i } ^ { \mathrm { R G B } }$ with $i = { 1 , 2 , 3 , 4 }$ for the RGB and auxiliary modality encoder, respectively. The feature tensor size at resolution ?? is $\begin{array} { r } { \frac { H } { 2 ^ { i + 1 } } \times \overset { \cdot } { 2 ^ { i + 1 } } \times \dot { D _ { i } } } \end{array}$ , where $D _ { i }$ denotes the number of channels (e.g., in ConvNext Small 96, 128, 384, 768).

Fusion Module The Fusion Module processes multiscale modality-specific features $( F _ { i } ^ { \mathrm { X } } , \dot { F } _ { i } ^ { \mathrm { R G B } } )$ and generates fused multi-scale outputs, ?? MM, $\dot { F } _ { i } ^ { \mathrm { M M } }$ for ?? $i = { 1 , 2 , 3 , 4 }$ . To align with SAM’s feature dimensions, we apply four linear projection layers to match the channel dimension ??. The fusion module must preserve information from both modalities to allow the adapter to dynamically select the relevant ones during inference, e.g., only the RGB information in optimal conditions. As shown in Section 4.3.3, to achieve this goal, we can employ a simple feature concatenation of the RGB and auxiliary features, which preserves each modality’s information. However, to obtain the best performance, we adopt the Road-Fusion module from Roadformer+ [22], which combines convolutional, self-attention, spatialattention, and coordinate-attention layers for advanced heterogeneous feature fusion.

# 3.1.3 Adapter

The adapter module takes as input the multimodal features produced by the Multimodal Fusion Encoder and incorporates that knowledge within the SAM backbone. Akin to the ViT-adapter [7], our adapter is a side-tuning network – a parallel branch that processes information alongside the main backbone. This strategy enables the adapter to learn domain-specific knowledge, in this case, multimodal information, while mitigating the risk of catastrophic forgetting in the main backbone. However, unlike the original proposal in [7], our adapter processes fused multimodal features rather than RGB-only features.

The adapter consists of a series of two modules: an injector, which introduces spatial multimodal knowledge to the SAM encoder, and an extractor, which retrieves hierarchical features from the SAM backbone.

Injector The ??MM, ??MM $F _ { 2 } ^ { \mathrm { M M } } , F _ { 3 } ^ { \mathrm { M M } }$ , and $F _ { 4 } ^ { \mathrm { M M } }$ multi-scale multimodal fused features are flattened and stacked channelwise to obtain $F ^ { M M , 1 }$ 1, a 2-D array of size $\begin{array} { r } { \big ( \frac { H W } { 8 ^ { 2 } } + \frac { H W } { 1 6 ^ { 2 } } + } \end{array}$ $\scriptstyle { \frac { H W } { 3 2 ^ { 2 } } } ) \times D$ $F ^ { \mathrm { M M , 1 } }$ 8 16  will be the input to the first Injector module. Each injector, Injector i, computes a crossattention between the SAM features, which are output of the block ?? − 1, ????SAM $i - 1 , F _ { \mathrm { S A M } } ^ { i }$ with dimensionality $\scriptstyle { \frac { H W } { 1 6 ^ { 2 } } } \times D$ , and $F ^ { \mathrm { M M } , i }$ , the multiscale multimodal fused features after being processed by ?? − 1 injectors and extractors. In particular, $F _ { \mathrm { S A M } } ^ { i }$ is taken as the query and $F ^ { \mathrm { M M } , i }$ is used as the key and value of a multi-scale deformable crossattention [50]. Hence, in our adapter, the computation performed by an Injector can be expressed as:

$$
\hat {F} _ {\mathrm{SAM}} ^ {i} = F _ {\mathrm{SAM}} ^ {i} + \gamma_ {i} \mathrm{Attn} (\Phi (F _ {\mathrm{SAM}} ^ {i}), \Phi (F ^ {\mathrm{MM}, i})) \qquad (1)
$$

where $\Phi ( \cdot )$ denotes the layer normalization operation [1], Attn(·) represents the multi-scale deformable crossattention, and $\gamma _ { i }$ is a ??-dimensional learnable vector initialized to zero to mitigate the impact of the crossattention at the beginning of the training process. The output from the ??-th injector is then processed by the ??-th SAM block, obtaining ????+1 $F _ { \mathrm { S A M } } ^ { i + 1 }$ with dimensionality $\scriptstyle { \frac { H W } { 1 6 ^ { 2 } } } \times D$ .

Extractor This component operates on the output of SAM’s blocks. The multi-scale deformable cross-$F _ { \mathrm { S A M } } ^ { i + 1 }$ is computed between SAM’s features, key and values, and the multi-scale fused $F ^ { \mathrm { M M } , i }$ the cross-attention, $\hat { F } ^ { \mathrm { M M , \it i } }$ , are fed to a feed-forward network (FFN), obtaining the final features $F ^ { \mathrm { M M , \it i + 1 } }$ , which will be the input to the next injector module. Hence, the computation performed by an extractor of our adapter can be expressed as:

$$
\hat {F} ^ {\mathrm{MM}, i} = F ^ {\mathrm{MM}, i} + \operatorname{Attn} \left(\Phi \left(F ^ {\mathrm{MM}, i}\right), \Phi \left(F _ {\mathrm{SAM}} ^ {i + 1}\right)\right) \tag {2}
$$

$$
F ^ {\mathrm{MM}, i + 1} = \hat {F} ^ {\mathrm{MM}, i} + F F N (\Phi (\hat {F} ^ {\mathrm{MM}, i})) \tag {3}
$$

# 3.1.4 Segmentation Head

We obtain the predictions by a Segmentation Head (SH) implemented as a Segformer [43] decoder. It processes the multimodal fused features, $\bar { F } _ { 1 } ^ { \mathrm { M M } }$ , with the adapter tokenized output, $F ^ { \mathrm { M M } , N }$ , which is reshaped to obtain feature maps ??MM,?? , $F _ { i } ^ { \mathrm { M M } , N } , i = 2 , 3$ , 4 at their original input resolutions.

![](images/a1736cd21c1fdc3b37c4f315f0533713b21567547d0e0c56305e53cd9c814f95.jpg)

<details>
<summary>text_image</summary>

RGB-EASY
RGB-HARD
DeLiVER
FMB
</details>

Figure 3: RGB-easy and RGB-hard examples for DeLiVER and FMB. The DeLiVER RGB-hard image has been stretched to enhance visibility (zoom in to better notice it).

# 3.2 Datasets

We employ the following multimodal semantic segmentation datasets in our experiments:

DeLiVER [48] is a synthetic dataset comprising 3,983 training, 2,005 validation, and 1,897 test samples. It includes LiDAR, Depth, Event, and RGB data from urban environments with diverse lighting and weather conditions. The tensors containing the auxiliary modalities have the same spatial dimensions as the RGB images (1042×1042), with data from all modalities being pixelwise aligned. The number of classes in DeLiVER is 25.

FMB [29] contains 1,500 pixel-aligned RGB-Thermal image pairs, 280 of which serve as the test set. The image pairs are acquired in urban driving scenes with different illumination and weather conditions and have spatial dimensions 800×600. The number of classes is 14. To select the best checkpoint for all models, we create a validation set by randomly extracting 160 samples from the training set.

MUSES [2] consists of 1,500 training, 250 validation and 750 test samples collected under various daytime and weather conditions. It includes LiDAR, Event, Radar, and RGB data, though we excluded Radar due to its sparsity and insufficient information for multimodal segmentation. We align the data from all modalities at pixel level by using the official code [2]. MUSES employs the same 19 classes as the RGB-only Cityscapes dataset [11]. As the test annotations are withheld, test set evaluations are possible only through online submissions.

# 3.2.1 RGB-easy and RGB-hard splits

As highlighted in Section 1, most samples in multimodal datasets are easily segmented by leveraging exclusively RGB information. For instance, from our experiments, approximately 97% of DeLiVER sample can be optimally segmented using solely the RGB image, while only the remaining 3% represents challenging scenarios in which the auxiliary information may be useful. To better assess the methods’ capability to exploit information from multiple modalities synergistically, we divided the DeliVER and FMB test sets into an RGB-hard and RGB-easy split. Purposely, we first train an RGB-only network on each multimodal dataset to obtain semantic predictions. Then, we visually compare the RGB-only predictions to the ground-truths while paying attention to the information conveyed by the auxiliary modality. Thereby, we identify as RGB-hard those samples in which the RGB-only network fails to recognize elements that may be correctly detected by another modality, and as RGB-easy all other samples. Accordingly, the De-LiVER test set is divided into 1797 RGB-easy and 100 RGB-hard samples, whereas for FMB we get 183 and 97 samples for the RGB-easy and RGB-hard splits. Figure 3 shows samples for the RGB-easy and RGB-hard splits of DeLiVER and FMB. We point out that we adopt a similar procedure to build a balanced validation set of FMB, i.e., we randomly select 80 RGB-easy and 80 RGB-hard samples from the original training set. We did not perform a similar test set split on MUSES [2] as the ground truths are unavailable. Yet, we report results for all the official splits (day, night, different weather conditions). In particular, this disentangled evaluation allows us to assess the behaviour of the methods in the fog and nighttime scenarios – those that pose more challenges for the RGB modality, as highlighted in the original paper [2].

# 3.3 Implementation Details

# 3.3.1 Architectural Details

The architectural specifications are summarized in Table 1 and Table 2, which offers a detailed overview of our network design. A legend explaining the abbreviations used in Table 1 and Table 2 is provided in Table 3.

The Fusion Module, as detailed in Table 1 and employed throughout this paper, is based on the Road-Former+ fusion block [22] (Road-fusion module). This module comprises several key components, including two Global Feature Enhancers (GFEs), two Local Feature Enhancers (LFEs), one Global Feature Recalibration Module (GFRM), one Local Feature Fusion Module (LFFM) and one Feature Enhancement and Integration Module (FEIM). The Road-fusion block initially extracts global and local features separately from RGB features and auxiliary modality features, both originating from the two Modality-Specific Encoders, by employing GFEs and LFEs. The former are a transformer-based modules, while the latter are convolution-based. Subsequently, the global RGB features and auxiliary modality features are fused using the GFRM. This module per-

Table 1: Details of our architecture (SAM Encoder, Modality-Specific Encoders, Fusion Module). ?? and ?? represents the Height and Width of the input image, ?? ∈ [1, 4], ?? = 4 and $N _ { \mathrm { c l a s s e s } }$ is the number of classes. SAM-L encoder has an embedding dimension of 1024. 

<table><tr><td>Stage</td><td>Layer Type</td><td>Input</td><td>Output</td><td>Output Shape</td></tr><tr><td colspan="5">SAM Encoder</td></tr><tr><td>Input</td><td>Image</td><td>-</td><td> $I^{RGB}$ </td><td> $H \times W \times 3$ </td></tr><tr><td>Emb</td><td>Patch E., Pos. E.</td><td> $I^{RGB}$ </td><td> $F_{1}^{SAM}$ </td><td> $\frac{HW}{16^{2}} \times 1024$ </td></tr><tr><td>Block i</td><td>-</td><td> $\hat{F}_{i}^{SAM}$ </td><td> $F_{i+1}^{SAM}$ </td><td> $\frac{HW}{16^{2}} \times 1024$ </td></tr><tr><td colspan="5">Modality-Specific Encoders</td></tr><tr><td>Input</td><td>ImageAux. mod.</td><td>-</td><td> $I^{RGB}$  $I^{X}$ </td><td> $H \times W \times 3$  $H \times W \times 3$ </td></tr><tr><td>RGB Enc.</td><td>ConvNeXt</td><td> $I^{RGB}$ </td><td> $F_{1}^{RGB}$  $F_{2}^{RGB}$  $F_{3}^{RGB}$  $F_{4}^{RGB}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 96$  $\frac{H}{8} \times \frac{W}{8} \times 192$  $\frac{H}{16} \times \frac{W}{16} \times 384$  $\frac{H}{32} \times \frac{W}{32} \times 768$ </td></tr><tr><td>X Enc.</td><td>ConvNeXt</td><td> $I^{X}$ </td><td> $F_{1}^{X}$  $F_{2}^{X}$  $F_{3}^{X}$  $F_{4}^{X}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 96$  $\frac{H}{8} \times \frac{W}{8} \times 192$  $\frac{H}{16} \times \frac{W}{16} \times 384$  $\frac{H}{32} \times \frac{W}{32} × 768$ </td></tr><tr><td colspan="5">Fusion module</td></tr><tr><td>GFE-RGB</td><td>Patch E.,MHSA, LN</td><td> $F_{1}^{RGB}$  $F_{2}^{RGB}$  $F_{3}^{RGB}$  $F_{4}^{RGB}$ </td><td> $F_{1}^{RGB,G}$  $F_{2}^{RGB,G}$  $F_{3}^{RGB,G}$  $F_{4}^{RGB,G}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 96$  $\frac{H}{8} \times \frac{W}{8} \times 192$  $\frac{H}{16} \times \frac{W}{16} \times 384$  $\frac{H}{32} \times \frac{W}{32}× 768$ </td></tr><tr><td>GFE-X</td><td>Patch E.,MHSA, LN</td><td> $F_{1}^{X}$  $F_{2}^{X}$  $F_{3}^{X}$  $F_{4}^{X}$ </td><td> $F_{1}^{X,G}$  $F_{2}^{X,G}$  $F_{3}^{X,G}$  $F_{4}^{X,G}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 96$  $\frac{H}{8} \times \frac{W}{8} \times 192$  $\frac{H}{16} \times \frac{W}{16} \times 384$  $\frac{H}{32} \times \frac{W}{32}\times 768$ </td></tr><tr><td>LFE-RGB</td><td>Conv $_{1\times 1}$ , RL,DWC $_{3\times 3}$ , RL,Conv $_{1\times 1}$ </td><td> $F_{1}^{RGB}$  $F_{2}^{RGB}$  $F_{3}^{RGB}$  $F_{4}^{RGB}$ </td><td> $F_{1}^{RGB,L}$  $F_{2}^{RGB,L}$  $F_{3}^{RGB,L}$  $F_{4}^{RGB,L}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 96$  $\frac{H}{8} \times \frac{W}{8} \times 192$  $\frac{H}{16} \times \frac{W}{16} \times 384$  $\frac{H}{32} \times \frac{W}{32}\text{×} 768$ </td></tr><tr><td>LFE-X</td><td>Conv $_{1\times 1}$ , RL,DWC $_{3\times 3}$ , RL,Conv $_{1\times 1}$ </td><td> $F_{1}^{X}$  $F_{2}^{X}$  $F_{3}^{X}$  $F_{4}^{X}$ </td><td> $F_{1}^{X,L}$  $F_{2}^{X,L}$  $F_{3}^{X,L}$  $F_{4}^{X,L}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 96$  $\frac{H}{8} \times \frac{W}{8} \times 192$  $\frac{H}{16} \times \frac{W}{16} \times 384$  $\frac{H}{32} \times \frac{W}{32}x 768$ </td></tr><tr><td>GFRM</td><td>CA RGB-X,CA X-RGB,Concat., LN,Av. Pool., Sigm.</td><td> $F_{1}^{RGB,G},F_{1}^{X,G}$  $F_{2}^{RGB,G},F_{2}^{X,G}$  $F_{3}^{RGB,G},F_{3}^{X,G}$  $F_{4}^{RGB,G},F_{4}^{X,G}$ </td><td> $F_{1}^{MM,G}$  $F_{2}^{MM,G}$  $F_{3}^{MM,G}$  $F_{4}^{MM,G}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 192$  $\frac{H}{8} \times \frac{W}{8} \times 384$  $\frac{H}{16} \times \frac{W}{16} \times 768$  $\frac{H}{32} \times \frac{W}{32} \times 1536$ </td></tr><tr><td>LFFM</td><td>Concat.,Conv $_{1\times 1}$ DWC $_{3\times 3}$ , GL,Conv $_{1\times 1}$ </td><td> $F_{1}^{RGB,L},F_{1}^{X,L}$  $F_{2}^{RGB,L},F_{2}^{X,L}$  $F_{3}^{RGB,L},F_{3}^{X,L}$  $F_{4}^{RGB,L},F_{4}^{X,L}$ </td><td> $F_{1}^{MM,L}$  $F_{2}^{MM,L}$  $F_{3}^{MM,L}$  $F_{4}^{MM,L}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 192$  $\frac{H}{8} \times \frac{W}{8} \times 384$  $\frac{H}{16} \times \frac{W}{16} \times 768$  $\frac{H}{32} \times \frac{W}{32}\text{×} 1536$ </td></tr><tr><td>FEIM</td><td>W. Feat. Sum,Coord. A.</td><td> $F_{1}^{MM,G},F_{1}^{MM,L}$  $F_{2}^{MM,G},F_{2}^{MM,L}$  $F_{3}^{MM,G},F_{3}^{MM,L}$  $F_{4}^{MM,G},F_{4}^{MM,L}$ </td><td> $F_{1}^{MM,F}$  $F_{2}^{MM,F}$  $F_{3}^{MM,F}$  $F_{4}^{MM,F}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 192$  $\frac{H}{8} \times \frac{W}{8} \times 384$  $\frac{H}{16} \times \frac{W}{16} \times 768$  $\frac{H}{32} \times \frac{W}{32 } \times 1536$ </td></tr></table>

forms cross-attention operations between the two modalities: one operation utilizes RGB modality features as queries and auxiliary modality features as keys/values, whereas the other operation reverses these roles. The resulting global feature representations are then concatenated, and the obtained multimodal representation is further refined through an average pooling layer followed by a sigmoid activation function. In parallel with the global features extraction, the local features from both modalities are fused using the LFFM. This module first performs a channel-wise concatenation, after which the resulting feature map is processed through a series of convolutional layers employing one GeLU activation function. Subsequently, local and global fused features are combined through the GFRM. This module employs coordinate attention to effectively encode spatial relationships within the features, enhancing their representational capacity. The resulting attention-enhanced representation is further refined through a series of con-

Table 2: Details of our architecture (Projector, Injector, Extractor, Feature Refinement, Segmentation Head). ?? and ?? represent the Height and Width of the input image, ?? ∈ [1, 4], ?? = 4 and $N _ { \mathrm { c l a s s e s } }$ is the number of classes. SAM-L encoder has an embedding dimension of 1024. 

<table><tr><td>Stage</td><td>Layer Type</td><td>Input</td><td>Output</td><td>Output Shape</td></tr><tr><td colspan="5">Projector</td></tr><tr><td>S-to-V</td><td> $Conv_{1\times 1}$ ,Flatten</td><td> $F_{1}^{MM,F}$  $F_{2}^{MM,F}$  $F_{3}^{MM,F}$  $F_{4}^{MM,F}$ </td><td> $F_{1}^{MM}$  $F_{2}^{MM}$  $F_{3}^{MM}$  $F_{4}^{MM}$ </td><td> $\frac{HW}{4^2} \times 1024$  $\frac{HW}{8^2} \times 1024$  $\frac{HW}{16^2} \times 1024$  $\frac{HW}{32^2} \times 1024$ </td></tr><tr><td>Feat. Stack.</td><td>Concat.</td><td> $F_{2}^{MM}$  $F_{3}^{MM}$  $F_{4}^{MM}$ </td><td> $F^{MM,1}$ </td><td> $(\frac{HW}{8^2} + \frac{HW}{16^2} + \frac{HW}{32^2}) \times 1024$ </td></tr><tr><td colspan="5">Injector i</td></tr><tr><td>Q-K Inter</td><td>MSDAW. Feat. Sum</td><td> $F_{i}^{SAM},F^{MM,i}$ </td><td> $\hat{F}_{i}^{SAM}$ </td><td> $\frac{HW}{16^2} \times 1024$ </td></tr><tr><td colspan="5">Extractor i</td></tr><tr><td>Q-K Inter</td><td>MSDA</td><td> $F^{MM,i},F_{i+1}^{SAM}$ </td><td> $\hat{F}^{MM,i}$ </td><td> $(\frac{HW}{8^2} + \frac{HW}{16^2} + \frac{HW}{32^2}) \times 1024$ </td></tr><tr><td>FFN</td><td>Lin.,  $DWC_{3\times 3}$ GL, Lin.</td><td> $\hat{F}^{MM,i}$ </td><td> $F^{MM,i+1}$ </td><td> $(\frac{HW}{8^2} + \frac{HW}{16^2} + \frac{HW}{32^2}) \times 1024$ </td></tr><tr><td colspan="5">Feature Refinement</td></tr><tr><td>Slice</td><td>Reshape</td><td> $F^{MM,N}$ </td><td> $F_{2}^{MM,N}$  $F_{3}^{MM,N}$  $F_{4}^{MM,N}$ </td><td> $\frac{H}{8} \times \frac{W}{8} \times 1024$  $\frac{H}{16} \times \frac{W}{16} \times 1024$  $\frac{H}{32} \times \frac{W}{32} \times 1024$ </td></tr><tr><td colspan="5">Segmentation Head</td></tr><tr><td>Prep.</td><td>Tran. Conv. &amp; Sum,Feature Sum</td><td> $F_{2}^{MM,N},F_{1}^{MM}$  $F_{1}^{MM,N},F_{N}^{SAM}$  $F_{2}^{MM,N},F_{N}^{SAM}$  $F_{3}^{MM,N},F_{N}^{SAM}$  $F_{4}^{MM,N},F_{N}^{SAM}$ </td><td> $F_{1}^{MM,N}$  $F_{1,mix}^{MM,N}$  $F_{2,mix}^{MM,N}$  $F_{3,mix}^{MM,N}$  $F_{4,mix}^{MM,N}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 1024$  $\frac{H}{4} \times \frac{W}{4} \times 1024$  $\frac{H}{8} \times \frac{W}{8} \times 1024$  $\frac{H}{16} \times \frac{W}{16} \times 1024$  $\frac{H}{32} \times \frac{W}{32} \times 1024$ </td></tr><tr><td>up</td><td> $Conv_{1\times 1}$ ,BN, RL,Interp.,Concat.</td><td> $F_{1,mix}^{MM,N}$  $F_{2,mix}^{MM,N}$  $F_{3,mix}^{MM,N}$  $F_{4,mix}^{MM,N}$ </td><td> $F_{up}^{MM}$ </td><td> $\frac{H}{4} \times \frac{W}{4} \times 512$ </td></tr><tr><td>Pred.</td><td> $Conv_{1\times 1}$ ,BN, RL, $Conv_{1\times 1}$ ,Interp.</td><td> $F_{up}^{MM}$ </td><td> $F_{Seg}^{MM}$ </td><td> $H \times W \times N_{classes}$ </td></tr></table>

volutional layers. The output of the fusion module is subsequently passed through a series of projection layers to align the channel dimensions of the fused features with those of SAM.

The Segmentation Head (SH) will take as input $F _ { 1 } ^ { \mathrm { M M } }$ and the adapter output, $F ^ { \mathrm { M M } , N }$ reshaped into $F _ { i } ^ { \mathrm { M M } , N } , i = 2 , 3 , 4$ . The $F _ { 1 } ^ { \mathrm { M M } }$ is summed with $F _ { 2 } ^ { \mathbf { M M } , N }$ , which has been upsampled using transpose convolution [46] to match the spatial resolution, resulting in $F _ { 1 } ^ { \mathrm { M M } , N }$ . Each spatial feature is subsequently enriched by summing it with the SAM tokens from the final layer, which are first interpolated to match the spatial resolution. This results in the mixed feature representation ??MM,?? $F _ { i , \operatorname* { m i x } } ^ { \mathrm { M M } , N }$ , where $i = 2 , 3 , 4 .$ . Subsequently, the Segformer head [43] processes each spatial feature ??MM,?? $F _ { i , \operatorname* { m i x } } ^ { \mathrm { M M } , N }$ ??,mix , using distinct MLPs, composed of $1 \times 1$ convolutional layers, batch normalization layers and ReLU activation functions. A channel-wise concatenation operation with bilinear interpolatiowith dimensionality plied, yielding . Finally, in th $F _ { \mathrm { u p } } ^ { \mathrm { M M } }$ $\begin{array} { r } { \frac { H } { 4 } \times \frac { W } { 4 } \times 5 1 2 } \end{array}$ stage, another MLP is used to generate the segmentation prediction, with dimensionality $H \times W \times N _ { \mathrm { c l a s s e s } }$ , where $N _ { \mathrm { c l a s s e s } }$ denotes the number of classes.

Table 3: Legend of abbreviations used in the Table 1 and Table 2 

<table><tr><td>Abbreviation</td><td>Meaning</td></tr><tr><td>Emb</td><td>Embedding</td></tr><tr><td>Patch E.</td><td>Patch Embedding</td></tr><tr><td>Pos. E.</td><td>Positional Embedding</td></tr><tr><td>Aux mod</td><td>Auxiliary Modality</td></tr><tr><td>RGB Enc.</td><td>RGB Encoder</td></tr><tr><td>X Enc.</td><td>X Encoder</td></tr><tr><td>GFE</td><td>Global Feature Enhancer</td></tr><tr><td>MHSA</td><td>Multi-Head Self-Attention</td></tr><tr><td>LN</td><td>Layer normalization</td></tr><tr><td>Concat.</td><td>Concatenation</td></tr><tr><td>LFE</td><td>Local Feature Enhancer</td></tr><tr><td> $DWC_{3\times3}$ </td><td>Depth-wise Separable Convolution</td></tr><tr><td> $Conv_{1\times1}$ </td><td>1 × 1 Convolution</td></tr><tr><td>GFRM</td><td>Global Feature Recalibration Module</td></tr><tr><td>Sigm.</td><td>Sigmoid</td></tr><tr><td>Av. Pool.</td><td>Average Pooling</td></tr><tr><td>CA</td><td>Cross-Attention</td></tr><tr><td>LFFM</td><td>Local Feature Fusion Module</td></tr><tr><td>FEIM</td><td>Feature Enhancement and Integration Module</td></tr><tr><td>W. Feat. Sum</td><td>Weigthed Feature Sum</td></tr><tr><td>Coord. A.</td><td>Coordinate Attention</td></tr><tr><td>MSDA</td><td>Multi-Scale Deformable Attention</td></tr><tr><td>BN</td><td>Batch Normalization</td></tr><tr><td>Prep.</td><td>Preprocessing</td></tr><tr><td>Up</td><td>Upsampling</td></tr><tr><td>Pred.</td><td>Prediction</td></tr><tr><td>Tran. Conv.</td><td>Transpose Convolution</td></tr><tr><td>Q-K Inter</td><td>Query-Key Interaction</td></tr><tr><td>S-to-V</td><td>Spatial-to-Vector</td></tr><tr><td>Feat. Stack.</td><td>Feature Stacking</td></tr><tr><td>Interp.</td><td>Interpolation</td></tr><tr><td>Lin.</td><td>Linear</td></tr><tr><td>RL</td><td>ReLU</td></tr><tr><td>GL</td><td>GeLU</td></tr></table>

# 3.3.2 Training Details

Our method was trained for 100 epochs using the OHEM cross-entropy loss on two NVIDIA RTX 3090 GPUs with batch size $8 - { \mathrm { w e } }$ employ gradient accumulation for memory constraints –, a base learning rate of $2 e ^ { - 4 }$ , which follows a polynomial strategy with a power of 0.9, and exponential warm-up with base 0.1. Therefore, the learning rate evolves according to the following formula:

$$
\eta (p) = \left\{ \begin{array}{l l} \eta_ {\text { base }} \cdot (\mathrm{wr}) ^ {1 - \frac {p}{N _ {\mathrm{w}}}}, & \text { if } p \leq N _ {\mathrm{w}}, \\ \left(\eta_ {\text { base }} - \eta_ {\min}\right) \cdot \left(1 - \frac {p}{P _ {\max}}\right) ^ {\alpha} + \eta_ {\min}, & \text { if } p > N _ {\mathrm{w}}. \end{array} \right. \tag {4}
$$

In the learning rate schedule described above, $\eta ( p )$ denotes the learning rate at epoch or iteration $p .$ The parameter $\eta _ { \mathrm { b a s e } }$ represents the base learning rate that the schedule aims to reach at the end of the warm-up phase $( 2 e ^ { - 4 } )$ , while $\eta _ { \mathrm { m i n } }$ defines the minimum learning rate that serves as a lower bound throughout training $( \eta _ { \mathrm { m i n } } = 0 )$ . During the warm-up phase, which lasts for $N _ { \mathrm { w } }$ epochs $( N _ { \mathrm { w } } = 1 0 )$ , the learning rate increases exponentially from an initial value determined by scaling $\eta _ { \mathrm { b a s e } }$ by the factor wr, which represent the warm-up ratio $( \mathrm { w r } = 0 . 1 )$ . Specifically, at the beginning of training, the learning rate is set to $\eta _ { \mathrm { b a s e } } \times \mathrm { w r }$ , and it grows smoothly and exponentially toward $\eta _ { \mathrm { b a s e } }$ as training progresses through the warm-up steps. After the warm-up period, the learning rate follows a polynomial decay controlled by the exponent ?? $( \alpha = 0 . 9 )$ , which determines how sharply the learning rate decreases. The variable $P _ { \mathrm { m a x } }$ specifies the total number of epochs in the training process, ensuring that the schedule spans the entire duration of learning without abrupt transitions.

The optimizer used during training is AdamW [33] with a weight decay of $1 e ^ { - 2 }$ . We use a layer-wise learning rate decay of 0.9, as in ViT-adapter architecture [7]. In this approach, the learning rate assigned to each layer decreases exponentially as one moves from the higher layers toward the lower layers of the backbone. The learning rate for the parameters in layer ℓ is defined as:

$$
\eta_ {\ell} = \eta_ {\text { curr }} \times \gamma^ {L - \ell - 1}, \tag {5}
$$

where $\eta _ { \mathrm { c u r r } }$ denotes the learning rate assigned to the highest layer, $\gamma$ is the decay rate $( \gamma = 0 . 9 )$ , ?? is the total number of layers of the transformer backbone, and ℓ is the index of the current layer, starting from zero.

Parameters associated with embeddings, special tokens, and normalization layers are assigned to layer zero and are typically exempted from weight decay. Conversely, the newly introduced layers in the architecture, such as adapter modules and decoding heads, are trained with higher learning rates to encourage rapid adaptation to the downstream task. This parameter grouping ensures that lower layers of the backbone, which encode generic visual features learned during pre-training, are updated more conservatively, thereby preserving their valuable representations. In order to improve robustness of our model, we perform online data augmentations based on random resize with a ratio from 0.5 to 2.0, random horizontal flipping, photometric distortion, random Gaussian blur with probability $\scriptstyle p = 0 . 2$ , and random crop to 1024 × 1024 resolution for DeLiVER and MUSES, $8 0 0 \times 8 0 0$ for FMB. Our method on FMB has been trained for 200

Table 4: DeLiVER test set results in the RGB-Depth (RGB-D), RGB-LiDAR (RGB-L), and RGB-Event (RGB-E) setups. 

<table><tr><td rowspan="2">Method</td><td colspan="3">All</td><td colspan="3">RGB-easy</td><td colspan="3">RGB-hard</td></tr><tr><td>RGB-D</td><td>RGB-L</td><td>RGB-E</td><td>RGB-D</td><td>RGB-L</td><td>RGB-E</td><td>RGB-D</td><td>RGB-L</td><td>RGB-E</td></tr><tr><td>CMNeXt [48]</td><td>53.87</td><td>51.32</td><td>50.81</td><td>54.07</td><td>51.97</td><td>51.56</td><td>49.66</td><td>39.29</td><td>37.08</td></tr><tr><td>GeminiFusion [24]</td><td>54.98</td><td>50.57</td><td>51.71</td><td>55.07</td><td>51.07</td><td>52.32</td><td>53.00</td><td>40.93</td><td>40.27</td></tr><tr><td>RoadFormer+ [22]</td><td>55.95</td><td>54.56</td><td>54.10</td><td>56.12</td><td>55.41</td><td>54.80</td><td>52.39</td><td>40.26</td><td>41.98</td></tr><tr><td>MM SAM-adapter</td><td>57.35</td><td>57.14</td><td>55.70</td><td>57.62</td><td>57.75</td><td>56.29</td><td>53.35</td><td>45.46</td><td>44.67</td></tr></table>

epochs.

# 3.3.3 Competitor Details

For competitors evaluation, we use official weights when available; otherwise, we train them following the official paper guidelines. In the DeLiVER benchmark, CMNeXt and GeminiFusion were evaluated using the official weights, which are based on the most effective backbone, MiT-B2 [43]. In contrast, we trained Road-Former+ from scratch, as no pre-trained weights were available, using ConvNeXt-Large, the best-performing configuration reported in the paper. For the FMB [29] and MUSES [2] datasets, we had to train all competitors due to the unavailability of pre-trained weights. Specifically, for FMB, we followed the official guidelines to ensure optimal performance, training CMNeXt with the MiT-B5 [43] backbone, GeminiFusion with Swin Transformer Large [31], and RoadFormer+ with ConvNeXt-Large [32]. Regarding MUSES, since it was recently published with no available training guidelines, we adopted the same configuration as DeLiVER, given the similarity between the datasets in terms of modalities and high-resolution images. Consequently, we trained CMNeXt and GeminiFusion with the MiT-B2 backbone and RoadFormer+ with the ConvNeXt-Large backbone. CAFuser [3], according to its authors, is inherently a multimodal model. Moreover, its training requires an additional text prompt describing the condition of the scene, which is not provided in FMB, but can be implicitly inferred from DeLiVER and MUSES. Thus, the results are reported following the original paper’s settings.

We highlight that, differently from most competitors that employ different backbones depending on the dataset to achieve the best performance, such as CM-NeXt [48] and GeminiFusion [24], our framework remains consistent across datasets yet achieves superior performance.

# 4 Results

This section compares MM SAM-adapter with state-ofthe-art multimodal semantic segmentation models such as CMNext [48], GeminiFusion [24], RoadFormer+ [22], and CAFuser [3]. We use Mean Intersection Over Union (mIoU) as the semantic segmentation evaluation metric. In addition, we analyze the design choices underlying the proposed multimodal adapter.

Table 5: FMB test set results in the RGB-Thermal (RGB-T) setup across different scenarios. 

<table><tr><td>Method</td><td>All RGB-T</td><td>RGB-easy RGB-T</td><td>RGB-hard RGB-T</td></tr><tr><td>CMNeXt [48]</td><td>61.66</td><td>64.81</td><td>56.85</td></tr><tr><td>GeminiFusion [24]</td><td>64.75</td><td>68.05</td><td>61.03</td></tr><tr><td>RoadFormer+ [22]</td><td>64.57</td><td>66.31</td><td>61.79</td></tr><tr><td>MM SAM-adapter</td><td>66.10</td><td>68.45</td><td>62.59</td></tr></table>

# 4.1 Main results

Table 4 reports the results on the DeLiVER test set in three multimodal setups: RGB-Depth (RGB-D), RGB-LiDAR (RGB-L), and RGB-Events (RGB-E). By analyzing the performance in the RGB-hard case, we note that methods are more effective when employing the Depth auxiliary modality rather than Event or LiDAR data. We ascribe it to the perfect synthetic Depth maps provided with DeLiVER, which convey extremely rich auxiliary information. In this RGB-D scenario, the improvements over competitors are marginal (e.g., 53.35 vs. 53.00 mIoU of our method vs. GeminiFusion). However, our method shines when the auxiliary modality is more noisy and realistic, such as in the RGB-L and RGB-E setups. In these cases, we achieve large performance improvements over competitors (e.g., 45.46 vs. 40.93 mIoU of our method vs. GeminiFusion in the RGB-L scenario). Finally, by examining our method’s results in the RGB-easy scenario across the RGB-D, RGB-L, and RGB-E setups, we note that it achieves very similar performance independently of the auxiliary modality (i.e., 57.62, 57.75, 56.29 mIoU in the RGB-D, RGB-L, and RGB-E setups, respectively). Conversely, other methods generally perform less consistently (e.g., CMNeXt obtains 54.07, 51.97, and 51.56 in the RGB-D, RGB-L, and RGB-E setups, respectively). In general, these results highlight how our method more effectively utilizes the synergies between modalities, e.g., in the case of good RGB images, the auxiliary modality contribute less to final results. Our method excels at fusing multimodal features and seamlessly adapting the SAM RGB backbone to incorporate this information. As can be seen from these results, our method establishes a new state-of-the-art on the DeLiVER test set in the RGB-D, RGB-L, and RGB-E setups, and achieves remarkable performance in both RGB-easy and RGB-hard scenarios.

Table 6: MUSES test results in the RGB-LiDAR (RGB-L) and RGB-Event (RGB-E) setups for different weather conditions. 

<table><tr><td rowspan="2">Method</td><td colspan="2">All</td><td colspan="2">Day</td><td colspan="2">Night</td><td colspan="2">Clear</td><td colspan="2">Fog</td><td colspan="2">Rain</td><td colspan="2">Snow</td></tr><tr><td>RGB-L</td><td>RGB-E</td><td>RGB-L</td><td>RGB-E</td><td>RGB-L</td><td>RGB-E</td><td>RGB-L</td><td>RGB-E</td><td>RGB-L</td><td>RGB-E</td><td>RGB-L</td><td>RGB-E</td><td>RGB-L</td><td>RGB-E</td></tr><tr><td>CMNeXt [48]</td><td>72.36</td><td>70.49</td><td>75.42</td><td>74.02</td><td>64.27</td><td>60.72</td><td>72.81</td><td>71.24</td><td>59.98</td><td>60.91</td><td>72.48</td><td>68.78</td><td>71.92</td><td>70.54</td></tr><tr><td>GeminiFusion [24]</td><td>74.22</td><td>68.62</td><td>75.90</td><td>72.04</td><td>67.67</td><td>58.19</td><td>74.79</td><td>70.40</td><td>62.95</td><td>60.53</td><td>73.30</td><td>65.27</td><td>74.16</td><td>68.87</td></tr><tr><td>RoadFormer+ [22]</td><td>80.38</td><td>77.70</td><td>82.56</td><td>80.62</td><td>74.51</td><td>69.02</td><td>79.39</td><td>78.44</td><td>69.64</td><td>69.29</td><td>80.69</td><td>76.40</td><td>79.49</td><td>77.70</td></tr><tr><td>MM SAM-adapter</td><td>81.07</td><td>79.92</td><td>83.34</td><td>83.39</td><td>74.97</td><td>72.38</td><td>80.82</td><td>81.52</td><td>74.12</td><td>68.97</td><td>80.00</td><td>78.92</td><td>80.85</td><td>78.72</td></tr></table>

Table 7: Comparison with all-modalities competitors on the DELIVER test (Test mIoU) and validation (Val mIoU) sets. 

<table><tr><td>Method</td><td>Modalities</td><td>Val mIoU</td><td>Test mIoU</td></tr><tr><td>CMNeXt [48]</td><td>RGB, Depth, LiDAR, Event</td><td>66.30</td><td>53.00</td></tr><tr><td>GeminiFusion [24]</td><td>RGB, Depth, LiDAR, Event</td><td>66.90</td><td>54.46</td></tr><tr><td>CAFuser [3]</td><td>RGB, Depth, LiDAR, Event</td><td>67.80</td><td>55.60</td></tr><tr><td>MM SAM-adapter</td><td>RGB, LiDAR</td><td>61.89</td><td>57.14</td></tr><tr><td>MM SAM-adapter</td><td>RGB, Event</td><td>60.74</td><td>55.70</td></tr><tr><td>MM SAM-adapter</td><td>RGB, Depth</td><td>69.60</td><td>57.35</td></tr></table>

Table 8: Comparison with all-modalities competitors on the MUSES test set. 

<table><tr><td>Method</td><td>Modalities</td><td>mIoU</td></tr><tr><td>CMNeXt</td><td>RGB, LiDAR, Event, Radar</td><td>72.40</td></tr><tr><td>GeminiFusion</td><td>RGB, LiDAR, Event, Radar</td><td>75.30</td></tr><tr><td>CAFuser</td><td>RGB, LiDAR, Event, Radar</td><td>78.18</td></tr><tr><td>MM SAM-adapter</td><td>RGB, Event</td><td>79.92</td></tr><tr><td>MM SAM-adapter</td><td>RGB, LiDAR</td><td>81.07</td></tr></table>

Table 5 shows the results on the FMB test set, which features RGB and Thermal images (RGB-T). Remarkably, our method achieves state-of-the-art performance also in this real dataset. We note a significant performance drop between RGB-easy and RGB-hard samples, which validates our manually defined splits. As evidenced by the results, in the RGB-hard scenario, our method demonstrates superior exploitation of auxiliary modality information, yielding a measurable performance gain (e.g., 62.59 vs. 61.79 mIoU of our method vs RoadFormer+). In the RGB-easy scenario, our approach likewise surpasses competing methods, though with a narrower margin (e.g., 68.45 vs. 68.05 mIoU of our method vs. GeminiFusion).

Table 6 presents the result on the MUSES dataset [2] for the RGB-LiDAR (RGB-L) and RGB-Event (RGB-E) setups. We highlight that the results of each model were submitted to the MUSES online benchmark to construct this table. Notably, we achieve state-of-the-art performance overall and in most daytime and weather conditions. In particular, we highlight how, in the foggy scenario, one of the most challenging for the RGB modality alone, our method largely surpasses the best competitor performance in the RGB-L setup (i.e., 74.12 vs 69.64 mIoU of our method vs RoadFormer+) and provides by far the most effective solution to tackle this difficult setting. In the nighttime scenario—also a challenging condition—our method consistently outperforms prior approaches, achieving higher mIoU scores (e.g., 72.48 vs. 69.02 mIoU in the RGB-E setting and 74.97 vs. 74.51 mIoU in the RGB-L setting, of our method vs. RoadFormer+). Remarkably, our method demonstrates strong performance under daytime conditions, a scenario where RGB images are particularly informative (e.g., 83.39 vs. 80.62 mIoU in the RGB-E Day setting, and 83.34 vs. 82.56 mIoU in the RGB-L Day setting, of our method vs RoadFormer+).

Qualitative results of our method and the competing models are shown in Figure 4 , in Figure 5. A closer inspection of the qualitative results in Figure 4 reveals that our method demonstrates a superior ability to capture fine-grained scene details, including road features (e.g., zebra crossings and lane markings), sidewalk elements (e.g., streetlights), and subtle distinctions between contiguous surfaces (e.g., terrain and concrete). Notably, our method produces more accurate qualitative predictions than competing approaches in both the RGB-hard and RGB-easy settings. In Figure 5, which illustrates examples from both FMB and MUSES under the RGBhard condition, our model exhibits a stronger understanding of the overall environment and background elements. In particular, the building partially occluded by trees in the FMB RGB-hard scenario is segmented with noticeably higher precision compared to the competing methods. These qualitative results highlight the robustness of our method in various scenarios and datasets, aligning with the findings from the quantitative results. Additional examples are provided in Section S.2.

# 4.2 All-modalities competitors

This section compares MM SAM-adapter with those methods that are capable of processing more than two modalities simultaneously.

Table 7 shows the results on the DeLiVER validation and test sets. We employ the official weights of the competitors trained on all available modalities (RGB, LiDAR, Depth, and Event). We note that, even though trained on only two modalities, our MM SAMadapter achieves state-of-the-art performance on the De-LiVER [48] test set regardless of the adopted auxiliary modality, and the best results also on the validation set when processing RGB and Depth. Table 8 reports results on the MUSES dataset. Again, MM SAM-adapter achieves state-of-the-art performance independently of the adopted auxiliary modality, i.e. with both RGB + Event as well as RGB + LiDAR, highlighting the potential of our approach for broad applicability to real contexts. These results are publicly available in the MUSES online benchmark 1. Therefore, the experiments in this section confirm that our approach—based on adapting the SAM backbone with multimodal fused features—is capable of effectively leveraging auxiliary knowledge while preserving the strong priors derived from the RGB-pretrained encoder.

![](images/39532ced218f136c3038bb1f1e68173a29cdb8be582cab6e4d74194b4b2023f2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Inputs"] --> B["RGB"]
    A --> C["RGB-EADY"]
    A --> D["RGB-HARD"]
    
    B --> E["Depth"]
    B --> F["LIDAR"]
    B --> G["Event"]
    
    C --> H["Depth"]
    C --> I["LIDAR"]
    C --> J["Event"]
    
    E --> K["CMNeXt"]
    E --> L["GeminiFusion"]
    E --> M["RoadFormer+"]
    E --> N["MM SAM-adapter"]
    
    F --> O["CMNeXt"]
    F --> P["GeminiFusion"]
    F --> Q["RoadFormer+"]
    F --> R["MM SAM-adapter"]
    
    G --> S["CMNeXt"]
    G --> T["GeminiFusion"]
    G --> U["RoadFormer+"]
    G --> V["MM SAM-adapter"]
    
    H --> W["CMNeXt"]
    H --> X["GeminiFusion"]
    H --> Y["RoadFormer+"]
    H --> Z["MM SAM-adapter"]
    
    I --> AA["CMNeXt"]
    I --> AB["GeminiFusion"]
    I --> AC["RoadFormer+"]
    I --> AD["MM SAM-adapter"]
    
    J --> AE["CMNeXt"]
    J --> AF["GeminiFusion"]
    J --> AG["RoadFormer+"]
    J --> AH["MM SAM-adapter"]
    
    K --> AI["CMNeXt"]
    K --> AJ["GeminiFusion"]
    K --> AK["RoadFormer+"]
    K --> AL["MM SAM-adapter"]
    
    L --> AM["CMNeXt"]
    L --> AN["GeminiFusion"]
    L --> AO["RoadFormer+"]
    L --> AP["MM SAM-adapter"]
    
    M --> AQ["CMNeXt"]
    M --> AR["GeminiFusion"]
    M --> AS["RoadFormer+"]
    M --> AT["MM SAM-adapter"]
    
    N --> AU["CMNeXt"]
    N --> AV["GeminiFusion"]
    N --> AW["RoadFormer+"]
    N --> AX["MM SAM-adapter"]
    
    O --> AY["CMNeXt"]
    O --> AZ["GeminiFusion"]
    O --> BA["RoadFormer+"]
    O --> BB["MM SAM-adapter"]
    
    P --> BC["CMNeXt"]
    P --> BD["GeminiFusion"]
    P --> BE["RoadFormer+"]
    P --> BF["MM SAM-adapter"]
    
    Q --> BG["CMNeXt"]
    Q --> BH["GeminiFusion"]
    Q --> BI["RoadFormer+"]
    Q --> BJ["MM SAM-adapter"]
    
    R --> BK["CMNeXt"]
    R --> BL["GeminiFusion"]
    R --> BM["RoadFormer+"]
    R --> BN["MM SAM-adapter"]
    
    S --> BO["CMNeXt"]
    S --> BP["GeminiFusion"]
    S --> BQ["RoadFormer+"]
    S --> BR["MM SAM-adapter"]
    
    T --> BS["CMNeXt"]
    T --> BT["GeminiFusion"]
    T --> BU["RoadFormer+"]
    T --> BV["MM SAM-adapter"]
    
    U --> BW["CMNeXt"]
    U --> BX["GeminiFusion"]
    U --> BY["RoadFormer+"]
    U --> BZ["MM SAM-adapter"]
    
    V --> CA["CMNeXt"]
    V --> CB["GeminiFusion"]
    V --> CC["RoadFormer+"]
    V --> DA["MM SAM-adapter"]
    
    W --> DB["CMNeXt"]
    W --> DC["GeminiFusion"]
    W --> DD["RoadFormer+"]
    W --> DE["MM SAM-adapter"]
    
    X --> ED["CMNeXt"]
    X --> EF["GeminiFusion"]
    X --> GF["RoadFormer+"]
    X --> GH["MM SAM-adapter"]
    
    Y --> BI
    Z --> BJ
```
</details>

Figure 4: DeLiVER [48] test set predictions in RGB-Depth, RGB-LiDAR, RGB-Event framework. RGB-easy (top) and RGB-hard (bottom) samples. Notably, the input RGB image in RGB-hard case has been stretched with an exponential operator.

![](images/698562b23aad69e1667589d9049ae9e2d74b2d130366f6b0c94f41fa4d30b802.jpg)

<details>
<summary>text_image</summary>

Inputs
GeminiFusion
Models
RoadFormer+
MM SAM-adapter
Ground truth
Overlay
RGB
Thermal
RGB
Event
Fog & Night
LiDAR
FMB
</details>

Figure 5: FMB test set and MUSES validation set predictions of the methods in Table 5 and in Table 6. Black pixels in the last column denote missing ground truth labels.   
![](images/a3a39fba933f8801a54aa2ca6b8b40a726b5d3c1018f66408684023fe8747dcb.jpg)

<details>
<summary>text_image</summary>

Inputs
Models
SAM+SH
None
SAM+SH+adapter
RGB
SAM+SH+adapter
LiDAR
MM SAM-adapter
RGB & LiDAR
Ground truth
Overlay
RGB
RGB-EASY
LiDAR
RGB
RGB-HARD
LiDAR
</details>

Figure 6: DeLiVER test set predictions of some of the methods in Table 9 in RGB-LiDAR framework. RGBeasy (top) and RGB-hard (bottom) samples.

# 4.3 Ablation studies

# 4.3.1 Main Contributions

We conduct experiments on DeLiVER to highlight the impact of the key contributions in our proposed multimodal adapter architecture, reporting results in Table 9 and Figure 6. The first row shows the results of the SAM image encoder fine-tuned for semantic segmenta-

tion by simply appending a SegFormer [43] Segmentation Head. The second and fourth rows, instead, report the results achieved by fine-tuning the SAM encoder via adapter-based approaches that utilize two different backbones: a Spatial Prior Module (SPM), as proposed in ViT-adapter [7], and a standard ConvNext-Small (ConvNext-S). In these two experiments, the adapter branch takes only RGB images as input, akin to the original ViT-adapter. We observe that the side-tuning SAM-adapter outperforms standard fine-tuning, likely because it better preserves SAM’s prior knowledge, reducing catastrophic forgetting. The third row depicts the result of SAM image encoder frozen, adapted using a different adaptation strategy based on LoRA [21]. To ensure a fair comparison with the SAM-adapter, we employ a ConvNext-S before the first LoRA layer as explained in Appendix S.1. When comparing the second, third, and fourth rows, we observe that the side-tuning SAM-adapter surpasses the LoRA adaptation, likely due to its ability to smoothly inject new domain-specific spatial information.. However, RGB-only adapters still struggle in the RGB-hard scenario, with a performance gap of about 20% mIoU vs. RGB-easy. In the fifth row, SAM with LoRA adapter performs poorly when the LiDAR modality is introduced at the first layer and unexpectedly struggles even in the RGB-hard scenario (from 50.81 in RGB-easy to 32.59 mIoU in RGB-hard). This is likely because a single LoRA layer is insufficient to integrate a modality like LiDAR into the RGB-based SAM backbone, given the inherent differences between RGB and LiDAR data. In the sixth row, we demonstrate that replacing the RGB input with the LiDAR modality in the side-tuning adapter branch significantly improves performance in the RGB-hard scenario (e.g., 44.06 vs. 37.93 mIoU for LiDAR vs. RGB adapter with the same ConvNext-S backbone). However, this comes at the cost of a performance drop in the RGB-easy scenario (from 57.07 to 55.49 mIoU), as noisier LiDAR measurements may corrupt highly informative RGB features. The seventh row shows the results of adapting using LoRA, yet by employing multimodal data as input. Multimodal LoRA utilizes our MM Fusion Encoder, which comprises two modality-specific encoders and a Fusion module. The fused encoder features are fed into the first LoRA layer. The results align with our main idea that we need to inject multimodal information to achieve effective adaptation. Indeed, multimodal LoRA obtains better performance than its single-modality counterpart in both RGB-easy and RGB-hard scenarios. However, we highlight that the best results are achieved by our proposed MM SAM-adapter strategy, as shown in the last row. Notably, our approach achieves the best performance across all scenarios. Figure 6 shows the predictions obtained by some of the methods in Table 9. Our MM SAM adapter yields way more accurate predictions than the RGB-only counterparts in the RGB-hard scenario. In the RGB-easy scenario, our model provides very similar segmentations compared to the RGB-only adapter, while performing much better than the model that uses the LiDAR-only auxiliary branch. These qualitative results vouch for our framework’s ability to learn which modality to prioritize during inference, namely to discard or deploy LiDAR features depending on whether the RGB content is sufficiently informative or not. This capability is achieved through the combination of the Multimodal Fusion Encoder with the adapter module, which smoothly contaminates SAM foundational knowledge with multimodal one.

Table 9: Analysis of our contributions on the De-LiVER test set. SH = Segmentation Head, SPM = Spatial Prior Module. SAM\* = SAM backbone frozen 

<table><tr><td>Method</td><td>Auxiliary Encoders</td><td>adapter Modalities</td><td>All mIoU</td><td>RGB-easy mIoU</td><td>RGB-hard mIoU</td></tr><tr><td>SAM + SH</td><td>None</td><td>None</td><td>53.32</td><td>54.28</td><td>35.57</td></tr><tr><td>SAM + SH + adapter</td><td>1 SPM</td><td>RGB</td><td>55.90</td><td>56.93</td><td>37.13</td></tr><tr><td>SAM* + SH + LoRA</td><td>1 ConvNext-S</td><td>RGB</td><td>53.18</td><td>54.28</td><td>35.05</td></tr><tr><td>SAM + SH + adapter</td><td>1 ConvNext-S</td><td>RGB</td><td>55.98</td><td>57.07</td><td>37.93</td></tr><tr><td>SAM* + SH + LoRA</td><td>1 ConvNext-S</td><td>LiDAR</td><td>49.77</td><td>50.81</td><td>32.59</td></tr><tr><td>SAM + SH + adapter</td><td>1 ConvNext-S</td><td>LiDAR</td><td>54.90</td><td>55.49</td><td>44.06</td></tr><tr><td>MM SAM*-LoRA</td><td>2 ConvNext-S</td><td>RGB &amp; LiDAR</td><td>53.97</td><td>54.74</td><td>40.22</td></tr><tr><td>MM SAM-adapter</td><td>2 ConvNext-S</td><td>RGB &amp; LiDAR</td><td>57.14</td><td>57.75</td><td>45.46</td></tr></table>

Table 10: Symmetric vs asymmetric architecture. Results on DeLiVER test set in the RGB-LiDAR setup. 

<table><tr><td>Method</td><td>Architecture</td><td>All mIoU</td><td>RGB-easy mIoU</td><td>RGB-hard mIoU</td></tr><tr><td>MM SAM-adapter</td><td>Symmetric</td><td>55.27</td><td>55.93</td><td>42.22</td></tr><tr><td>MM SAM-adapter</td><td>Asymmetric</td><td>57.14</td><td>57.75</td><td>45.46</td></tr></table>

Table 11: Comparison of fusion modules. Results on the DeLiVER test in the RGB-LiDAR setup. Addition or Concatenation sums or concatenates the two modality features, respectively. Road-fusion is the RoadFormer+ [22] fusion module. 

<table><tr><td>Method</td><td>Fusion Module</td><td>All mIoU</td><td>RGB-easy mIoU</td><td>RGB-hard mIoU</td></tr><tr><td>MM SAM-adapter</td><td>Addition</td><td>56.09</td><td>56.66</td><td>45.53</td></tr><tr><td>MM SAM-adapter</td><td>Concatenation</td><td>56.87</td><td>57.51</td><td>44.78</td></tr><tr><td>MM SAM-adapter</td><td>Road-Fusion</td><td>57.14</td><td>57.75</td><td>45.46</td></tr></table>

# 4.3.2 Asymmetric vs Symmetric Architecture

To validate our asymmetric network design, we replace the two ConvNext-Small (ConvNeXt-S) encoders in the auxiliary branch with two ConvNext-Base (ConvNeXt-B) networks, so as to realize a symmetric architecture where the auxiliary branch (Multimodal fusion encoder + adapter) and the SAM encoder have a similar number of parameters. In Table 10, we present results on the DeLiVER test set for the RGB-LiDAR setup. Interestingly, the symmetric MM SAM-adapter performs worse, which underscores the importance of maintaining an asymmetric structure to prioritize SAM knowledge over the auxiliary modality one. Additionally, our asymmetric design enables to use lighter networks for the auxiliary branch, reducing computational complexity without sacrificing performance. This experiment aligns with our intuition that RGB is typically the primary source of information, while auxiliary modalities are needed mainly when RGB data is lacking. Therefore, adopting a symmetric architecture could undermine this logic by assigning excessive emphasis to the auxiliary modality.

Table 12: Single modality-agnostic vs two modalityspecific encoders. Results on the DeLiVER test set in the RGB-LiDAR setup.   
MA = Modality-Agnostic. MS = Modality-Specific. 

<table><tr><td>Method</td><td>Auxiliary Encoders</td><td>All mIoU</td><td>RGB-easy mIoU</td><td>RGB-hard mIoU</td></tr><tr><td>MM SAM-adapter</td><td>MA</td><td>56.21</td><td>56.81</td><td>44.79</td></tr><tr><td>MM SAM-adapter</td><td>MS</td><td>57.14</td><td>57.75</td><td>45.46</td></tr></table>

Table 13: Comparison of our method vs SAM-frozen version. Results on DeLiVER test set in the RGB-LiDAR setup. 

<table><tr><td>Method</td><td>SAM frozen</td><td>All mIoU</td><td>RGB-easy mIoU</td><td>RGB-hard mIoU</td></tr><tr><td>MM SAM-adapter</td><td>√</td><td>55.35</td><td>56.00</td><td>43.30</td></tr><tr><td>MM SAM-adapter</td><td>✗</td><td>57.14</td><td>57.75</td><td>45.46</td></tr></table>

# 4.3.3 Alternative Fusion Modules

Our architecture is flexible, as it can incorporate different fusion modules. Even the simpler fusion strategies enable the model to achieve remarkable performance. In Table 11, we explore alternative architectures for the Fusion Module. In particular, we replace Road-Fusion [22] with simpler fusion techniques such as addition and concatenation. First of all, we point out that our intuition of adapting SAM using fused multimodal features proves effective with all the considered fusion techniques: even simple ones, like addition and concatenation, can yield state-of-the-art results on the RGB-LiDAR setup of De-LiVER. We also notice that in the case of addition-based fusion, the results for RGB-hard samples are comparable to those of Road-Fusion. In contrast, a drop in performance is observed in the RGB-easy case. This aligns with our expectations, as addition-based fusion makes it harder for the network to determine what information to use based on the scenario due to LiDAR noise being directly injected in SAM features. Conversely, with concatenation-based fusion, the results are close to those of Road-Fusion, especially in the RGB-easy scenario. Indeed, concatenating the features from the two modalities allows the adapter to easily sift out only the required information at test time. Nevertheless, the Road-Fusion strategy generates better fused features, enabling the adapter to inject and extract knowledge effectively, thereby achieving superior performance in both RGB-easy and RGB-hard scenarios.

# 4.3.4 Modality Specific vs Modality Agnostic Encoders

To assess whether using two modality-specific encoders in the Multimodal Fusion Encoder is more effective than a shared, modality-agnostic encoder, we carry out an ablation study on the DeLiVER [48] test set in the RGB-LiDAR setup. The findings, shown in Table 12, indicate that employing two modality-specific encoders performs better than a single modality-agnostic encoder for the RGB-LiDAR modality pair. This can be attributed to the fundamental differences between RGB and LiDAR data: while RGB images provide dense visual information, LiDAR data is inherently sparse.

Table 14: Comparison of our method vs scratchtrained ViT-L backbone version. Results on DeLiVER test set in the RGB-LiDAR setup.   
ST = scratch-training of ViT-L with no pretrained weights, FT = fine-tuning of SAM from pretrained weights. 

<table><tr><td>Method</td><td>SAM training</td><td>All mIoU</td><td>RGB-easy mIoU</td><td>RGB-hard mIoU</td></tr><tr><td>MM SAM-adapter</td><td>ST</td><td>53.73</td><td>54.46</td><td>40.97</td></tr><tr><td>MM SAM-adapter</td><td>FT</td><td>57.14</td><td>57.75</td><td>45.46</td></tr></table>

# 4.3.5 Frozen vs Finetuned SAM

In Table 13, we compare our method with a variant where the SAM image backbone remains frozen during training. As noted by ViT-adapter [7], this adaptation strategy benefits significantly from fine-tuning the SAM backbone. Consequently, fine-tuning SAM achieves the best performance. Remarkably, the results obtained with the SAM-frozen backbone are nearly the same as CAFuser (i.e., 55.35 vs. 55.60 mIoU – see Table 7). This demonstrates that, even with the backbone entirely frozen, our multimodal adapter and fusion modules effectively guide the SAM backbone by integrating multimodal information in a beneficial way.

# 4.3.6 Importance of SAM pre-trained weights

In Table 14, we present a reference baseline by evaluating our proposed architecture without using the pretrained weights of the SAM image encoder. As expected, leveraging SAM’s pre-trained weights is essential for achieving state-of-the-art performance.

# 5 Conclusion

We propose MM SAM-adapter, a novel multimodal semantic segmentation framework that leverages an adapter strategy to harness SAM’s rich foundational knowledge. By enabling the adapter to utilize fused features from the RGB and one auxiliary modality, MM SAM-adapter ensures robust performance in challenging conditions while maintaining optimal accuracy in simpler scenarios, resulting in state-of-the-art results across all the considered benchmarks.

One limitation of the MM SAM-adapter is that it currently supports only two input modalities due to constraints imposed by the road-fusion module. While our approach already outperforms existing all-modalities methods, an exciting avenue for future research is to extend this framework to accommodate more complex scenarios. This will require designing an innovative and efficient fusion module capable of effectively integrating more than two modalities. Additionally, exploring the potential of this framework in other tasks, such as panoptic segmentation, presents another promising avenue for further research.

# References

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.   
[2] Tim Brodermann, David Bruggemann, Christos ¨ Sakaridis, Kevin Ta, Odysseas Liagouris, Jason Corkill, and Luc Van Gool. Muses: The multisensor semantic perception dataset for driving under uncertainty. In European Conference on Computer Vision (ECCV), 2024.   
[3] Tim Brodermann, Christos Sakaridis, Yuqian Fu, ¨ and Luc Van Gool. Cafuser: Condition-aware multimodal fusion for robust semantic perception of driving scenes. IEEE Robotics and Automation Letters, 10(4):3134–3141, 2025.   
[4] Jun Chen, Heye Zhang, Mingming Gong, and Zhifan Gao. Collaborative compensative transformer network for salient object detection. Pattern Recognition, 154:110600, 2024.   
[5] Tianrun Chen, Ankang Lu, Lanyun Zhu, Chaotao Ding, Chunan Yu, Deyi Ji, Zejian Li, Lingyun Sun, Papa Mao, and Ying Zang. Sam2-adapter: Evaluating & adapting segment anything 2 in downstream tasks: Camouflage, shadow, medical image segmentation, and more. arXiv preprint arXiv:2408.04579, 2024.   
[6] Tianrun Chen, Lanyun Zhu, Chaotao Deng, Runlong Cao, Yan Wang, Shangzhan Zhang, Zejian Li, Lingyun Sun, Ying Zang, and Papa Mao. Samadapter: Adapting segment anything in underperformed scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3367–3375, 2023.   
[7] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for dense predictions. In The Eleventh International Conference on Learning Representations, 2023.   
[8] Zhiwen Chen, Zhiyu Zhu, Yifan Zhang, Junhui Hou, Guangming Shi, and Jinjian Wu. Segment any event streams via weighted adaptation of pivotal tokens. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3890–3900, June 2024.

[9] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In CVPR, 2022.   
[10] Bowen Cheng, Alexander G. Schwing, and Alexander Kirillov. Per-pixel classification is not all you need for semantic segmentation. In NeurIPS, 2021.   
[11] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3213–3223, 2016.   
[12] Camille Couprie, Clement Farabet, Laurent Na- ´ jman, and Yann LeCun. Indoor semantic segmentation using depth information. arXiv preprint arXiv:1301.3572, 2013.   
[13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.   
[14] Shaohua Dong, Yunhe Feng, Qing Yang, Yan Huang, Dongfang Liu, and Heng Fan. Efficient multimodal semantic segmentation via dual-prompt learning. arXiv preprint arXiv:2312.00360, 2023.   
[15] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.   
[16] Qishen Ha, Kohei Watanabe, Takumi Karasawa, Yoshitaka Ushiku, and Tatsuya Harada. Mfnet: Towards real-time semantic segmentation for autonomous vehicles with multi-spectral scenes. In 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 5108– 5115, 2017.   
[17] Abdul Mueed Hafiz and Ghulam Mohiuddin Bhat. A survey on instance segmentation: state of the art. International journal of multimedia information retrieval, 9(3):171–189, 2020.   
[18] Caner Hazirbas, Lingni Ma, Csaba Domokos, and Daniel Cremers. Fusenet: Incorporating depth into semantic segmentation via fusion-based cnn architecture. In Shang-Hong Lai, Vincent Lepetit,

Ko Nishino, and Yoichi Sato, editors, Computer Vision – ACCV 2016, pages 213–228, Cham, 2017. Springer International Publishing.   
[19] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked au- ´ toencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16000–16009, June 2022.   
[20] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 2790–2799. PMLR, 09–15 Jun 2019.   
[21] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.   
[22] Jianxin Huang, Jiahang Li, Ning Jia, Yuxiang Sun, Chengju Liu, Qijun Chen, and Rui Fan. Roadformer+: Delivering rgb-x scene parsing through scale-aware information decoupling and advanced heterogeneous feature fusion. IEEE Transactions on Intelligent Vehicles, 2024. DOI:10.1109/TIV.2024.3448251.   
[23] Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. OneFormer: One Transformer to Rule Universal Image Segmentation. In CVPR, 2023.   
[24] Ding Jia, Jianyuan Guo, Kai Han, Han Wu, Chao Zhang, Chang Xu, and Xinghao Chen. Gemini-Fusion: Efficient pixel-wise multimodal fusion for vision transformer. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 21753–21767. PMLR, 21–27 Jul 2024.   
[25] Habib Khan, Muhammad Talha Usman, Imad Rida, and JaKeoung Koo. Attention enhanced machine instinctive vision with human-inspired saliency detection. Image and Vision Computing, 152:105308, 2024.   
[26] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C.

Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.   
[27] Bingyu Li, Da Zhang, Zhiyuan Zhao, Junyu Gao, and Xuelong Li. Stitchfusion: Weaving any visual modalities to enhance multimodal semantic segmentation. arXiv preprint arXiv:2408.01343, 2024.   
[28] Jiahang Li, Yikang Zhang, Peng Yun, Guangliang Zhou, Qijun Chen, and Rui Fan. Roadformer: Duplex transformer for rgb-normal semantic road scene parsing. IEEE Transactions on Intelligent Vehicles, 2024. DOI:10.1109/TIV.2024.3388726.   
[29] Jinyuan Liu, Zhu Liu, Guanyao Wu, Long Ma, Risheng Liu, Wei Zhong, Zhongxuan Luo, and Xin Fan. Multi-interactive feature learning and a fulltime multi-modality benchmark for image fusion and segmentation. In International Conference on Computer Vision, 2023.   
[30] Peng Liu, Jinhong Deng, Lixin Duan, Wen Li, and Fengmao Lv. Segmenting anything in the dark via depth perception. IEEE Transactions on Multimedia, pages 1–12, 2025.   
[31] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.   
[32] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.   
[33] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.   
[34] Shervin Minaee, Yuri Boykov, Fatih Porikli, Antonio Plaza, Nasser Kehtarnavaz, and Demetri Terzopoulos. Image segmentation using deep learning: A survey. IEEE transactions on pattern analysis and machine intelligence, 44(7):3523–3542, 2021.   
[35] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Radle, Chloe Rolland,¨ Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. In The Thirteenth

International Conference on Learning Representations, 2025.   
[36] Yiran Song, Qianyu Zhou, Xuequan Lu, Zhiwen Shao, and Lizhuang Ma. Su-sam: A simple unified framework for adapting segment anything model in underperformed scenes. arXiv preprint arXiv:2401.17803, 2024.   
[37] Yuxiang Sun, Weixun Zuo, and Ming Liu. Rtfnet: Rgb-thermal fusion network for semantic segmentation of urban scenes. IEEE Robotics and Automation Letters, 4(3):2576–2583, 2019.   
[38] Hans Thisanke, Chamli Deshan, Kavindu Chamith, Sachith Seneviratne, Rajith Vidanaarachchi, and Damayanthi Herath. Semantic segmentation using vision transformers: A survey. Engineering Applications of Artificial Intelligence, 126:106669, 2023.   
[39] Kunpeng Wang, Danying Lin, Chenglong Li, Zhengzheng Tu, and Bin Luo. Adapting segment anything model to multi-modal salient object detection with semantic feature fusion guidance. arXiv preprint arXiv:2408.15063, 2024.   
[40] Yikai Wang, Xinghao Chen, Lele Cao, Wenbing Huang, Fuchun Sun, and Yunhe Wang. Multimodal token fusion for vision transformers. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.   
[41] Junde Wu, Wei Ji, Yuanpei Liu, Huazhu Fu, Min Xu, Yanwu Xu, and Yueming Jin. Medical sam adapter: Adapting segment anything model for medical image segmentation. arXiv preprint arXiv:2304.12620, 2023.   
[42] Aoran Xiao, Weihao Xuan, Heli Qi, Yun Xing, Naoto Yokoya, and Shijian Lu. Segment anything with multiple modalities. arXiv preprint arXiv:2408.09085, 2024.   
[43] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. In Neural Information Processing Systems (NeurIPS), 2021.   
[44] Yi Xin, Siqi Luo, Haodi Zhou, Junlong Du, Xiaohong Liu, Yue Fan, Qing Li, and Yuntao Du. Parameter-efficient fine-tuning for pretrained vision models: A survey. arXiv preprint arXiv:2402.02242, 2024.   
[45] Bowen Yao, Yongjian Deng, Yuhan Liu, Hao Chen, Youfu Li, and Zhen Yang. Sam-event-adapter: Adapting segment anything model for event-rgb semantic segmentation. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 9093–9100, 2024.

[46] Matthew D. Zeiler, Dilip Krishnan, Geoffrey W. Taylor, and Rob Fergus. Deconvolutional networks. In 2010 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2528– 2535. IEEE, 2010.   
[47] Jiaming Zhang, Huayao Liu, Kailun Yang, Xinxin Hu, Ruiping Liu, and Rainer Stiefelhagen. Cmx: Cross-modal fusion for rgb-x semantic segmentation with transformers. IEEE Transactions on Intelligent Transportation Systems, 2023.   
[48] Jiaming Zhang, Ruiping Liu, Hao Shi, Kailun Yang, Simon Reiß, Kunyu Peng, Haodong Fu, Kaiwei Wang, and Rainer Stiefelhagen. Delivering arbitrary-modal semantic segmentation. In CVPR, 2023.   
[49] Yifei Zhang, Desir´ e Sidib´ e, Olivier Morel, and´ Fabrice Meriaudeau. Deep multimodal fusion for ´ semantic image segmentation: A survey. Image and Vision Computing, 105:104042, 2021.   
[50] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable {detr}: Deformable transformers for end-to-end object detection. In International Conference on Learning Representations, 2021.   
[51] Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. Advances in Neural Information Processing Systems, 36, 2024.

# Biographies

Iacopo Curti received his Master Degree in Automation Engineering in 2023 from the University of Bologna. He is a PhD student at the Department of Computer Science and Engineering, University of Bologna. His research interests include image processing and deep learning frameworks, in particular multimodal architectures for dense prediction tasks.

Pierluigi Zama Ramirez received his PhD in Computer Science and Engineering in 2021. He is currently an Assistant Professor at the University of Bologna. He co-authored more than 30 publications on computer vision research topics such as semantic segmentation, depth estimation, anomaly detection, domain adaptation, neural fields, and 3D computer vision.

Alioscia Petrelli received the Master Degree in Computer Science Engineering and the PhD Degree in Computer Science from the University of Bologna in 2005 and 2016, respectively. He spent four years as research fellow at the Computer Vision Laboratory (CVLab) of the Department of Computer Science and Engineering (DISI) in Bologna. His research interests focus on computer vision, including 3D surface matching, visual search, and machine learning.

Luigi Di Stefano received the PhD degree in electronic engineering and computer science from the University of Bologna in 1994. He is currently a full professor at the Department of Computer Science and Engineering, University of Bologna, where he founded and leads the Computer Vision Laboratory (CVLab). His research interests include image processing, computer vision, and machine/deep learning. He is the author of more than 150 papers and several patents. He has been a scientific consultant for major computer vision and machine learning companies. He is a member of the IEEE Computer Society and the IAPR-IC.

# Additional Implementation Details

# S.1 LoRA multimodal implementation

The LoRA [21] multimodal network, showed in Table 9 has been implemented to assess the choice of the adapter method, presented in [7]. LoRA layers have been incorporated into every layer of the SAM Vision Transformer. However, unlike the standard LoRA implementation, we introduce modality-specific encoders and a fusion module at the first LoRA layer, to make it compatible with multimodal input. In the case of RGB-only data, a ConvNeXt network extracts features from the RGB image, which are then projected to a dimensionality compatible with SAM tokens. The standard LoRA layers subsequently process these projected features. For LiDAR input, the ConvNeXt network directly processes the Li-DAR data, and the resulting features follow the same processing as in the RGB-only case. In the RGB-LiDAR scenario, the fusion module is identical to that used in the MM SAM-adapter, with two modality-specific encoders dedicated to processing RGB and LiDAR inputs, respectively.

# S.2 Competitor details

Each competitor has been trained and tested using the official repository provided by their authors. We used CMNeXt [48] GitHub repository, RoadFormer+ [22] GitHub repository, GeminiFusion [24] GitHub repository.

# Additional qualitative results

We present qualitative results of our proposed method and competitors in Figure A, Figure B, Figure C, Figure D.

![](images/bb9beee707e537553e5a3386702d3fa63b937c6ccbe2b36a29cddf6a45109046.jpg)

<details>
<summary>text_image</summary>

Inputs
RGB LiDAR CMNeXt GeminiFusion RoadFormer+ MM SAM-adapter Ground truth Overlay
Ground Truth
Top in the Day
Top in the Day
Summary Night
Round & Right
</details>

Figure A: MUSES [2] validation set predictions in RGB-LiDAR framework. Different conditions have been illustrated.

![](images/ce464c08e09d60096f564722c29188277e4f7330f711e2b071dafde43fbeda4d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Inputs"] --> B["RGB"]
    A --> C["Event"]
    D["Models"] --> E["CMNoX1"]
    D --> F["GeminiFusion"]
    D --> G["RoadFormer+"]
    D --> H["MM SAM-adapter"]
    I["Ground truth Overlay"] --> J["Output"]
    K["Input at Day"] --> L["Black Box"]
    M["Input at Night"] --> N["Fight at Night"]
    O["Input at Night"] --> P["Snow at Night"]
    Q["Input at Night"] --> R["Snow at Night"]
```
</details>

Figure B: MUSES [2] validation set predictions in RGB-Event framework. Different conditions have been illustrated

![](images/e9719abb0f4545b0057cd30699bdc6f4fe05247d693151138d1089e0bc09c479.jpg)

<details>
<summary>text_image</summary>

Inputs
RGB LiBAR CMNXT Models GMNIFusion RoadFormer+ MM SAM-adapter Ground truth Overlay
NOSY
GRIHTRON
</details>

Figure C: DeLiVER [48] test set predictions in RGB-LiDAR framework. RGB-easy (top) and RGB-hard (bottom) samples. Notably, the input RGB image in RGBhard case has been stretched with an exponential operator.

![](images/73ba18c45abcf0ee1c18af88c6013d1f189a5f1db14db1dc1c955d34977e8d1c.jpg)

<details>
<summary>text_image</summary>

Inputs
RGB Thermal CMNeXt GeminiFusion RoadFormer+ MM SAM-adapter Ground truth Overlay
RGB Thermal
CMNeXt
GeminiFusion
RoadFormer+
MM SAM-adapter
Ground truth Overlay
01.04.02.03
</details>

Figure D: FMB [29] test set predictions. RGB-easy (top) and RGB-hard (bottom) samples.