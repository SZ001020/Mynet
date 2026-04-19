import os

# 读取原始翻译文件的内容
filepath = "/root/Mynet/docs/MFNet对应论文/A_Unified_Framework_With_Multimodal_Fine-Tuning_for_Remote_Sensing_Semantic_Segmentation (1).pdf-1827966f-06da-4b21-94b7-2df9d670e892/part1_translated.md"
with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("# IV. 实验"):
        break
    new_lines.append(line)

append_text = r"""
# IV. 实验与讨论 (EXPERIMENTS AND DISCUSSION)

## A. 数据集 (Datasets)

1) Vaihingen 数据集：它包含 16 张高分辨率正射影像，每张平均大小为 $2500 \times 2000$ 像素。这些正射影像由三个通道组成：近红外、红色和绿色 (NIRRG)，并配有 9 厘米地面采样距离的归一化数字地表模型 (DSM)。这 16 张正射影像被划分为包含 12 个图像块的训练集以及包含 4 个图像块的测试集。为了提高大型图像块的存储和读取效率，在训练和测试阶段，采用了大小为 $256 \times 256$ 的滑动窗口，而不是直接将图像块随意裁剪成小图像，这最终产生了约 960 张训练图像和 320 张测试图像。

![](images/177ab86e6618777466cbf61b5d10704436c24dc3af145d9a84e0416d2b478fa7.jpg)  
图 7. (Fig. 7.) 在此，我们展示了 (a) 和 (b) 两个选自 Vaihingen 大小为 $2048 \times 2048$ 的样本，(c) 和 (d) 两个选自 Potsdam 大小为 $2048 \times 2048$ 的样本（后两列），以及 (e) 和 (f) 两个选自 MMHunan 大小为 $256 \times 256$ 的样本。第一行展示了 Vaihingen 的 NIRRG 通道正射影像、Potsdam 的 RGB 通道正射影像以及 MMHunan 的 RGB 通道正射影像。第二行和第三行分别展示了相应的像素级深度信息以及真实标签 (ground truth)。它们展示了来自不同来源的多模态遥感数据的独立特性和互补特性。

2) Potsdam 数据集：这是一个远大于 Vaihingen 的数据集，包含 24 张高分辨率正射影像，每张大小为 $6000 \times 6000$ 像素。它包括四个多光谱波段：红外、红色、绿色和蓝色 (IRRGB)，以及 5 厘米的归一化 DSM。在此数据集中，我们使用了后三个波段 (RGB) 来使实验更具多样性。24 张正射影像被划分为 18 张作为训练集，6 张作为测试集。利用相同的滑动窗口方法，该数据集产生了 10368 个训练样本和 3456 个测试样本。

Vaihingen 和 Potsdam 数据集主要分类五种前景类别：建筑物 (Building/Bui.)、树木 (Tree/Tre.)、低矮植被 (Low Vegetation/Low.)、汽车 (Car) 和不透水路面 (Impervious Surface/Imp.)。此外，还有一个被标记为杂物 (clutter) 的背景类，包含无法区分的碎物和水面。值得注意的是，用于收集训练样本的滑动窗口会以较小的步长移动，而在测试阶段，重叠区域会被平均以减少边界效应。

3) MMHunan 数据集：该数据集 [50] 在空间分辨率上与 ISPRS 数据集有显著不同，其分辨率为 10 米。它包含 500 个 Sentinel-2 图像块，每个大小为 $256 \times 256$ 像素，并附有相应的数字高程数据。我们选择了红、绿、蓝波段来构架可见图像。该数据集包含了对七种土地覆盖类型的标注：农田 (Cropland)、森林 (Forest)、草地 (Grassland)、湿地 (Wetland)、水体 (Water)、未利用地 (Unused Land) 和建成区 (Built-up Area)。

图 7 展现了这三个数据集中的直观视觉示例。无论是在数据特征还是在土地覆盖类别上存在的显著差异，都极大地丰富了我们实验的多样性。

## B. 实现细节 (Implementation Details)

所有实验均利用 PyTorch 在单张配备 24 GB RAM 的 NVIDIA GeForce RTX 3090 GPU 上进行。针对本次研究中的所有模型，我们采用了随机梯度下降 (SGD) 算法进行训练，学习率设为 0.01，动量为 0.9，权重衰减为 0.0005，批大小 (batch size) 为 10。在应用 ViT-H 骨干网络时，为满足显存限制，批大小减小至 4。所有模型总共训练了 50 个轮次 (epoch)，每个 epoch 包含 1000 个批次。在用 $256 \times 256$ 滑动窗口进行样本采集后，实施包含随机旋转和翻转在内的基础数据增强技术。对于 MMAdapter，其下投影缩放率设为 0.25。对于 MMLoRA，我们遵循文献 [27] 将低秩数值设定为 4。更多细节可见 https://github.com/sstary/SSRS 。

为了评估在多模态遥感数据上的语义分割性能，我们选用了总体准确率 (Overall Accuracy, OA)、平均 F1 分数 (mean F1 score, mF1) 以及平均交并比 (mean Intersection over Union, mIoU)。这些标准化指标能够确保我们提出的 MFNet 在与其它最先进的方法比较时维持公平。具体而言，OA 评估了所有的前景类以及背景类，而 mF1 和 mIoU 皆专门基于五个前景类别计算。

## C. 性能对比 (Performance Comparison)

我们将提出的 MFNet 与 15 种当前领先的最先进方法进行了基准测试对比，包括 PSPNet [54], MAResU-Net [51], vFuseNet [36], FuseNet [10], ESANet [52], SA-GATE [55], CMGFNet [53], TransUNet [56], CMFNet [15], UNetFormer [35], MFTransNet [16], FTransUNet [33], RS3Mamba [57], FTransDeepLab [58], 以及 MultiSenseSeg [59]，其中大部分方法是专为遥感任务设计的。在我们的实验中，PSPNet, MAResU-Net, UNetFormer, 和 $\mathsf{RS}^3$Mamba 仅利用了光学图像，这旨在突出 DSM 数据造成的影响并展示多模态方法相比单模态方法的优越性。其他则是基于不用网络架构（涵盖 CNN, Transformer 和 Mamba）的最先进多模态模型。考虑到 SAM 提供了三种不同的骨干网络结构，结合本文提出的两种多模态微调架构，我们为每个数据集给出了六组实验结果。对比的定量结果悉数列于表 I 和表 II 中。

1) 在 Vaihingen 数据集上的性能对比：如表 I 所示，与现有的分割方法相比，提出的 MFNet 在 OA, mF1, 以及 mIoU 这三项指标上均展现出了大幅提升。这些结果确证了我们的 MFNet 能够切实有效地利用起那些深植于 SAM 当中的广博通用知识。特别是，MFNet 在四大具体分类（即：建筑物、树木、低矮植被和不透水路面）上的表现均超过了其他最先进模型。在整体性能方面，搭载 ViT-H 的提出 MFNet(MMAdapter) 取得了 $92.97\%$的 OA，$91.71\%$的 mF1 得分，以及 $85.03\%$的 mIoU，不仅全面拔得头筹，而且比起位居第二的最佳方法 MultiSenseSeg，分别胜出了 $0.24\%$, $0.29\%$, 和 $0.50\%$。不仅如此，三种 MFNet 的骨干变体也均体现出各具千秋的优势。即便用的是当中最小的变体（ViT-B），它亦能媲美绝大部分现存做法。这也就进一步验证到了我们的多模态微调框架切实做到了高效汲取 SAM 里头的通用知识以专门辅佐多模态遥感数据开展语义分割之举。该结果彰显出本篇里提出的 MFNet 协同 MMAdapter 抑或是 MMLoRA 用于向多模态遥感任务领域内去引介基础模型（譬如 SAM），起到了极具实用意义的导航作用。而在另一边我们察觉到，建构并依靠于 MMLoRA 方案之下的 MFNet，其取得的表现则显得没那种建立在 MMAdapter 上的那么有成效，关于这点，接下来的第四节F（Section IV-F）中会有详细拆解。

表 I (TABLE I) Vaihingen 数据集上的定量结果。最佳结果以粗体显示。次佳结果以下划线标出 (%)

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="6">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Bui.</td><td>Tre.</td><td>Low.</td><td>Car</td><td>Imp.</td><td>Total</td></tr><tr><td>FuseNet [10]</td><td>VGG16</td><td>96.28</td><td>90.28</td><td>78.98</td><td>81.37</td><td>91.66</td><td>90.51</td><td>87.71</td><td>78.71</td></tr><tr><td>vFuseNet [36]</td><td>VGG16</td><td>95.92</td><td>91.36</td><td>77.64</td><td>76.06</td><td>91.85</td><td>90.49</td><td>87.89</td><td>78.92</td></tr><tr><td>MAResU-Net [51]</td><td>ResNet18</td><td>94.84</td><td>89.99</td><td>79.09</td><td>85.89</td><td>92.19</td><td>90.17</td><td>88.54</td><td>79.89</td></tr><tr><td>ESANet [52]</td><td>ResNet34</td><td>95.69</td><td>90.50</td><td>77.16</td><td>85.46</td><td>91.39</td><td>90.61</td><td>88.18</td><td>79.42</td></tr><tr><td>CMGFNet [53]</td><td>ResNet34</td><td>97.75</td><td>91.60</td><td>80.03</td><td>87.28</td><td>92.35</td><td>91.72</td><td>90.00</td><td>82.26</td></tr><tr><td>PSPNet [54]</td><td>ResNet101</td><td>94.52</td><td>90.17</td><td>78.84</td><td>79.22</td><td>92.03</td><td>89.94</td><td>86.55</td><td>76.96</td></tr><tr><td>SA-GATE [55]</td><td>ResNet101</td><td>94.84</td><td>92.56</td><td>81.29</td><td>87.79</td><td>91.69</td><td>91.10</td><td>89.81</td><td>81.27</td></tr><tr><td>CMFNet [15]</td><td>VGG16</td><td>97.17</td><td>90.82</td><td>80.37</td><td>85.47</td><td>92.36</td><td>91.40</td><td>89.48</td><td>81.44</td></tr><tr><td>UNetFormer [35]</td><td>ResNet18</td><td>96.23</td><td>91.85</td><td>79.95</td><td>86.99</td><td>91.85</td><td>91.17</td><td>89.48</td><td>81.97</td></tr><tr><td>MFTransNet [16]</td><td>ResNet34</td><td>96.41</td><td>91.48</td><td>80.09</td><td>86.52</td><td>92.11</td><td>91.22</td><td>89.62</td><td>81.61</td></tr><tr><td>TransUNet [56]</td><td>R50-ViT-B</td><td>96.48</td><td>92.77</td><td>76.14</td><td>69.56</td><td>91.66</td><td>90.96</td><td>87.34</td><td>78.26</td></tr><tr><td>FTransUNet [33]</td><td>R50-ViT-B</td><td>98.20</td><td>91.94</td><td>81.49</td><td>91.27</td><td>93.01</td><td>92.40</td><td>91.21</td><td>84.23</td></tr><tr><td>RS3Mamba [57]</td><td>R18-Mamba-T</td><td>97.40</td><td>92.14</td><td>79.56</td><td>88.15</td><td>92.19</td><td>91.64</td><td>90.34</td><td>82.78</td></tr><tr><td>FTransDeepLab [58]</td><td>ResNet101</td><td>98.11</td><td>93.45</td><td>80.35</td><td>89.98</td><td>93.23</td><td>92.61</td><td>91.00</td><td>83.87</td></tr><tr><td>MultiSenseSeg [59]</td><td>Segformer-B2</td><td>97.91</td><td>93.04</td><td>81.58</td><td>89.06</td><td>93.56</td><td>92.73</td><td>91.42</td><td>84.53</td></tr><tr><td rowspan="3">MFNet (MMLoRA)</td><td>ViT-B</td><td>97.83</td><td>94.26</td><td>77.82</td><td>85.43</td><td>91.98</td><td>91.93</td><td>89.89</td><td>82.09</td></tr><tr><td>ViT-L</td><td>96.85</td><td>92.89</td><td>81.09</td><td>89.95</td><td>93.28</td><td>92.22</td><td>91.09</td><td>83.96</td></tr><tr><td>ViT-H</td><td>97.98</td><td>92.35</td><td>82.96</td><td>90.09</td><td>93.25</td><td>92.73</td><td>91.50</td><td>84.66</td></tr><tr><td rowspan="3">MFNet (MMAdapter)</td><td>ViT-B</td><td>98.73</td><td>91.41</td><td>83.09</td><td>85.63</td><td>92.91</td><td>92.62</td><td>90.60</td><td>83.24</td></tr><tr><td>ViT-L</td><td>98.84</td><td>93.17</td><td>81.16</td><td>89.23</td><td>93.39</td><td>92.93</td><td>91.51</td><td>84.72</td></tr><tr><td>ViT-H</td><td>98.38</td><td>93.94</td><td>80.70</td><td>90.47</td><td>93.59</td><td>92.97</td><td>91.71</td><td>85.03</td></tr></table>

表 II (TABLE II) Potsdam 数据集上的定量结果。最佳结果以粗体显示。次佳结果以下划线标出 (%)

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="6">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Bui.</td><td>Tre.</td><td>Low.</td><td>Car</td><td>Imp.</td><td>Total</td></tr><tr><td>FuseNet [10]</td><td>VGG16</td><td>97.48</td><td>85.14</td><td>87.31</td><td>96.10</td><td>92.64</td><td>90.58</td><td>91.60</td><td>84.86</td></tr><tr><td>vFuseNet [36]</td><td>VGG16</td><td>97.23</td><td>84.29</td><td>89.03</td><td>95.49</td><td>91.62</td><td>90.22</td><td>91.26</td><td>84.26</td></tr><tr><td>MAResU-Net [51]</td><td>ResNet18</td><td>96.82</td><td>83.97</td><td>87.70</td><td>95.88</td><td>92.19</td><td>89.82</td><td>90.86</td><td>83.61</td></tr><tr><td>ESANet [52]</td><td>ResNet34</td><td>97.10</td><td>85.31</td><td>87.81</td><td>94.08</td><td>92.76</td><td>89.74</td><td>91.22</td><td>84.15</td></tr><tr><td>CMGFNet [53]</td><td>ResNet34</td><td>97.41</td><td>86.80</td><td>86.68</td><td>95.68</td><td>92.60</td><td>90.21</td><td>91.40</td><td>84.53</td></tr><tr><td>PSPNet [54]</td><td>ResNet101</td><td>97.03</td><td>83.13</td><td>85.67</td><td>88.81</td><td>90.91</td><td>88.67</td><td>88.92</td><td>80.36</td></tr><tr><td>SA-GATE [55]</td><td>ResNet101</td><td>96.54</td><td>81.18</td><td>85.35</td><td>96.63</td><td>90.77</td><td>87.91</td><td>90.26</td><td>82.53</td></tr><tr><td>CMFNet [15]</td><td>VGG16</td><td>97.63</td><td>87.40</td><td>88.00</td><td>95.68</td><td>92.84</td><td>91.16</td><td>92.10</td><td>85.63</td></tr><tr><td>UNetFormer [35]</td><td>ResNet18</td><td>97.69</td><td>86.47</td><td>87.93</td><td>95.91</td><td>92.27</td><td>90.65</td><td>91.71</td><td>85.05</td></tr><tr><td>MFTransNet [16]</td><td>ResNet34</td><td>97.37</td><td>85.71</td><td>86.92</td><td>96.05</td><td>92.45</td><td>89.96</td><td>91.11</td><td>84.04</td></tr><tr><td>TransUNet [56]</td><td>R50-ViT-B</td><td>96.63</td><td>82.65</td><td>89.98</td><td>93.17</td><td>91.93</td><td>90.01</td><td>90.97</td><td>83.74</td></tr><tr><td>FTransUNet [33]</td><td>R50-ViT-B</td><td>97.78</td><td>88.27</td><td>88.48</td><td>96.31</td><td>93.17</td><td>91.34</td><td>92.41</td><td>86.20</td></tr><tr><td>RS3Mamba [57]</td><td>R18-Mamba-T</td><td>97.70</td><td>86.11</td><td>89.53</td><td>96.23</td><td>91.36</td><td>90.49</td><td>91.69</td><td>85.01</td></tr><tr><td>FTransDeepLab [58]</td><td>ResNet101</td><td>97.58</td><td>85.87</td><td>90.08</td><td>96.94</td><td>92.81</td><td>90.97</td><td>92.08</td><td>85.62</td></tr><tr><td>MultiSenseSeg [59]</td><td>Segformer-B2</td><td>98.32</td><td>87.65</td><td>89.54</td><td>96.27</td><td>92.46</td><td>91.30</td><td>92.35</td><td>86.10</td></tr><tr><td rowspan="3">MFNet (MMLoRA)</td><td>ViT-B</td><td>97.60</td><td>86.45</td><td>87.87</td><td>94.39</td><td>92.44</td><td>90.57</td><td>91.48</td><td>84.61</td></tr><tr><td>ViT-L</td><td>97.59</td><td>88.57</td><td>88.34</td><td>96.35</td><td>92.68</td><td>90.99</td><td>92.13</td><td>85.71</td></tr><tr><td>ViT-H</td><td>98.19</td><td>87.30</td><td>89.89</td><td>96.27</td><td>92.80</td><td>91.43</td><td>92.49</td><td>86.34</td></tr><tr><td rowspan="3">MFNet (MMAdapter)</td><td>ViT-B</td><td>97.93</td><td>87.13</td><td>87.72</td><td>95.68</td><td>92.68</td><td>90.89</td><td>91.79</td><td>85.14</td></tr><tr><td>ViT-L</td><td>98.31</td><td>88.78</td><td>87.27</td><td>96.29</td><td>93.69</td><td>91.62</td><td>92.51</td><td>86.37</td></tr><tr><td>ViT-H</td><td>98.44</td><td>87.37</td><td>90.36</td><td>96.24</td><td>93.17</td><td>91.71</td><td>92.70</td><td>86.69</td></tr></table>

图 8 呈现了一直观的可视化结果比对，内容囊括由多方不同手段及表现为最佳配置（搭载有 ViT-H 骨干网络，使用 MMAdapter）的 MFNet 输出。我们可以看到 MFNet 在对各类地物目标（例如：树木、汽车、以及各类建筑物）的划分能力上凸显出压倒性的卓越性能，边缘划分变得更为规整兼清晰，这进一步留存住了各类地物更为连贯一体的基本身形轮廓。总之从全局来看，MFNet 所制成的各处可视化成品展现着一份愈发纯粹乃至有条不紊的外在风貌。我们将这一连串强化成果的主因归功在了 SAM 本身非同凡响的优质特征提取能力上。靠着投入进多模态微调的这一套运转体系，SAM 在囊括分割自然领域各色单元的长项被顺理成章地转嫁套用在了各方地表物体的划入范畴。

2) 在 Potsdam 数据集上的性能对比：对 Potsdam 所展开的实验产出了同样与前一 Vaihingen 一致的好成绩。恰如表 II 所详列呈现，最高规格的配置组基于 ViT-H 骨干网搭载 MMAdapter 的 MFNet 一举斩获了 $91.71\%$， $92.70\%$，乃至于 $86.69\%$ 的 OA，mF1, 同 mIoU 分值，换言之分别把针对 FTransUNet 所赢取的拉大差幅扩充上了 $0.37\%$, $0.29\%$, 和 $0.49\%$。有一点不容忽视，对于像是建筑物、树木群、低矮丛被和透水程度低的铺建带等一系列项目，拿去与各类现存先进办法横行打分对比时也都一并观测见到了不同强弱维度的长足进步成效。甚至采用相对精简小巧体型作为主导框架（backbone）时出成绩的 MFNet 也同具不可磨灭的抢眼性能表现。借去这类从容有弹性的特质大大容许让 MFNet 可以就着诸类不同的运作情境所需，良好调停硬件设备水准需求同性能期许两者中的取舍关系。

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
  
图 8. (Fig. 8.) 在 Vaihingen 测试集上大小为 $512 \times 512$ 的可视化对比。(a) IRRG 图像，(b) DSM，(c) 真实标签 (ground truth)，(d) CMFNet，(e) FTransUNet，(f) MFTransNet (注：按原文排版推断此处插图对应)，(g) CMGFNet，(h) FTransDeepLab，(i) MultiSenseSeg，以及 (j) 提出的 MFNet。为了突出差异，在所有子图中均添加了一些紫色方框。下标 $(= 1, 2)$ 代表展示样本的序号。

表 III (TABLE III) MMHunan 数据集上的定量结果。最佳结果以粗体显示。次佳结果以下划线标出 (%)

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="8">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Cropland</td><td>Forest</td><td>Grassland</td><td>Wetland</td><td>Water</td><td>Unused Land</td><td>Built-up Area</td><td>Total</td></tr><tr><td>FuseNet [10]</td><td>VGG16</td><td>76.10</td><td>89.21</td><td>37.11</td><td>8.37</td><td>70.12</td><td>72.75</td><td>16.99</td><td>76.35</td><td>54.80</td><td>42.30</td></tr><tr><td>CMFNet [15]</td><td>VGG16</td><td>83.83</td><td>82.27</td><td>43.88</td><td>39.57</td><td>70.14</td><td>81.72</td><td>23.66</td><td>77.41</td><td>59.63</td><td>46.52</td></tr><tr><td>MFTransNet [16]</td><td>ResNet34</td><td>78.99</td><td>91.30</td><td>33.80</td><td>26.80</td><td>77.38</td><td>78.29</td><td>27.17</td><td>80.07</td><td>61.45</td><td>48.25</td></tr><tr><td>FTransUNet [33]</td><td>R50-ViT-B</td><td>78.75</td><td>90.54</td><td>32.13</td><td>27.51</td><td>76.64</td><td>75.59</td><td>48.63</td><td>79.47</td><td>61.95</td><td>48.78</td></tr><tr><td>CMGFNet [53]</td><td>ResNet34</td><td>82.08</td><td>87.90</td><td>37.50</td><td>26.48</td><td>79.82</td><td>74.64</td><td>41.15</td><td>79.85</td><td>62.69</td><td>49.44</td></tr><tr><td>FTTransDeepLab [58]</td><td>ResNet101</td><td>79.39</td><td>88.89</td><td>35.71</td><td>30.88</td><td>83.95</td><td>78.14</td><td>32.60</td><td>80.62</td><td>62.51</td><td>49.66</td></tr><tr><td>MultiSenseSeg [59]</td><td>Segformer-B2</td><td>78.03</td><td>90.93</td><td>40.47</td><td>38.16</td><td>80.19</td><td>81.03</td><td>38.14</td><td>80.51</td><td>63.74</td><td>50.76</td></tr><tr><td rowspan="3">MFNet (MMLoRA)</td><td>ViT-B</td><td>76.83</td><td>90.69</td><td>24.16</td><td>22.03</td><td>80.17</td><td>78.12</td><td>29.66</td><td>79.35</td><td>58.79</td><td>46.54</td></tr><tr><td>ViT-L</td><td>79.39</td><td>88.89</td><td>35.71</td><td>30.88</td><td>83.95</td><td>78.14</td><td>32.60</td><td>80.62</td><td>62.51</td><td>49.66</td></tr><tr><td>ViT-H</td><td>76.42</td><td>90.65</td><td>38.08</td><td>20.78</td><td>74.48</td><td>77.93</td><td>38.69</td><td>78.87</td><td>60.38</td><td>47.63</td></tr><tr><td rowspan="3">MFNet (MMAdapter)</td><td>ViT-B</td><td>74.81</td><td>91.47</td><td>27.59</td><td>25.00</td><td>86.47</td><td>78.92</td><td>37.46</td><td>80.68</td><td>62.10</td><td>49.20</td></tr><tr><td>ViT-L</td><td>79.19</td><td>89.52</td><td>46.07</td><td>42.23</td><td>81.15</td><td>77.58</td><td>39.65</td><td>80.93</td><td>65.33</td><td>51.82</td></tr><tr><td>ViT-H</td><td>79.66</td><td>90.06</td><td>42.61</td><td>38.81</td><td>80.31</td><td>78.92</td><td>40.04</td><td>81.07</td><td>64.13</td><td>51.08</td></tr></table>

图 9 给出了提取自 Potsdam 中的一组样例显像，于此处我们可瞧见更加划定明确的分界线以及一应保存完好不加破坏的事物呈现。这类体现在影像展现层面上的改良成色，很大程度是跟被放进表 II 数据之中呈现的那一些优渥 mF1 和 mIoU 打分项维系成正比的。很毋庸置疑，它更加作实论证出了本文提交出的这组 MFNet 还有附带的 MMAdapter 或 MMLoRA 落到运用端里的适用优势。

然而另外还能看得见的是，要想同时在确切辨识树（Tree）与偏矮长植被（Low Vegetation）双方面都一跃达至最优异的成效仍有一定波折。造就这一受阻瓶颈产生的因由则归咎为这两者类群全都挂着长势边界呈不规律的特质。不仅如此，两边相似的地貌兼呈盘根错节又或夹杂掩盖在一块儿的布排特性，更是大大增设了区别区分两者的绊脚难度门槛。如果把 SAM 跟有设配专门去排查并对付这些带刁钻考验项作出的独家研定给绑一块使合在一处，不失为一种尤为有趣也有盼头在后续铺展开的路径。

3) 在 MMHunan 数据集上的性能对比：位于 MMHunan 这套数据集里得出的实验比对结果被刊定展现在表 III 之内。带着配备有 ViT-L 这层架构作为主干的 MFNet (MMAdapter)，在其项下产出了分别为 $80.93\%$，$65.33\%$，还有 $51.82\%$ 这一路数据的 OA，mF1, 以及 mIoU 测算成绩，也就是相比之前最佳的 MultiSenseSeg 又实现了 $0.42\%$，$1.59\%$，乃至 $1.06\%$ 的提点。

这些成绩验证了我们方法带来的一贯的总体性能提升。此外，我们观察到了一个有趣的现象：在两种微调策略下，ViT-H 在这个数据集上的表现都不如 ViT-L。这表明在小型数据集场景中，较大的主干网络可能更容易发生过拟合，凸显了在不同遥感场景下进行主干网络选择的重要性。图 10 给出了 MMHunan 的可视化示例。在大规模场景中，遥感与自然图像之间的领域差距更为显著。然而，我们提出的方法成功地弥合了这一差距，使得视觉基础模型的普遍理解能力能够被有效地迁移到遥感任务中，从而带来了始终如一的性能提高。

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
图 9. (Fig. 9.) 在 Potsdam 测试集上大小为 $1024 \times 1024$ 的可视化对比。 (a) IRRG 图像，(b) DSM，(c) 真实标签 (ground truth)，(d) CMFNet，(e) FTransUNet，(f) MFTransNet，(g) CMGFNet，(h) FTransDeepLab，(i) MultiSenseSeg，以及 (j) 提出的 MFNet。为了突出差异，在所有子图中均添加了一些紫色方框。下标 (=1, 2) 代表展示样本的序号。

## D. 模态与微调分析 (Modality and Fine-Tuning Analysis)

为了说明多模态微调框架的必要性，我们进行了模态与微调分析，结果如表 IV 所示。在第一个实验中，我们仅使用了单模态数据并且没有应用任何微调机制，而在第二个和第五个实验中，我们应用了标准 Adapter/LoRA 机制对 SAM 的图像编码器进行了微调。这些实验凸显了多模态信息和微调机制的重要性和必要性。在第三个和第六个实验中，SAM 的图像编码器保留了标准的 Adapter/LoRA 但排除了我们提出的 MMAdapter/MMLoRA。因此，这些实验仍可以独立地提离遥感多模态特征，但它们在编码阶段缺乏至关重要的信息融合。第四和第七个实验包含了多模态信息以及我们提出的 MMAdapter/MMLoRA。

表 IV (TABLE IV) 不同模态与微调机制下 Vaihingen 数据集上的定量结果 (%)

<table><tr><td rowspan="2">Modality</td><td rowspan="2">Fine-tuning</td><td colspan="6">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Bui.</td><td>Tre.</td><td>Low.</td><td>Car</td><td>Imp.</td><td>Total</td></tr><tr><td>NIIRG</td><td>Without Adapter</td><td>94.64</td><td>89.47</td><td>71.71</td><td>76.83</td><td>89.51</td><td>88.01</td><td>85.34</td><td>75.11</td></tr><tr><td>NIIRG</td><td>Standard LoRA</td><td>96.50</td><td>93.62</td><td>80.35</td><td>86.32</td><td>92.78</td><td>92.00</td><td>90.35</td><td>82.74</td></tr><tr><td>NIIRG + DSM</td><td>Standard LoRA</td><td>97.26</td><td>92.61</td><td>81.58</td><td>86.53</td><td>91.58</td><td>91.86</td><td>89.90</td><td>82.06</td></tr><tr><td>NIIRG + DSM</td><td>MFNet with MMLoRA</td><td>96.85</td><td>92.89</td><td>81.09</td><td>89.95</td><td>93.28</td><td>92.22</td><td>91.09</td><td>83.96</td></tr><tr><td>NIIRG</td><td>Standard Adapter</td><td>96.29</td><td>93.09</td><td>80.15</td><td>89.08</td><td>92.59</td><td>92.02</td><td>90.94</td><td>83.69</td></tr><tr><td>NIIRG + DSM</td><td>Standard Adapter</td><td>99.02</td><td>91.68</td><td>83.04</td><td>89.71</td><td>92.90</td><td>92.80</td><td>91.30</td><td>84.35</td></tr><tr><td>NIIRG + DSM</td><td>MFNet with MMApapter</td><td>98.84</td><td>93.17</td><td>81.16</td><td>89.23</td><td>93.39</td><td>92.93</td><td>91.51</td><td>84.72</td></tr></table>

考察表 IV 首先突出了微调机制的必要性。如果没有 Adapter 或 LoRA，SAM 很难有效地提取遥感特征，从而导致性能显著下降。此外，结果揭示了融合多模态信息的巨大有效性。这一提升在建筑物和不透水路面（这往往具有显著且稳定的地表高程信息）的类别中尤为明显。在此之后，这种增强也提高了模型区分其他类别的能力。此外，我们观察到第三个实验的性能低于第二个实验。这是因为低秩分解显著降低了特定任务信息的维度空间。这就导致模态之间的异质性使得在编码器之后去融合它们变得十分复杂。它突出了不当的微调在利用多模态信息时所造成的挑战。我们的 MMLoRA 通过一种在图像编码器中进行的循序渐进的特征融合方法有效地解决了这一挑战。

总的来说，多模态信息的引入提供了全方位的综合收益。引入 MMAdapter 和 MMLoRA 使得能够更有效地利用 DSM 信息，重重提升了模型提取和融合多模态信息的能力。由此，语义分割性能得到了进一步提高。

## E. 消融实验 (Ablation Study)

1) 组件消融 (Component Ablation)：提出的 MFNet 包含了两个核心组件：包含 MMAdapter 或是 MMLoRA 的 SAM 图像编码器，以及 DFM。为了证实它们的有效性，我们通过系统性地移除特定组件展开了消融实验。如表 V 所示，设计了两个消融实验。在第一个实验中，DFM 被从 MFNet 中移除了，这导致缺少了对高规格抽象遥感语义特征进行深度剖析和整合的环节。第二个实验则因循了表 IV 中的第三和第六个实验的配置。

在分析消融实验之前，有一点需要高度申明，从 SAM 图像编码器中去掉所有所有 Adapter 或 LoRA 会导致模型性能断崖式剧跌，图 11 以证实了这点，同时这也说明了多模态微调机制的效力。审查图 11(c) 还有 (d) 将会揭破一个事实：也就是如果没有走微调这一步，SAM 就无法自如地从那些遥感资料内挖出起重要作用的相关特征量，使得其完全无法适切于任何挂钩语义分割的任务。然而，观察图 11 (d) 与 (f)，以及 (g) 和 (h) 都揭露出，在使用 MMAdapter 或者 MMLoRA 进行微调后，热力图形势大变。再有，图 11(f) 和 (h) 更加直白地反映出 SAM 在纵然一概针对着 RGB 摄得自然采像历经了特研培训，但它落在非光学 DSM 单端数据面上头亦具有显著奇效。显而易见 DSM 能够非常带劲地给出加助信息源。进而印证了成功微调的 SAM 图像编码器在执行多模态任务内识别还有划取区分遥感目标方面的能力。

表 V (TABLE V) 所出 MFNet 的消融实验。最优结果以粗体标识出。

<table><tr><td>MMAdapter</td><td>DFM</td><td>OA(%)</td><td>mF1(%)</td><td>mIoU(%)</td><td>MMLoRA</td><td>DFM</td><td>OA(%)</td><td>mF1(%)</td><td>mIoU(%)</td></tr><tr><td>✓</td><td></td><td>92.73</td><td>91.23</td><td>84.25</td><td>✓</td><td></td><td>92.02</td><td>90.64</td><td>83.24</td></tr><tr><td></td><td>✓</td><td>92.80</td><td>91.30</td><td>84.35</td><td></td><td>✓</td><td>91.86</td><td>89.90</td><td>82.06</td></tr><tr><td>✓</td><td>✓</td><td>92.93</td><td>91.51</td><td>84.72</td><td>✓</td><td>✓</td><td>92.22</td><td>91.09</td><td>83.96</td></tr></table>

考察表 V 表明，无论是多模态微调还是 DFM 对于增强所提出的 MFNet 模型的性能来说都是必不可少的。具体来讲，MMAdapter 以及 MMLoRA 都带动着情报不间断贯穿性得交杂在一块，如此一来便使得当编码进深渐变递增之际就能对多门路信息施起并吞吸纳及合璧。DFM 则印证下了高级别提取出得表征在处理好各类遥感相关内容执行起切分开任务当中扮演多么要紧角色。在这项研究里，我们主要是引介进来一个框架怎么使着利用好 SAM 的体系，而非专门重点把高级别的多模态特质并流之法拿去当噱头推崇。要是往里面替换些更高深的并流熔接模型估摸有望赢到进一步性能跨阶攀登改善空间。

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
图 11. (Fig. 11.) 四组热力图。(a) NIIRG 图像，(b) DSM，(c) 基于 NIIRG 图像由原始 SAM 生成的热力图，(d) 基于 DSM 数据由原始 SAM 生成的热力图，(e) 基于 NIIRG 图像由提出的 MMAdapter 生成的热力图，(f) 基于 DSM 由提出的 MMAdapter 生成的热力图，(g) 基于 NIIRG 图像由提出的 MMLoRA 生成的热力图，以及 (h) 基于 DSM 数据由提出的 MMLoRA 生成的热力图。热力图中的高亮高分值区域指示了模型所识别到的建筑物结构地块。这十分清晰地供人目见体察到本文章提及到的 MMAdapter 以及 MMLoRA 的实质奏效性到底打哪儿体现。

![](images/83f5afac6beb273f0597b274f6b2d8e3b5a536fdf61e1979c1d634b05b2c6cd9.jpg)  
图 12. (Fig. 12.) 训练阶段数据用料与模型造化成果间的成带曲线关连度挂勾说明图。也是顾及当模型手里抓不到资料的情况下无法给出评测输出，故而我们在运用完100%全套集之后，分别设办铺排出了划到占据比例有：25%，50%，还有 75%  这三种梯段量级下的测验对比去论清此事。

2) 数据量消融 (Data Amount Ablation)：为了探究 SAM 在遥感任务上的微调效率，我们使用不同比例的训练数据进行了实验，以探索训练数据量与模型性能之间的关系。具体来说，我们仅使用训练集的 $25\%$、$50\%$ 和 $75\%$ 对模型进行了微调，同时在完整的测试集上进行评估。展示在图 12 中的结果揭示了一个现象：在 $25\%$ 到 $50\%$ 之间，数据量起着至关重要的作用，但在跨过 $50\%$ 的阈值后，性能增长趋于饱和。这表明 SAM 能够通过微调快速掌握特定任务的知识量，这使得训练数据的进一步增加对下游性能仅产生递减的回报。这一发现在相对应联的任务里供出了有建设性的数据配额预估判断尺标，对于如何高效精要拨出定盘试炼期所需数据的开支上贡献着明细指引。

## F. 模型规模分析 (Model Scale Analysis)

MFNet 性能的提升在很大程度上归功于视觉基础模型 SAM 提供的通用知识。然而，SAM 也是一个大型模型，同诸如现下其它多数一般作法拉去横向挂比起来，这种大规模量级的模型不管是就耗费多高运算的演算复杂耗流，还是论断出跑批速度都讨不到什么绝对便宜。因故而此，在这块，我们将大篇幅花笔力重心披露通报那份模型所需的吃得脱的可优化调度出参数数，以及拿显存吞占去权衡硬性配适度规格要求之所在。

表 VI 呈现了本研究中比较分析的所有各方在模型规格数据面上的成绩表现对齐总章。如此中一幕所述展示，本章抛出的这项多模态微调术有效达成使大型基础模范得以塞进去一台单独 GPU 里发力运转之构想实操，而同步的仍控制稳固留有合适规整量数参数集外带在预算范围显存流耗把控上做到进退有据收放自洽。针对到这方派出的 MFNet 下调出计算到参数明细面额是由分割成这几样头项来相叠求算的成分组搭而定，一乃附于 SAM 图片提选大网身上那一出用定下的调频参列 + 搭配落进 DFM 内并兼带尾端解调网络身上那批次参头值两者相和而成；由于后面这点对于各个不一样框架定做的各种 MFNet 皆系长年照章定调死规不变。

表 VI (TABLE VI) 指代依托在一张单独运算运转开着在 NVIDIA GEFORCE RTX 3090 GPU 上单出跑跑一列挂到 $256 \times 256$ 成像上出的测验去把量出规格评底线章程。（表述内容部分参指MFNET 的各级编装打理：微调用装给挂的参配列 + 入编下放到到DFM且连解调网络模块身上带着得参量）MIOU各项得分表率选打出自于VAIHINGEN数据库内所得出最佳果报均粗楷亮大加显明。

<table><tr><td>Method</td><td>Parameter (M)</td><td>Memory (MB)</td><td>MIoU (%)</td></tr><tr><td>PSPNet [54]</td><td>46.72</td><td>3124</td><td>76.96</td></tr><tr><td>MAResU-Net [51]</td><td>26.27</td><td>1908</td><td>79.89</td></tr><tr><td>UNetFormer [35]</td><td>24.20</td><td>1980</td><td>81.97</td></tr><tr><td>RS3Mamba [57]</td><td>43.32</td><td>1548</td><td>82.78</td></tr><tr><td>TransUNet [56]</td><td>93.23</td><td>3028</td><td>78.26</td></tr><tr><td>FuseNet [10]</td><td>42.08</td><td>2284</td><td>78.71</td></tr><tr><td>vFuseNet [36]</td><td>44.17</td><td>2618</td><td>78.92</td></tr><tr><td>ESANet [52]</td><td>34.03</td><td>1914</td><td>79.42</td></tr><tr><td>SA-GATE [55]</td><td>110.85</td><td>3174</td><td>81.27</td></tr><tr><td>CMFNet [15]</td><td>123.63</td><td>4058</td><td>81.44</td></tr><tr><td>MFTransUNet [16]</td><td>43.77</td><td>1549</td><td>81.61</td></tr><tr><td>CMGFNet [53]</td><td>64.20</td><td>2463</td><td>82.26</td></tr><tr><td>FTransUNet [33]</td><td>160.88</td><td>3463</td><td>84.23</td></tr><tr><td>FTransDeepLab [58]</td><td>69.86</td><td>1624</td><td>83.87</td></tr><tr><td>MultiSenseSeg [59]</td><td>60.46</td><td>2264</td><td>84.53</td></tr><tr><td>MFNet (MMLoRA) (ViT-B)</td><td>1.03+6.22</td><td>1924</td><td>82.09</td></tr><tr><td>MFNet (MMLoRA) (ViT-L)</td><td>2.75+6.22</td><td>4158</td><td>83.96</td></tr><tr><td>MFNet (MMLoRA) (ViT-H)</td><td>4.59+6.22</td><td>6520</td><td>84.66</td></tr><tr><td>MFNet (MMAdapter) (ViT-B)</td><td>14.20+6.22</td><td>1872</td><td>83.24</td></tr><tr><td>MFNet (MMAdapter) (ViT-L)</td><td>50.45+6.22</td><td>4242</td><td>84.72</td></tr><tr><td>MFNet (MMAdapter) (ViT-H)</td><td>105.06+6.22</td><td>6854</td><td>85.03</td></tr></table>

将 MMLoRA 与 MMAdapter 进行比较能够得出这一表象定论出具：MMLoRA 乃籍由打将成千计列算得维度维向硬靠低阶打拆揉拆化作四等位阶值，就借这一出子便打压抹削下来巨大占比海量数值空间之参配率列。然而就靠着这样一头进益确实讨好了收敛轻快优待却同样暗暗也造受损失出些本来至关定盘起到分水岭功流的绝密隐暗资讯信息走板，况且是每逢赶遇上了多模态里那头复杂多端偏偏生变不拘得遥感源料处理这关。出于这些林总不留面由得制动因果点连成的一道大网关致使得MMAdapter从效力结果单看跑越到了跑越前赢了MMLoRA一马。

在本次所有的比试历中内，我们在均定维同一硬件同定统一系设参基上顺利让 ViT-L 走上一阵经打磨过顺着套改得微调套路，大丰收一举拿到过尽所有陈在老手方法大拿前挂号之优单结果成绩单。为了给体面硕主 ViT-H 拉出能耐表现腾拿让路下配设，迫受压榨这小板GPU 内存，不得不把过盘流批量值大小直接连减落了四步降做个值数字设在可算范畴批算额批值（也就是从10退给到4）。缩点这波流减并未能动辄将效力走带偏落而是促得性能反其道去进一步进高走起。所有上上下下这打底明细统率一举都尽述确凿了一头定事主调那这挂大容量量级图鉴成效大统下强效吸纳采纳再合流进收并吞成效那叫一个所向无可敌。这份心血工作另外更借着眼对头着眼长空去到一应挂限绑手受捆设备端前，就如何去给在巨量版型大模下盘清整透这一挂横着跨带多样性源性素材的功流推拨点上引照上了颇有参考值明引。

## G. 讨论 (Discussion)

本文引入了一种统一的多模态微调框架，其搭载了两种以 SAM 为根基底座发散而来的微调大局机制。身为探索踏向这门门阀之中的头波尖兵，我们靠引定构设发端开发推盘引进了 Adapter 协同 LoRA 这么两对派定极显经典作派的调校法则手段后，去向将多门各界同汇并流水交集带出的这桩遥感统筹活细致拆解下定其眼界量底摸脉盘实走大局成绩。为此也带足展开做了通篇里大量分析检验查究用来审定他们高低水准功绩何在。除此之外所推出的极具爽落顺当平直坦然调调兼流并合交接的大模网（也就是 MFNet）亦是一块打下了夯实垫木就势留以指拨下面跟上后人再继续迈足探问四大朝向走线。

1) 改进微调模块 (Improving Fine-Tuning Modules)：本工作由于采用了两家相当门面打大旗极具指标的微调用作如若是 LoRA 也还有向外说起这般配对头（Adapter）作底色用已证明立言实据证实能做做样说效有用，将来跟上的发掘调研理当被点拔受意用起更优等新高演变衍生改项体（好比 [60], [61], [62], 乃还有 [63] ），一并抛投应用尽揽这多种门子多元多相多模态在案上候遥感大类事务之上。尤要点明挂着心上大模作骨子需耗吸很大肚量记存留驻地资源之缘故以求这端发力推引出来大批模骨头上更行得动用少精更捷微操之作法极为非常值大肆深掘进追探一番的。

2) 改进融合模块 (Improving Fusion Modules)：本工作使用了一种在一众编码定局过手中发扬适应带定配挂成重向打和之特征作法手法。以备将来行有余力好端打量往那些可就着专做独门偏得合口味 MMAdapter与连同在一并指点着 MMLoRA 去订的，愈发上段位又顶有效高门合并合谋方策作进一步探索的。同样的连这般交杂接轨在深度层次上头的交并，自然更受借可期靠换着来点合上带跨合汇眼色心（如利用 cross-attention 等做做搭门等交接术）能耐拔高之手段助推去行进得高出一头改善作为一番。

3) 应对具有挑战性的类别 (Addressing Challenging Categories)：SAM 针对上向譬如想叫人去拨分开异常像连带树类群混在一旁矮本草那般高度让人拿捏不准认不太明这类硬难啃不肯动块，要不就算得上认拿定精准盯在像叫什么车轮四驱点这类巴小件套去盘抓这头就尽指说得在眼下多跟进大张旗开出长效力研查才算得了门。欲往深究想谋得抬分这门判出看成的高定定点准确性可能得势必落落到定做专项挂针对眼下偏专攻破单这几手活定做定制专门款辨认识出点大发力的特殊专攻部项模组上来办结了不可。

4) 探索其他遥感模态 (Exploration of Other Remote Sensing Modalities)：就该案本身而谈明这全用到了搭合光学面相拉线合并那张高压点模型作盘明用上了光面图片配以着上头 DSM 各色资料合用作例示范大把表明立出了一等强过高出其它高调位多模挂靠在一起并推做微调整这这番卓越面且更加供给力拨出不可忽略重要预见带人眼观出了二家交加有合有多大力盘有待挖门。这倒不是没好盼念而偏巧叫大出光彩一头是在放望让上这一大宗能 SAM 若是用放到如其它个种门像那红外光，极激光乃或就是那个波成作像 SAR 等等等之类头挂门子多重外门感作手段之上该展现何般面貌才最是教人感到鼓舞能大有看期引诱走步研究地平门径线。真到走此探索往那一头外门感手法里定然大举往深进一步拓提 SAM 个中可带力并往那各式大类遥感之里广推用处散花四面八方去了。

纵上面总述合共而观之定性也罢此作这盘本底这大件也真只可认作为那根基起造搭楼个底图开路发版作大局底限所用这其中余有好门面宽大放宽出让人能长线摸底求进深入往探探个究竟之大门未封留给了日后的跟上长途用处了。我们也只指望靠这样打板底出样来能大大宽推延伸去向那多元带多杂交手路遥感事务行上边多派上发面广出些功就了罢。

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
图 10. (Fig. 10.) 在 MMHunan 测试集上大小为 $256 \times 256$ 和 $1024 \times 1024$ 的可视化对比。该部分图像由于原提取顺位所致，放在此处说明。

"""

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
    f.write(append_text.strip() + "\n")
