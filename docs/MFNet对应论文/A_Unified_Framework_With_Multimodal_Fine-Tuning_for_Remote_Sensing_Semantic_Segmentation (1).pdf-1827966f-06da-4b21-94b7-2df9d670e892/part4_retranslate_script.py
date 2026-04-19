import os

filepath = "/root/Mynet/docs/MFNet对应论文/A_Unified_Framework_With_Multimodal_Fine-Tuning_for_Remote_Sensing_Semantic_Segmentation (1).pdf-1827966f-06da-4b21-94b7-2df9d670e892/MFNet论文翻译md"

with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()

part4_start = text.find("# IV. 实验")
part5_start = text.find("# V. 结论")

if part4_start != -1 and part5_start != -1:
    before = text[:part4_start]
    after = text[part5_start:]
    
    new_part4 = r"""# IV. 实验与讨论 (EXPERIMENTS AND DISCUSSION)

## A. 数据集 (Datasets)

1) Vaihingen 数据集：该数据集包含 16 张高分辨率正射影像，每张平均大小为 $2 5 0 0 \times 2 0 0 0$ 像素。这些正射影像由三个通道组成：近红外、红色和绿色 (NIRRG)，并配有 9 厘米地面采样距离的归一化数字地表模型 (DSM)。这 16 张正射影像被划分为包含 12 个图像块的训练集以及包含 4 个图像块的测试集。为了提高大型图像块的存储和读取效率，在训练和测试阶段，采用了大小为 $2 5 6 \times 2 5 6$ 的滑动窗口进行处理，而不是直接将图像块裁剪成较小的图像，这最终产生了约 960 张训练图像和 320 张测试图像。

![](images/177ab86e6618777466cbf61b5d10704436c24dc3af145d9a84e0416d2b478fa7.jpg)  
图 7. (Fig. 7.) 在此，我们展示了 (a) 和 (b) 两个选自 Vaihingen 数据集尺寸为 $2 0 4 8 \times 2 0 4 8$ 的样本，(c) 和 (d) 两个选自 Potsdam 数据集尺寸为 $2 0 4 8 \times 2 0 4 8$ 的样本（后两列），以及 (e) 和 (f) 两个选自 MMHunan 数据集尺寸为 $2 5 6 \times 2 5 6$ 的样本。第一行分别展示了 Vaihingen 的 NIRRG 通道正射影像、Potsdam 的 RGB 通道正射影像以及 MMHunan 的 RGB 通道正射影像。第二行和第三行则展示了相应的像素级深度信息和真实标签 (ground truth)。它们展现了来自不同来源的多模态遥感数据的独立特征与互补特性。

2) Potsdam 数据集：该数据集比 Vaihingen 数据集大得多，包含 24 张高分辨率正射影像，每张尺寸为 $6 0 0 0 \times 6 0 0 0$ 像素。它包括四个多光谱波段：红外、红色、绿色和蓝色 (IRRGB)，以及 5 厘米的归一化 DSM。在此数据集中，我们使用了后三个波段 (RGB) 以增加实验的多样性。这 24 张正射影像被划分为 18 张作为训练集，6 张作为测试集。采用相同的滑动窗口方法，该数据集产生了 10368 个训练样本和 3456 个测试样本。

Vaihingen 和 Potsdam 数据集主要包含五种前景类别：建筑物 (Building/Bui.)、树木 (Tree/Tre.)、低矮植被 (Low Vegetation/Low.)、汽车 (Car) 和不透水路面 (Impervious Surface/Imp.)。此外，还有一个被标记为背景杂物 (clutter) 的类别，包含难以区分的碎片和水面。值得注意的是，用于收集训练样本的滑动窗口会以较小的步长移动，而在测试阶段，重叠区域会被平均以减少边界效应。

3) MMHunan 数据集：该数据集 [50] 在空间分辨率上与 ISPRS 数据集有显著不同，其分辨率为 10 米。它包含 500 个 Sentinel-2 图像块，每个大小为 $2 5 6 \times 2 5 6$ 像素，并附有相应的数字高程数据。我们选择了红、绿、蓝波段来构建可见光图像。该数据集包含了对七种土地覆盖类型的标注：农田 (Cropland)、森林 (Forest)、草地 (Grassland)、湿地 (Wetland)、水体 (Water)、未利用地 (Unused Land) 和建成区 (Built-up Area)。

图 7 展示了这三个数据集中的视觉示例。无论是在数据特征还是在土地覆盖类别上的显著差异，都极大地提升了我们实验的多样性。

## B. 实现细节 (Implementation Details)

所有实验均利用 PyTorch 框架在单张配备 24 GB 显存的 NVIDIA GeForce RTX 3090 GPU 上进行。我们采用随机梯度下降 (SGD) 算法对所有模型进行训练，学习率设为 0.01，动量为 0.9，权重衰减为 0.0005，批大小 (batch size) 为 10。在应用 ViT-H 骨干网络时，为满足显存限制，批大小减小至 4。所有模型总共训练了 50 个轮次 (epoch)，每个 epoch 包含 1000 个迭代批次。在使用 $2 5 6 \times 2 5 6$ 滑动窗口进行样本采集后，实施了包含随机旋转和翻转在内的基础数据增强技术。对于 MMAdapter，其下投影缩放率设为 0.25。对于 MMLoRA，我们参考 [27] 将低秩数值设定为 4。更多细节可见 https://github.com/sstary/SSRS 。

为了评估在多模态遥感数据上的语义分割性能，我们选用了总体准确率 (Overall Accuracy, OA)、平均 F1 分数 (mean F1 score, mF1) 以及平均交并比 (mean Intersection over Union, mIoU)。这些标准化指标确保了我们提出的 MFNet 与其它最先进方法比较时的公平性。具体而言，OA 评估了所有的前景类别以及背景类别，而 mF1 和 mIoU 则是专门针对五个前景类别计算的。

## C. 性能对比 (Performance Comparison)

我们将提出的 MFNet 与 15 种当前最先进的方法进行了基准测试对比，包括 PSPNet [54], MAResU-Net [51], vFuseNet [36], FuseNet [10], ESANet [52], SA-GATE [55], CMGFNet [53], TransUNet [56], CMFNet [15], UNetFormer [35], MFTransNet [16], FTransUNet [33], RS3Mamba [57], FTransDeepLab [58], 以及 MultiSenseSeg [59]，其中大部分方法是专为遥感任务设计的。在我们的实验中，PSPNet, MAResU-Net, UNetFormer, 和 $\mathsf { R } \mathsf { S } ^ { 3 }$Mamba 仅利用了光学图像，这旨在突出 DSM 数据的影响，并展示多模态方法相比单模态方法的优势。其他方法则是基于不同网络架构（涵盖 CNN, Transformer 和 Mamba）的最先进多模态模型。考虑到 SAM 提供了三种不同的骨干网络结构，结合本文提出的两种多模态微调架构，我们为每个数据集提供了六组实验结果。对比的定量结果详见表 I 和表 II。

1) 在 Vaihingen 数据集上的性能对比：如表 I 所示，与现有的分割方法相比，提出的 MFNet 在 OA, mF1, 以及 mIoU 这三项指标上均展现出了大幅提升。这些结果证实了我们的 MFNet 能够切实有效地利用 SAM 中蕴含的丰富通用知识。特别值得注意的是，MFNet 在四大具体分类（建筑物、树木、低矮植被和不透水路面）上的表现均超过了其他最先进模型。在整体性能方面，搭载 ViT-H 的 MFNet (MMAdapter) 取得了 $9 2 . 9 7 \%$ 的 OA、$9 1 . 7 1 \%$ 的 mF1 得分，以及 $8 5 . 0 3 \%$ 的 mIoU，相较于排名第二的方法 MultiSenseSeg，分别提升了 $0 . 2 4 \%$、$0 . 2 9 \%$ 和 $0 . 5 0 \%$。此外，三种 MFNet 的骨干变体也展示出独特的优势。即便使用最小的变体 (ViT-B)，也能够媲美绝大部分现有方法。这进一步验证了我们的多模态微调框架能够高效地汲取 SAM 的通用知识，以辅助多模态遥感数据的语义分割。该结果彰显了本文提出的 MFNet 结合 MMAdapter 或 MMLoRA 在向多模态遥感任务引入基础模型（如 SAM）方面的重要实用价值。另一方面我们也观察到，基于 MMLoRA 的 MFNet 表现不如基于 MMAdapter 的 MFNet，我们将在第四节F (Section IV-F) 进行详细分析。

表 I (TABLE I) Vaihingen 数据集上的定量结果。最佳结果以粗体显示。次佳结果以下划线标出 (%)

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="6">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Bui.</td><td>Tre.</td><td>Low.</td><td>Car</td><td>Imp.</td><td>Total</td></tr><tr><td>FuseNet [10]</td><td>VGG16</td><td>96.28</td><td>90.28</td><td>78.98</td><td>81.37</td><td>91.66</td><td>90.51</td><td>87.71</td><td>78.71</td></tr><tr><td>vFuseNet [36]</td><td>VGG16</td><td>95.92</td><td>91.36</td><td>77.64</td><td>76.06</td><td>91.85</td><td>90.49</td><td>87.89</td><td>78.92</td></tr><tr><td>MAResU-Net [51]</td><td>ResNet18</td><td>94.84</td><td>89.99</td><td>79.09</td><td>85.89</td><td>92.19</td><td>90.17</td><td>88.54</td><td>79.89</td></tr><tr><td>ESANet [52]</td><td>ResNet34</td><td>95.69</td><td>90.50</td><td>77.16</td><td>85.46</td><td>91.39</td><td>90.61</td><td>88.18</td><td>79.42</td></tr><tr><td>CMGFNet [53]</td><td>ResNet34</td><td>97.75</td><td>91.60</td><td>80.03</td><td>87.28</td><td>92.35</td><td>91.72</td><td>90.00</td><td>82.26</td></tr><tr><td>PSPNet [54]</td><td>ResNet101</td><td>94.52</td><td>90.17</td><td>78.84</td><td>79.22</td><td>92.03</td><td>89.94</td><td>86.55</td><td>76.96</td></tr><tr><td>SA-GATE [55]</td><td>ResNet101</td><td>94.84</td><td>92.56</td><td>81.29</td><td>87.79</td><td>91.69</td><td>91.10</td><td>89.81</td><td>81.27</td></tr><tr><td>CMFNet [15]</td><td>VGG16</td><td>97.17</td><td>90.82</td><td>80.37</td><td>85.47</td><td>92.36</td><td>91.40</td><td>89.48</td><td>81.44</td></tr><tr><td>UNetFormer [35]</td><td>ResNet18</td><td>96.23</td><td>91.85</td><td>79.95</td><td>86.99</td><td>91.85</td><td>91.17</td><td>89.48</td><td>81.97</td></tr><tr><td>MFTransNet [16]</td><td>ResNet34</td><td>96.41</td><td>91.48</td><td>80.09</td><td>86.52</td><td>92.11</td><td>91.22</td><td>89.62</td><td>81.61</td></tr><tr><td>TransUNet [56]</td><td>R50-ViT-B</td><td>96.48</td><td>92.77</td><td>76.14</td><td>69.56</td><td>91.66</td><td>90.96</td><td>87.34</td><td>78.26</td></tr><tr><td>FTransUNet [33]</td><td>R50-ViT-B</td><td>98.20</td><td>91.94</td><td>81.49</td><td>91.27</td><td>93.01</td><td>92.40</td><td>91.21</td><td>84.23</td></tr><tr><td>RS3Mamba [57]</td><td>R18-Mamba-T</td><td>97.40</td><td>92.14</td><td>79.56</td><td>88.15</td><td>92.19</td><td>91.64</td><td>90.34</td><td>82.78</td></tr><tr><td>FTransDeepLab [58]</td><td>ResNet101</td><td>98.11</td><td>93.45</td><td>80.35</td><td>89.98</td><td>93.23</td><td>92.61</td><td>91.00</td><td>83.87</td></tr><tr><td>MultiSenseSeg [59]</td><td>Segformer-B2</td><td>97.91</td><td>93.04</td><td>81.58</td><td>89.06</td><td>93.56</td><td>92.73</td><td>91.42</td><td>84.53</td></tr><tr><td rowspan="3">MFNet (MMLoRA)</td><td>ViT-B</td><td>97.83</td><td>94.26</td><td>77.82</td><td>85.43</td><td>91.98</td><td>91.93</td><td>89.89</td><td>82.09</td></tr><tr><td>ViT-L</td><td>96.85</td><td>92.89</td><td>81.09</td><td>89.95</td><td>93.28</td><td>92.22</td><td>91.09</td><td>83.96</td></tr><tr><td>ViT-H</td><td>97.98</td><td>92.35</td><td>82.96</td><td>90.09</td><td>93.25</td><td>92.73</td><td>91.50</td><td>84.66</td></tr><tr><td rowspan="3">MFNet (MMAdapter)</td><td>ViT-B</td><td>98.73</td><td>91.41</td><td>83.09</td><td>85.63</td><td>92.91</td><td>92.62</td><td>90.60</td><td>83.24</td></tr><tr><td>ViT-L</td><td>98.84</td><td>93.17</td><td>81.16</td><td>89.23</td><td>93.39</td><td>92.93</td><td>91.51</td><td>84.72</td></tr><tr><td>ViT-H</td><td>98.38</td><td>93.94</td><td>80.70</td><td>90.47</td><td>93.59</td><td>92.97</td><td>91.71</td><td>85.03</td></tr></table>

表 II (TABLE II) Potsdam 数据集上的定量结果。最佳结果以粗体显示。次佳结果以下划线标出 (%)

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="6">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Bui.</td><td>Tre.</td><td>Low.</td><td>Car</td><td>Imp.</td><td>Total</td></tr><tr><td>FuseNet [10]</td><td>VGG16</td><td>97.48</td><td>85.14</td><td>87.31</td><td>96.10</td><td>92.64</td><td>90.58</td><td>91.60</td><td>84.86</td></tr><tr><td>vFuseNet [36]</td><td>VGG16</td><td>97.23</td><td>84.29</td><td>89.03</td><td>95.49</td><td>91.62</td><td>90.22</td><td>91.26</td><td>84.26</td></tr><tr><td>MAResU-Net [51]</td><td>ResNet18</td><td>96.82</td><td>83.97</td><td>87.70</td><td>95.88</td><td>92.19</td><td>89.82</td><td>90.86</td><td>83.61</td></tr><tr><td>ESANet [52]</td><td>ResNet34</td><td>97.10</td><td>85.31</td><td>87.81</td><td>94.08</td><td>92.76</td><td>89.74</td><td>91.22</td><td>84.15</td></tr><tr><td>CMGFNet [53]</td><td>ResNet34</td><td>97.41</td><td>86.80</td><td>86.68</td><td>95.68</td><td>92.60</td><td>90.21</td><td>91.40</td><td>84.53</td></tr><tr><td>PSPNet [54]</td><td>ResNet101</td><td>97.03</td><td>83.13</td><td>85.67</td><td>88.81</td><td>90.91</td><td>88.67</td><td>88.92</td><td>80.36</td></tr><tr><td>SA-GATE [55]</td><td>ResNet101</td><td>96.54</td><td>81.18</td><td>85.35</td><td>96.63</td><td>90.77</td><td>87.91</td><td>90.26</td><td>82.53</td></tr><tr><td>CMFNet [15]</td><td>VGG16</td><td>97.63</td><td>87.40</td><td>88.00</td><td>95.68</td><td>92.84</td><td>91.16</td><td>92.10</td><td>85.63</td></tr><tr><td>UNetFormer [35]</td><td>ResNet18</td><td>97.69</td><td>86.47</td><td>87.93</td><td>95.91</td><td>92.27</td><td>90.65</td><td>91.71</td><td>85.05</td></tr><tr><td>MFTransNet [16]</td><td>ResNet34</td><td>97.37</td><td>85.71</td><td>86.92</td><td>96.05</td><td>92.45</td><td>89.96</td><td>91.11</td><td>84.04</td></tr><tr><td>TransUNet [56]</td><td>R50-ViT-B</td><td>96.63</td><td>82.65</td><td>89.98</td><td>93.17</td><td>91.93</td><td>90.01</td><td>90.97</td><td>83.74</td></tr><tr><td>FTransUNet [33]</td><td>R50-ViT-B</td><td>97.78</td><td>88.27</td><td>88.48</td><td>96.31</td><td>93.17</td><td>91.34</td><td>92.41</td><td>86.20</td></tr><tr><td>RS3Mamba [57]</td><td>R18-Mamba-T</td><td>97.70</td><td>86.11</td><td>89.53</td><td>96.23</td><td>91.36</td><td>90.49</td><td>91.69</td><td>85.01</td></tr><tr><td>FTransDeepLab [58]</td><td>ResNet101</td><td>97.58</td><td>85.87</td><td>90.08</td><td>96.94</td><td>92.81</td><td>90.97</td><td>92.08</td><td>85.62</td></tr><tr><td>MultiSenseSeg [59]</td><td>Segformer-B2</td><td>98.32</td><td>87.65</td><td>89.54</td><td>96.27</td><td>92.46</td><td>91.30</td><td>92.35</td><td>86.10</td></tr><td rowspan="3">MFNet (MMLoRA)</td><td>ViT-B</td><td>97.60</td><td>86.45</td><td>87.87</td><td>94.39</td><td>92.44</td><td>90.57</td><td>91.48</td><td>84.61</td></tr><tr><td>ViT-L</td><td>97.59</td><td>88.57</td><td>88.34</td><td>96.35</td><td>92.68</td><td>90.99</td><td>92.13</td><td>85.71</td></tr><tr><td>ViT-H</td><td>98.19</td><td>87.30</td><td>89.89</td><td>96.27</td><td>92.80</td><td>91.43</td><td>92.49</td><td>86.34</td></tr><td rowspan="3">MFNet (MMAdapter)</td><td>ViT-B</td><td>97.93</td><td>87.13</td><td>87.72</td><td>95.68</td><td>92.68</td><td>90.89</td><td>91.79</td><td>85.14</td></tr><tr><td>ViT-L</td><td>98.31</td><td>88.78</td><td>87.27</td><td>96.29</td><td>93.69</td><td>91.62</td><td>92.51</td><td>86.37</td></tr><tr><td>ViT-H</td><td>98.44</td><td>87.37</td><td>90.36</td><td>96.24</td><td>93.17</td><td>91.71</td><td>92.70</td><td>86.69</td></tr></table>

图 8 呈现了由不同方法及最佳配置的 MFNet (采用 ViT-H 和 MMAdapter) 产生结果的直观可视化对比。我们可以看到 MFNet 在生成各类地物目标（例如树木、汽车、建筑物）的清晰、精确边界方面表现出卓越性能，从而实现了更好的分割隔离。这有助于保留地物的完整性。整体来看，MFNet 生成的可视化结果展现出更干净和更加有条理的外观。我们将这些改进主要归功于 SAM 强大的特征提取能力。通过引入多模态微调机制，SAM 识别一切自然元素的能力被有效地拓展到了识别各种地表目标物上。

2) 在 Potsdam 数据集上的性能对比：在 Potsdam 数据集上的实验得出了与 Vaihingen 数据集一致的结果。如表 II 所示，搭载 ViT-H 骨干网络并结合 MMAdapter 的 MFNet 分别获得了 $9 1 . 7 1 \%$、$9 2 . 7 0 \%$，以及 $8 6 . 6 9 \%$ 的 OA、mF1 和 mIoU 分数，比 FTransUNet 分别提升了 $0 . 3 7 \%$、$0 . 2 9 \%$ 和 $0 . 4 9 \%$。值得注意的是，与其他最先进方法相比，其在建筑物、树木、低矮植被和不透水路面类别均观察到了显著的提升。采用较小骨干网络的 MFNet 同样展现出优异的性能。这种灵活性允许 MFNet 在各种应用场景中平衡硬件要求和性能需求。

图 9 给出了 Potsdam 数据集的可视化示例。我们观察到了更明确的边界和保留完好的地物表征。这些视觉上的改善与表 II 中展示的 mF1 和 mIoU 指标相吻合。毫无疑问，这进一步证实了本文提出的 MFNet 及附带的 MMAdapter/MMLoRA 的实际适用性。

此外还可以观察到，MFNet 在同时完美识别树木 (Tree) 与低矮植被 (Low Vegetation) 时仍面临挑战。出现这一情况的原因在于这两个类别都有着不规则的边界特征。同时，由于它们特征相似，且分布上交错或重叠，使得区分它们非常困难。讲 SAM 与应对此类挑战性类别的专门设计结合起来，将是一个引人关注的未来发展方向。

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
  
图 8. (Fig. 8.) 在 Vaihingen 测试集中大小为 $5 1 2 \times 5 1 2$ 的可视化效果对比。 (a) IRRG 图像，(b) DSM，(c) 真实标签 (ground truth)，(d) CMFNet，(e) FTransUNet，(f) MFTransNet，(g) CMGFNet，(h) FTransDeepLab，(i) MultiSenseSeg，以及 (j) 提出的 MFNet。为了突出差异，在所有的子图中均添加了一些紫色方框。下标 $(= 1 , 2)$ 代表展示样本的序列编号。

表 III (TABLE III) MMHunan 数据集上的定量结果。最佳结果以粗体显示。次佳结果以下划线标出 (%)

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="8">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Cropland</td><td>Forest</td><td>Grassland</td><td>Wetland</td><td>Water</td><td>Unused Land</td><td>Built-up Area</td><td>Total</td></tr><tr><td>FuseNet [10]</td><td>VGG16</td><td>76.10</td><td>89.21</td><td>37.11</td><td>8.37</td><td>70.12</td><td>72.75</td><td>16.99</td><td>76.35</td><td>54.80</td><td>42.30</td></tr><tr><td>CMFNet [15]</td><td>VGG16</td><td>83.83</td><td>82.27</td><td>43.88</td><td>39.57</td><td>70.14</td><td>81.72</td><td>23.66</td><td>77.41</td><td>59.63</td><td>46.52</td></tr><tr><td>MFTransNet [16]</td><td>ResNet34</td><td>78.99</td><td>91.30</td><td>33.80</td><td>26.80</td><td>77.38</td><td>78.29</td><td>27.17</td><td>80.07</td><td>61.45</td><td>48.25</td></tr><tr><td>FTransUNet [33]</td><td>R50-ViT-B</td><td>78.75</td><td>90.54</td><td>32.13</td><td>27.51</td><td>76.64</td><td>75.59</td><td>48.63</td><td>79.47</td><td>61.95</td><td>48.78</td></tr><tr><td>CMGFNet [53]</td><td>ResNet34</td><td>82.08</td><td>87.90</td><td>37.50</td><td>26.48</td><td>79.82</td><td>74.64</td><td>41.15</td><td>79.85</td><td>62.69</td><td>49.44</td></tr><tr><td>FTTransDeepLab [58]</td><td>ResNet101</td><td>79.39</td><td>88.89</td><td>35.71</td><td>30.88</td><td>83.95</td><td>78.14</td><td>32.60</td><td>80.62</td><td>62.51</td><td>49.66</td></tr><tr><td>MultiSenseSeg [59]</td><td>Segformer-B2</td><td>78.03</td><td>90.93</td><td>40.47</td><td>38.16</td><td>80.19</td><td>81.03</td><td>38.14</td><td>80.51</td><td>63.74</td><td>50.76</td></tr><tr><td rowspan="3">MFNet (MMLoRA)</td><td>ViT-B</td><td>76.83</td><td>90.69</td><td>24.16</td><td>22.03</td><td>80.17</td><td>78.12</td><td>29.66</td><td>79.35</td><td>58.79</td><td>46.54</td></tr><tr><td>ViT-L</td><td>79.39</td><td>88.89</td><td>35.71</td><td>30.88</td><td>83.95</td><td>78.14</td><td>32.60</td><td>80.62</td><td>62.51</td><td>49.66</td></tr><tr><td>ViT-H</td><td>76.42</td><td>90.65</td><td>38.08</td><td>20.78</td><td>74.48</td><td>77.93</td><td>38.69</td><td>78.87</td><td>60.38</td><td>47.63</td></tr><tr><td rowspan="3">MFNet (MMAdapter)</td><td>ViT-B</td><td>74.81</td><td>91.47</td><td>27.59</td><td>25.00</td><td>86.47</td><td>78.92</td><td>37.46</td><td>80.68</td><td>62.10</td><td>49.20</td></tr><tr><td>ViT-L</td><td>79.19</td><td>89.52</td><td>46.07</td><td>42.23</td><td>81.15</td><td>77.58</td><td>39.65</td><td>80.93</td><td>65.33</td><td>51.82</td></tr><tr><td>ViT-H</td><td>79.66</td><td>90.06</td><td>42.61</td><td>38.81</td><td>80.31</td><td>78.92</td><td>40.04</td><td>81.07</td><td>64.13</td><td>51.08</td></tr></table>

3) 在 MMHunan 数据集上的性能对比：MMHunan 数据集的实验结果在表 III 中展示。配备 ViT-L 骨干的 MFNet (MMAdapter) 取得了 $8 0 . 9 3 \%$、$6 5 . 3 3 \%$ 和 $5 1 . 8 2 \%$ 的 OA、mF1 和 mIoU 得分，相比先前最佳的 MultiSenseSeg 分别提升了 $0 . 4 2 \%$、$1 . 5 9 \%$ 和 $1 . 0 6 \%$。

这些结果验证了我们的方法带来的稳定总体性能提升。此外，我们观察到了一个有趣的现象：在两种微调策略下，ViT-H 在这个数据集上的表现都不如 ViT-L。这表明在小型数据集场景中，较大的主干网络可能更容易发生过拟合，突显了根据不同遥感场景选择合适主干网络的重要性。图 10 给出了 MMHunan 的可视化示例。在大规模场景中，遥感图像与自然图像之间的领域鸿沟更为显著。然而，我们提出的方法成功地弥合了这一鸿沟，使得视觉基础模型的通用理解能力能够被有效地迁移到遥感任务中，从而带来了持续一致的性能提升。

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
图 9. (Fig. 9.) 在 Potsdam 测试集中大小为 $1 0 2 4 \times 1 0 2 4$ 的可视化效果对比。 (a) IRRG 图像，(b) DSM，(c) 真实标签 (ground truth)，(d) CMFNet，(e) FTransUNet，(f) MFTransNet，(g) CMGFNet，(h) FTransDeepLab，(i) MultiSenseSeg，以及 (j) 提出的 MFNet。为了突出差异，在所有的子图中均添加了一些紫色方框。下标 $(=1, 2)$ 代表展示样本的序列编号。

## D. 模态与微调分析 (Modality and Fine-Tuning Analysis)

为了说明多模态微调框架的必要性，我们进行了模态与微调分析，结果如表 IV 所示。在第一个实验中，我们仅使用了单模态数据并且没有应用任何微调机制，而在第二个和第五个实验中，我们应用了标准的 Adapter/LoRA 机制对 SAM 的图像编码器进行了微调。这些实验凸显了多模态信息和微调机制的重要性和必要性。在第三个和第六个实验中，SAM 的图像编码器保留了标准的 Adapter/LoRA 但排除了我们提出的 MMAdapter/MMLoRA。因此，这些实验仍可以独立地提取出遥感多模态特征，但它们在编码阶段缺乏至关重要的信息融合过程。第四和第七个实验则包含了多模态信息以及我们提出的 MMAdapter/MMLoRA。

表 IV (TABLE IV) 采用不同模态与微调机制的 Vaihingen 数据集上的定量结果 (%)

<table><tr><td rowspan="2">Modality</td><td rowspan="2">Fine-tuning</td><td colspan="6">OA</td><td rowspan="2">mF1</td><td rowspan="2">mIoU</td></tr><tr><td>Bui.</td><td>Tre.</td><td>Low.</td><td>Car</td><td>Imp.</td><td>Total</td></tr><tr><td>NIIRG</td><td>Without Adapter</td><td>94.64</td><td>89.47</td><td>71.71</td><td>76.83</td><td>89.51</td><td>88.01</td><td>85.34</td><td>75.11</td></tr><tr><td>NIIRG</td><td>Standard LoRA</td><td>96.50</td><td>93.62</td><td>80.35</td><td>86.32</td><td>92.78</td><td>92.00</td><td>90.35</td><td>82.74</td></tr><tr><td>NIIRG + DSM</td><td>Standard LoRA</td><td>97.26</td><td>92.61</td><td>81.58</td><td>86.53</td><td>91.58</td><td>91.86</td><td>89.90</td><td>82.06</td></tr><tr><td>NIIRG + DSM</td><td>MFNet with MMLoRA</td><td>96.85</td><td>92.89</td><td>81.09</td><td>89.95</td><td>93.28</td><td>92.22</td><td>91.09</td><td>83.96</td></tr><tr><td>NIIRG</td><td>Standard Adapter</td><td>96.29</td><td>93.09</td><td>80.15</td><td>89.08</td><td>92.59</td><td>92.02</td><td>90.94</td><td>83.69</td></tr><tr><td>NIIRG + DSM</td><td>Standard Adapter</td><td>99.02</td><td>91.68</td><td>83.04</td><td>89.71</td><td>92.90</td><td>92.80</td><td>91.30</td><td>84.35</td></tr><tr><td>NIIRG + DSM</td><td>MFNet with MMApapter</td><td>98.84</td><td>93.17</td><td>81.16</td><td>89.23</td><td>93.39</td><td>92.93</td><td>91.51</td><td>84.72</td></tr></table>

表 IV 首先凸显了微调机制的必要性。如果没有 Adapter 或 LoRA，SAM 很难有效地提取遥感特征，从而导致性能显著下降。此外，结果揭示了融合多模态信息的巨大优势。这一提升在建筑物和不透水路面类别中尤为明显，因为这两个类别往往具有显著且稳定的地表高程信息。随后，这种增强效果也提高了模型区分其他类别的能力。此外，我们观察到第三个实验的性能低于第二个实验。这是因为低秩分解显著降低了特定任务信息的维度空间。因此，模态之间的异质性使得在编码器之后去融合它们变得十分复杂。这指出了不当的微调方法在利用多模态信息时造成的挑战。我们的 MMLoRA 通过在图像编码器中进行的渐进特征融合方法有效地解决了这一挑战。

总的来说，多模态信息的引入提供了全方位的综合收益。引入 MMAdapter 和 MMLoRA 使得模型能够更有效地利用 DSM 信息，大幅提升了模型提取和融合多模态特征的能力。因此，遥感语义分割的性能得到了进一步的提高。

## E. 消融实验 (Ablation Study)

1) 组件消融 (Component Ablation)：提出的 MFNet 包含了两个核心组件：配备了 MMAdapter 或 MMLoRA 的 SAM 图像编码器，以及 DFM。为了验证它们的有效性，我们通过系统性地移除特定组件展开了消融实验。如表 V 所示，我们设计了两个消融实验。在第一个实验中，我们从 MFNet 中移除了 DFM，这导致网络无法对高级抽象的遥感语义特征进行深度的剖析和整合。第二个实验则采用了表 IV 中第三和第六个实验的配置。

在分析消融实验之前，需要注意的是，从 SAM 图像编码器中去掉所有的 Adapter 或 LoRA 会导致模型性能急剧下降，图 11 证实了这一点，这也同时说明了多模态微调机制的效力。观察图 11(c) 和 (d) 可以发现，如果不进行微调，SAM 就无法从遥感数据中提取出有意义的特征，使得其无法被应用于语义分割任务。然而，观察图 11(e) 和 (f) (以及 g 和 h) 可以看出，在使用 MMAdapter 或 MMLoRA 进行微调后，热力图的变化非常显著。此外，图 11(f) 和 (h) 清晰地展示了，尽管 SAM 最初是在 RGB 光学图像上训练的，但它在非光学的 DSM 数据上同样行之有效。可以观察到，DSM 能够有效地提供补充信息。因此，微调后的 SAM 图像编码器展示出了在多模态任务中有效识别和分割遥感目标的能力。

表 V (TABLE V) 所提出的 MFNet 的消融实验。最佳结果以粗体显示

<table><tr><td>MMAdapter</td><td>DFM</td><td>OA(%)</td><td>mF1(%)</td><td>mIoU(%)</td><td>MMLoRA</td><td>DFM</td><td>OA(%)</td><td>mF1(%)</td><td>mIoU(%)</td></tr><tr><td>✓</td><td></td><td>92.73</td><td>91.23</td><td>84.25</td><td>✓</td><td></td><td>92.02</td><td>90.64</td><td>83.24</td></tr><tr><td></td><td>✓</td><td>92.80</td><td>91.30</td><td>84.35</td><td></td><td>✓</td><td>91.86</td><td>89.90</td><td>82.06</td></tr><tr><td>✓</td><td>✓</td><td>92.93</td><td>91.51</td><td>84.72</td><td>✓</td><td>✓</td><td>92.22</td><td>91.09</td><td>83.96</td></tr></table>

观察表 V 表明，无论是多模态微调还是 DFM，对于提升 MFNet 模型的性能都是必不可少的。具体来讲，MMAdapter 和 MMLoRA 促进了信息的持续融合，使得在编码深度增加时多模态信息得以被提取和融合。DFM 验证了高级特征在多模态遥感数据语义分割中的重要性。在本文中，我们主要介绍了一种利用 SAM 的新框架，而不是把重点放在高级特征的融合技术上。如果将 DFM 替换为更高级的融合模型，有望获得进一步的性能提升。

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
图 11. (Fig. 11.) 四组热力图对比。 (a) NIIRG 图像， (b) DSM， (c) 原始 SAM 模型根据 NIIRG 图像生成的热力图， (d) 原始 SAM 模型根据 DSM 生成的热力图， (e) 我们提出的 MMAdapter 根据 NIIRG 图像生成的热力图， (f) 我们提出的 MMAdapter 根据 DSM 生成的热力图， (g) 我们提出的 MMLoRA 根据 NIIRG 图像生成的热力图，以及 (h) 我们提出的 MMLoRA 根据 DSM 生成的热力图。热力图中的高响应区域代表了被模型识别为建筑物的目标。我们可以清晰地观察到我们提出的 MMAdapter 和 MMLoRA 的有效性。

![](images/83f5afac6beb273f0597b274f6b2d8e3b5a536fdf61e1979c1d634b05b2c6cd9.jpg)  
图 12. (Fig. 12.) 训练阶段中训练数据量与模型性能之间的关系。由于模型在没有训练数据时无法生成预测，除了使用完整的训练集 $( 1 0 0 \% )$ 之外，我们还设计并执行了三组具有不同训练数据可用性水平的实验：$2 5 \%$， $50 \%$，和 $7 5 \%$。

2) 数据量消融 (Data Amount Ablation)：为了探究 SAM 在遥感任务上的微调效率，我们使用不同比例的训练数据进行了实验，以探索训练数据量与模型性能之间的关系。具体来说，我们仅使用训练集的 $2 5 \%$、$50 \%$ 和 $7 5 \%$ 对模型进行了微调，并在完整的测试集上进行评估。展示在图 12 中的结果揭示了一个现象：在 $2 5 \%$ 到 $50 \%$ 之间，数据量起着至关重要的作用，但在越过 $50 \%$ 的阈值后，性能的增长趋于饱和。这表明 SAM 能够通过微调快速掌握特定任务的知识，这使得训练数据的进一步增加对下游任务的性能提升产生了边际递减效应。这一发现为了解相关任务的数据需求提供了有价值的见解，并为高效利用训练数据提供了指导。

## F. 模型规模分析 (Model Scale Analysis)

MFNet 性能的提升在很大程度上归功于视觉基础模型 SAM 提供的通用知识。然而，SAM 也是一个大模型，与现有常规方法相比，大模型在计算复杂度或推理速度上并不具备优势。因此，我们着重报告模型的可训练参数数量和内存占用，以衡量其硬件需求。

表 VI 呈现了本研究中所有对比方法的模型规模分析结果。如表 VI 所指出的，本文提出的多模态微调技术使得大型基础模型可以在单张 GPU 上运行，同时将可训练参数的数量和内存开销控制在可管理的范围内。MFNet 的参数统计被划分为两部分：SAM 图像编码器中的微调参数 $+$ DFM 和解码器中的参数。后者的参数在不同的 MFNet 配置中保持不变。

表 VI (TABLE VI) 模型规模分析。在单张 NVIDIA GEFORCE RTX 3090 GPU 上处理一张 $2 5 6 \times 2 5 6$ 图像进行测量得出。对于不同的 MFNET 配置，参数统计为：SAM 图像编码器中的微调参数 + DFM 和解码器中的参数。MIOU 值为在 VAIHINGEN 数据集上的结果。最佳结果以粗体显示

<table><tr><td>Method</td><td>Parameter (M)</td><td>Memory (MB)</td><td>MIoU (%)</td></tr><tr><td>PSPNet [54]</td><td>46.72</td><td>3124</td><td>76.96</td></tr><tr><td>MAResU-Net [51]</td><td>26.27</td><td>1908</td><td>79.89</td></tr><tr><td>UNetFormer [35]</td><td>24.20</td><td>1980</td><td>81.97</td></tr><tr><td>RS3Mamba [57]</td><td>43.32</td><td>1548</td><td>82.78</td></tr><tr><td>TransUNet [56]</td><td>93.23</td><td>3028</td><td>78.26</td></tr><tr><td>FuseNet [10]</td><td>42.08</td><td>2284</td><td>78.71</td></tr><tr><td>vFuseNet [36]</td><td>44.17</td><td>2618</td><td>78.92</td></tr><tr><td>ESANet [52]</td><td>34.03</td><td>1914</td><td>79.42</td></tr><tr><td>SA-GATE [55]</td><td>110.85</td><td>3174</td><td>81.27</td></tr><tr><td>CMFNet [15]</td><td>123.63</td><td>4058</td><td>81.44</td></tr><tr><td>MFTransUNet [16]</td><td>43.77</td><td>1549</td><td>81.61</td></tr><tr><td>CMGFNet [53]</td><td>64.20</td><td>2463</td><td>82.26</td></tr><tr><td>FTransUNet [33]</td><td>160.88</td><td>3463</td><td>84.23</td></tr><tr><td>FTransDeepLab [58]</td><td>69.86</td><td>1624</td><td>83.87</td></tr><tr><td>MultiSenseSeg [59]</td><td>60.46</td><td>2264</td><td>84.53</td></tr><tr><td>MFNet (MMLoRA) (ViT-B)</td><td>1.03+6.22</td><td>1924</td><td>82.09</td></tr><tr><td>MFNet (MMLoRA) (ViT-L)</td><td>2.75+6.22</td><td>4158</td><td>83.96</td></tr><tr><td>MFNet (MMLoRA) (ViT-H)</td><td>4.59+6.22</td><td>6520</td><td>84.66</td></tr><tr><td>MFNet (MMAdapter) (ViT-B)</td><td>14.20+6.22</td><td>1872</td><td>83.24</td></tr><tr><td>MFNet (MMAdapter) (ViT-L)</td><td>50.45+6.22</td><td>4242</td><td>84.72</td></tr><tr><td>MFNet (MMAdapter) (ViT-H)</td><td>105.06+6.22</td><td>6854</td><td>85.03</td></tr></table>

将 MMLoRA 与 MMAdapter 进行比较可以看出，MMLoRA 通过低秩分解将数千维的空间压缩为秩为 4 的矩阵，从而显著减少了参数数量。虽然这种方法通常比较高效，但它可能导致丢失一些关键信息，特别是在处理复杂的遥感数据时尤为如此。因此，MMAdapter 在性能上优于 MMLoRA。

在我们的实验中，我们在相同的硬件环境和相同的超参数设置下成功微调了 ViT-L 骨干网络，不仅获得了超越所有现有方法的结果，同时保证了高效性。对于 ViT-H 骨干网络，由于 GPU 的显存限制，我们将批大小 (batch size) 从 10 减少到了 4。批大小的这种减少不仅没有降低模型的表现，反而进一步提升了性能。这些结果证明了大型视觉基础模型强大的特征提取和融合能力。本研究也为在受限硬件条件下利用大模型探索多模态任务提供了有价值的见解和范例。

## G. 讨论 (Discussion)

本文引入了一种统一的多模态微调框架，该框架包含两种基于 SAM 的多模态微调机制。作为该领域的早期探索，我们通过开发两种经典的微调方法（Adapter 和 LoRA），彻底调查了视觉基础模型在遥感多模态任务上的表现。我们进行了全面的分析实验来评估这些方法。此外，MFNet 提供了一个直观的多模态融合网络，为未来的研究铺展了方向。

1) 改进微调模块 (Improving Fine-Tuning Modules)：本工作采用了两种代表性的微调技术，即 LoRA 和 Adapter，来展示框架的有效性。我们鼓励未来的研究将更先进的变体（例如 [60], [61], [62] 和 [63]）应用到多样化的多模态遥感任务中。特别是，针对大规模模型的更高效的微调策略值得进一步探索，因为基础模型通常需要巨大的内存资源。

2) 改进融合模块 (Improving Fusion Modules)：本工作在特征编码阶段采用了基于自适应权重的方法进行特征融合。未来的工作可以探索专为 MMAdapter 和 MMLoRA 打造的更高级、更有效的融合策略。类似地，深层次的高级特征融合还可以利用交叉注意力 (cross-attention) 等其他机制来进一步提升性能。

3) 应对具有挑战性的类别 (Addressing Challenging Categories)：SAM 在应对挑战性类别时的表现（例如区分非常相似的树木和低矮植被，或准确检测汽车等小体型目标）仍有待进一步研究。为了提高准确性，可能需要开发专注于应对这些任务的专门的物体识别模块，这可能涉及针对特定类别的特征提取技术。

4) 探索其他遥感模态 (Exploration of Other Remote Sensing Modalities)：本研究以光学图像和 DSM 数据为例，展示了多模态微调框架的优越性，并为了解结合这两种模态的潜力提供了有价值的见解。然而，SAM 在其他遥感模态（如多光谱、LiDAR 激光雷达和 SAR 合成孔径雷达）上的表现同样值得探索。探索这些额外模态可以进一步增强 SAM 的能力，并大幅拓宽其在各种遥感任务中的适用性。

总体而言，本研究建立了一个基础性质的研究框架，并且留下了几个可以进一步发掘探索的关键领域。我们希望它能够被广泛扩展适用到其他各类多模态遥感任务中。

"""
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(before + new_part4 + "\n" + after)
