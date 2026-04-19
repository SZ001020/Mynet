import os

filepath = "/root/Mynet/docs/MFNet对应论文/A_Unified_Framework_With_Multimodal_Fine-Tuning_for_Remote_Sensing_Semantic_Segmentation (1).pdf-1827966f-06da-4b21-94b7-2df9d670e892/part1_translated.md"
with open(filepath, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("# II. 相关工作"):
        break
    new_lines.append(line)

append_text = """
# II. 相关工作 (RELATED WORKS)

## A. 多模态遥感语义分割 (Multimodal Remote Sensing Semantic Segmentation)

语义分割是遥感图像处理中的一个关键预处理步骤，利用多模态信息通常会比依赖单一模态产生更好的结果。近年来，深度学习的出现彻底改变了包括语义分割在内的整个遥感领域。基于经典的编码器-解码器框架 [8], [9]，许多基于 CNN 和 Transformer 的多模态融合方法推动了该领域的显著进步 [15], [16], [33], [36]。ResUNet-a [32] 是一种早期的基于 CNN 的架构，它仅仅将多模态数据堆叠成四个通道。此外，vFuseNet [36] 引入了一个双分支编码器来分别提取多模态特征，通过特征级别的逐元素操作实现更深层次的多模态融合。最近，Transformer [12], [13] 的引入进一步丰富了多模态网络。例如，CMFNet [15] 使用 CNN 进行特征提取，并使用 Transformer 结构跨尺度连接多模态特征，强调了尺度在多模态融合中的重要性。类似地，MFTransNet [33] 在使用 CNN 进行特征提取的同时，利用空间和通道注意力增强了自注意力模块，以实现更好的特征融合。FTransUNet [33] 提出了一种多级融合方法，以改进浅层和深层遥感语义特征的融合。尽管它们取得了出色的性能，但我们认为现有模型缺乏足够的通用知识，这对多模态融合方法的进步构成了根本限制。

## B. SAM在遥感中的应用 (SAM in Remote Sensing)

SAM [18] 作为一种通用的图像分割模型拥有其独特的地位。这个视觉基础模型在一个非常庞大的视觉语料库上进行了训练。它赋予了 SAM 泛化到未见过的目标上的非凡能力，使其非常适合在各种场景下的应用。如今，SAM 已经应用于各个领域，如自动驾驶 [38]、医学图像处理 [39] 以及遥感 [40], [41], [42], [43]。在遥感领域，SAMRS [40] 利用 SAM 整合了许多现有的遥感数据集，使用了一种名为旋转边界框的新型提示。此外，近期的一些工作已经考虑微调 SAM 以用于遥感任务，如语义分割 [29], [30], [44] 和变化检测 [45], [46]。

对于单模态任务，CWSAM [29] 通过引入特定任务的输入模块和类别级的掩码解码器将 SAM 适配于合成孔径雷达 (SAR) 图像。MeSAM [30] 将 inception 混合器整合进 SAM 的图像编码器中以保留高频特征，并为光学图像引入了多尺度连接的掩码解码器。SAM_MLoRA [31] 则并行使用多个 LoRA 模块来增强 LoRA 的分解能力。对于多模态任务，RingMo-SAM [44] 引入了专门为多模态遥感数据量身定制的提示编码器，以及类别解耦掩码解码器。可以观察到，这些方法主要侧重于改进微调机制以及设计特定任务的提示或掩码解码器。它们已经初步探索了 SAM 在遥感任务中的泛化能力。然而，正如第一节 (Section I) 所讨论的，SAM 中的通用知识主要集中在图像编码器上。虽然这些方法成功地利用了 SAM 在遥感领域的通用知识，但其复杂的架构严重阻碍了它们对现有通用语义分割网络的适配。此外，目前还没有专门为 DSM 数据设计的基于 SAM 的多模态方法。
"""

with open(filepath, "w") as f:
    f.writelines(new_lines)
    f.write(append_text.strip() + "\n")
