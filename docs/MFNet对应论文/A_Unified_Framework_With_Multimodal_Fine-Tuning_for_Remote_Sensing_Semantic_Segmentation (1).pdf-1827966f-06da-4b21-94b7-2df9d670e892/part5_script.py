import os

filepath = "/root/Mynet/docs/MFNet对应论文/A_Unified_Framework_With_Multimodal_Fine-Tuning_for_Remote_Sensing_Semantic_Segmentation (1).pdf-1827966f-06da-4b21-94b7-2df9d670e892/part1_translated.md"
ref_path = "/root/Mynet/docs/MFNet对应论文/A_Unified_Framework_With_Multimodal_Fine-Tuning_for_Remote_Sensing_Semantic_Segmentation (1).pdf-1827966f-06da-4b21-94b7-2df9d670e892/references.txt"

with open(filepath, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("# V. 结论"):
        break
    new_lines.append(line)

append_v = r"""
# V. 结论 (CONCLUSION)

在这项研究中，我们提出了一种用于遥感语义分割的统一多模态微调融合框架，利用了视觉基础模型 SAM 中蕴含的通用知识。通过使用两种具有代表性的单模态微调机制（即 Adapter 和 LoRA），我们展示了现有机制能够无缝整合进所提出的统一框架内，以提取并融合来自遥感数据的多模态特征。融合后的深层特征被基于金字塔的 DFM 进一步优化，最终重建为分割图像。在三个基准多模态数据集（ISPRS Vaihingen、ISPRS Potsdam 和 MMHunan）上展开的全面实验证实，与当前最先进的分割方法相比，MFNet 取得了更卓越的性能。这项研究首度验证了 SAM 应对 DSM 数据的可靠性，并为主流视觉基础模型应用于遥感多模态领域铺开了一条极具前景的发展门路。不仅如此，本文给出的这套框架也具备充分的潜力，有望扩展应用至其它遥感关联任务场景，当中涵盖且不限于半监督及无监督学习任务等。

"""

with open(filepath, "w", encoding="utf-8") as f:
    f.writelines(new_lines)
    f.write(append_v.strip() + "\n\n")

if os.path.exists(ref_path):
    with open(ref_path, "r", encoding="utf-8") as reff:
        refs = reff.read()
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(refs)

