import os

filepath = "/root/Mynet/docs/MFNet对应论文/A_Unified_Framework_With_Multimodal_Fine-Tuning_for_Remote_Sensing_Semantic_Segmentation (1).pdf-1827966f-06da-4b21-94b7-2df9d670e892/part1_translated.md"
with open(filepath, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("# III. 提出方法"):
        break
    new_lines.append(line)

append_text = r"""
# III. 提出方法 (PROPOSED METHOD)

我们首先通过详细阐述 MMAdapter 和 MMLoRA 来介绍统一的多模态微调框架。具体而言，我们在第三节A（Section III-A）回顾了传统的单模态微调策略 Adapter 以及提出的 MMAdapter。

![](images/5cd46fd127f38d1bb43f888ac18e39258ad8d1e7272ea9a582fd79bee3b90376.jpg)

![](images/a380406abc97cb70341b9c4d855ce0802e7a7ab7b228b350a1939f784de71978.jpg)

![](images/218acdcb6cfdd66397715fb58cfc852f959d102cfde1660ce425962e80442740.jpg)  
图 3. (Fig. 3.) (a) SAM 图像编码器中不带 Adapter 的 ViT 块，(b) 配备标准 Adapter [37] 的 ViT 块，(c) 赋予所提出的 MMAdapter 的 ViT 块，以及 (d) MMAdapter 的详细结构。Adapter 促进了在特定任务中高效利用通用知识。与标准 Adapter 相比，MMAdapter 的特点是具有共享权重的双分支用于多模态特征提取。标准 Adapter 和提出的 MMAdapter 分别用于微调和融合特征。

在此之后，我们在第三节B（Section III-B）介绍了另一种经典的单模态微调策略 LoRA 以及提出的 MMLoRA。在这套多模态微调机制的基础之上，我们在第三节C（Section III-C）详细描述了本文提出的 MFNet。值得注意的是，依据在 MMAdapter 和 MMLoRA 之间的选择，MFNet 有两种不同的架构。最后，为了做出清晰的解释，我们以两种模态为例进行说明。

## A. 标准 Adapter 与提出的 MMAdapter (Standard Adapter and the Proposed MMAdapter)

具体介绍如下。

1) 标准 Adapter (Standard Adapter)：如第一节（Section I）所述，通用知识被限制在 SAM 的图像编码器中，具体来说是 ViT 块，其结构如图 3(a) 所示。在文献 [37] 中，提出使用 Adapter 通过微调来增强医学任务中 ViT 块的能力，如图 3(b) 所示。它并非调整所有参数，而是冻结预训练的 SAM 参数，同时引入两个 Adapter 模块以学习特定任务的知识。每个 Adapter 包含一个下投影（downprojection）、一个 ReLU 激活函数和一个上投影（upprojection）。下投影使用简单的 MLP 层将输入嵌入压缩到较低维度，而上投影利用另一个 MLP 层将压缩后的嵌入恢复到原始维度。对于给定的输入特征 $\pmb { x } _ { i } \in \mathbb { R } ^ { h \times w \times c }$，其中 $h$、$w$ 和 $c$ 分别代表输入特征的高度、宽度和通道数，Adapter 生成自适应特征的过程可表达为：

$$
\boldsymbol {x} _ {a} ^ {\mathrm {S A}} = \operatorname {R e L U} \left(\operatorname {L N} \left(\boldsymbol {x} _ {i}\right) \cdot \boldsymbol {W} _ {d}\right) \cdot \boldsymbol {W} _ {u} \tag {1}
$$

其中 $\pmb { W } _ { d } \in \mathbb { R } ^ { c \times \hat { c } }$ 和 $\boldsymbol { W } _ { u } \in \mathbb { R } ^ { \hat { c } \times c }$ 分别是下投影和上投影矩阵，$\hat { c } \ll c$ 是 Adapter 压缩后的中间维度。之后，自适应特征 $x _ { a }$ 以及原始 MLP 分支的输出通过残差连接与 $\boldsymbol { x } _ { i }$ 融合生成输出特征 $x _ { o }$：

$$
\boldsymbol {x} _ {o} ^ {\mathrm {S A}} = \mathcal {F} (\boldsymbol {x} _ {i}) + s \cdot \boldsymbol {x} _ {a} ^ {\mathrm {S A}} + \boldsymbol {x} _ {i} \tag {2}
$$

其中 $\mathcal { F } ( \cdot )$ 表示 MLP 操作，$s$ 是一个缩放因子，用于对特定任务知识和任务无关知识进行加权。由于 [37] 中提出的 Adapter 是专为单模态数据设计的，在下文中将其称为标准 Adapter。

2) 提出的 MMAdapter (Proposed MMAdapter)：接下来，我们将标准 Adapter 扩展到多模态任务中。提出的 MMAdapter 是我们多模态微调框架中的一个核心组件。如图 3(c) 所示，我们采用具有共享权重的双分支来处理多模态信息。多头注意力（multihead attention）之后的 Adapter 被保留以独立提取每种模态的特征，而 MLP 阶段的 Adapter 则被替换为所提出的 MMAdapter。MMAdapter 的细节展示在图 3(d) 中。在保留 Adapter 核心结构的同时，MMAdapter 通过一个融合模块实现了模态交互。值得注意的是，这种设计可以兼容任意的特征融合策略。为了突出多模态微调框架的有效性，我们采用了基于两个权重因子 $\lambda _ { 1 }$ 和 $\lambda _ { 2 }$ 的最简单的逐元素相加方法。对于两个特定的多模态输入特征 $\pmb { x } _ { i } \in \mathbb { R } ^ { h \times w \times c }$ 和 $\mathbf { y } _ { i } \in \mathbb { R } ^ { h \times w \times c }$，使用 MMAdapter 生成自适应特征的过程可描述为：

$$
\boldsymbol {x} _ {a} ^ {\mathrm {M M A}} = \operatorname {R e L U} \left(\ln \left(\boldsymbol {x} _ {i}\right) \cdot \boldsymbol {W} _ {d x}\right) \cdot \boldsymbol {W} _ {u x} \tag {3}
$$

$$
\mathbf {y} _ {a} ^ {\mathrm {M M A}} = \operatorname {R e L U} \left(\operatorname {L N} \left(\mathbf {y} _ {i}\right) \cdot \mathbf {W} _ {d y}\right) \cdot \mathbf {W} _ {u y} \tag {4}
$$

其中 $\pmb { W } _ { d x } , \pmb { W } _ { d y } \in \mathbb { R } ^ { c \times \hat { c } }$ 和 $\boldsymbol { W } _ { u x } , \boldsymbol { W } _ { u y } \in \mathbb { R } ^ { \hat { c } \times c }$ 分别是下投影和上投影矩阵。之后，利用 $\lambda _ { 1 }$ 和 $\lambda _ { 2 }$ 生成多模态输出特征 $\pmb { x } _ { o } ^ { \mathrm { M M A } }$ 和 $\mathbf { y } _ { o } ^ { \mathrm { M M A } }$ 的公式如下：

$$
\boldsymbol {x} _ {o} ^ {\mathrm {M M A}} = \mathcal {F} (\boldsymbol {x} _ {i}) + \lambda_ {1} \cdot \boldsymbol {x} _ {a} ^ {\mathrm {M M A}} + (1 - \lambda_ {1}) \cdot \boldsymbol {y} _ {a} ^ {\mathrm {M M A}} + \boldsymbol {x} _ {i} \tag {5}
$$

$$
\mathbf {y} _ {o} ^ {\mathrm {M M A}} = \mathcal {F} \left(\mathbf {y} _ {i}\right) + \lambda_ {2} \cdot \mathbf {y} _ {a} ^ {\mathrm {M M A}} + (1 - \lambda_ {2}) \cdot \mathbf {x} _ {a} ^ {\mathrm {M M A}} + \mathbf {y} _ {i}. \tag {6}
$$

![](images/32d017882ba97fdcaf8ecafca07fd70cc890f1c0f6379c706546ec5d9dfcd552.jpg)

![](images/2ef6875aa55d03ba456669b7a4f570659ddb5808887d8475c2e9f08b5b42fa64.jpg)  
(b)   
图 4. (Fig. 4.) (a) 标准 LoRA 的详细结构 [27] 以及 (b) 提出的 MMLoRA。可以观察到 MMLoRA 采用了 MMAdapter 的设计原则，这不仅降低了模块的复杂性，同时也突显了该策略的通用性。

在微调期间，仅优化新添加的参数，而其他参数保持固定。详细的注释已在图 3(b) 中提供，其他子图为了清晰和简洁起见隐去了这些注释。

## B. 标准 LoRA 与提出的 MMLoRA (Standard LoRA and the Proposed MMLoRA)

具体介绍如下。

1) 标准 LoRA (Standard LoRA)：基础模型由许多密集的层组成，通常采用全秩矩阵乘法。为了将这些预训练模型适配于特定任务，LoRA [27] 假设在适配过程中，权重的更新具有较低的“内在秩 (intrinsic rank)” [47]。这种机制可应用于任何线性层。对于一个预训练的权重矩阵 $W _ { 0 } \in \mathbb { R } ^ { d \times d }$，其更新可以通过低秩分解来表达：

$$
\boldsymbol {W} _ {0} + \Delta \boldsymbol {W} = \boldsymbol {W} _ {0} + \boldsymbol {B} \boldsymbol {A} \tag {7}
$$

其中 $\pmb { { B } } \in \mathbb { R } ^ { d \times r }$，$\pmb { A } \in \mathbb { R } ^ { r \times d }$，且秩 $r \ll d$。

在训练期间，$W _ { 0 }$ 保持固定且不接受梯度更新，而可训练的参数包含在 $\pmb { A }$ 和 $\pmb { B }$ 中。给定输入特征 $\boldsymbol { x } _ { i }$，适配模块的前向计算可表示为：

$$
\boldsymbol {x} _ {o} ^ {\mathrm {S L}} = \left(\boldsymbol {W} _ {0} + \Delta \boldsymbol {W}\right) \boldsymbol {x} _ {i} = \boldsymbol {W} _ {0} \boldsymbol {x} _ {i} + \boldsymbol {x} _ {a} ^ {\mathrm {S L}}. \tag {8}
$$

矩阵 $\pmb { A }$ 使用随机高斯分布进行初始化，而 $\pmb { B }$ 则被初始化为 0，从而使得在训练刚开始时 $\Delta W = 0$。LoRA 的架构如图 4(a) 所示。在整个研究中，LoRA 的单模态实现都被称作标准 LoRA。

2) 提出的 MMLoRA (Proposed MMLoRA)：类似于 MMAdapter，我们将标准 LoRA 扩展以处理多模态任务。如图 4(b) 所示，我们采用具有共享权重的双分支结构来处理多模态信息。这种设计在融合模块的辅助下，实现了在单一模态内部以及跨模态之间对特定任务知识的学习。由于给定的输入为 $\boldsymbol { x } _ { i }$ 和 $\boldsymbol { y } _ { i }$，使用 MMLoRA 生成自适应特征的过程可以描述如下：

$$
\boldsymbol {x} _ {o} ^ {\mathrm {M M L}} = \boldsymbol {W} _ {x 0} \boldsymbol {x} _ {i} + \lambda_ {1} \cdot \boldsymbol {x} _ {a} ^ {\mathrm {M M L}} + (1 - \lambda_ {1}) \boldsymbol {y} _ {a} ^ {\mathrm {M M L}} \tag {9}
$$

$$
\boldsymbol {y} _ {o} ^ {\mathrm {M M L}} = \boldsymbol {W} _ {y 0} \boldsymbol {y} _ {i} + \lambda_ {2} \cdot \boldsymbol {y} _ {a} ^ {\mathrm {M M L}} + (1 - \lambda_ {2}) \boldsymbol {x} _ {a} ^ {\mathrm {M M L}} \tag {10}
$$

其中：

$$
\boldsymbol {x} _ {a} ^ {\text {M M L}} = \boldsymbol {B} _ {x} \boldsymbol {A} _ {x} \boldsymbol {x} _ {i} \tag {11}
$$

$$
\boldsymbol {y} _ {a} ^ {\mathrm {M M L}} = \boldsymbol {B} _ {y} \boldsymbol {A} _ {y} \boldsymbol {y} _ {i}. \tag {12}
$$

最后值得一提的是，图 3 和图 4 中所示的设计可以很容易地推广到两种以上的多模态场景中。

## C. 本文提出的 MFNet (Proposed MFNet)

图 5 展示了提出的 MFNet 全貌，以及两种截然不同的多模态微调策略。MFNet 的输入首先由配备有 MMAdapter 或 MMLoRA 的 SAM 图像编码器进行处理，其负责利用多模态微调机制来提取和融合多模态的遥感特征。其输出随后被送入深度融合模块 (DFM) 中，该模块从编码器接收两个单尺度的多模态输出，并利用金字塔模块将其扩展为两组多尺度的多模态特征。这些高级抽象特征进而通过四个通道注意力 (Squeeze-and-Excitation, SE) 融合模块进行融合，生成了一组多尺度特征。最后，DFM 的输出被传递给解码器以生成分割的预测结果图。在本节中，我们将详细介绍提出的 MFNet 里的各个关键组件。

1) SAM 的图像编码器 (SAM’s Image Encoder)：我们将光学图像及其相应的 DSM 数据分别表示为 $\pmb { X } \in \mathbb { R } ^ { H \times W \times 3 }$ 和 $\textbf { \textit { Y } } \in \mathbb { R } ^ { H \times W \times 1 }$，其中 $H$ 和 $W$ 分别表示输入数据的高度和宽度。采用了非分层 ViT 主干的 SAM 图像编码器，首先将输入嵌入到大小为 $\mathbb { R } ^ { h \times w \times c }$ 的张量中，其中 $h = (H / 16)$，$w = (W / 16)$，$c$ 为嵌入维度。接着，利用堆叠的 ViT 块提取特征，该特征的尺寸在整个编码过程中得以保持不变 [48]。如图 5(a) 所示，$\pmb { X }$ 和 $\textbf { \textit { Y } }$ 均被输入至 SAM 的图像编码器。值得注意的是，相同的 SAM 编码器也被用于 DSM 数据，这证明了 SAM 完全可以被用来从非图像数据中提取特征。SAM 的图像编码器提取并融合了多模态特征，通过多模态微调模块生成了高级抽象特征 $\pmb { F } _ { x } \in \mathbb { R } ^ { h \times w \times c }$ 和 $\pmb { F } _ { y } \in \mathbb { R } ^ { h \times w \times c }$。

![](images/4c3d6e1f124dc847dceaf610b66d2a06b8bd64213ce3db93e535c311c03db1e9.jpg)  
(a)

![](images/cf8d1f76cd9c859c4cb66028433dcf93d0f96fdb46f9b78f4f2ea64eff88496d.jpg)  
(b)

![](images/3f88be0e54bdc6269193e34362ca93711e2cf117d3bb3d7b6d047f209d5eceb2.jpg)  
图 5. (Fig. 5.) (a) 提出的 MFNet 总览图，其由带有强化多模态微调机制的 SAM 图像编码器、DFM（深度融合模块）以及一个通用解码器构成。(b) 附带了 MMAdapter 的 ViT 块结构，以及 (c) 附带了 MMLoRA 的 ViT 块结构。这些构成了 MFNet 的两种不同架构。

2) 带有 MMLoRA 的 ViT 块 (ViT Block With MMLoRA)：图 5(b) 描绘了 ViT 块内 MMAdapter 的架构。与之相对应的是，MMLoRA 则充当了一个与线性层并行应用的多模态微调手段。为了阐述得更为清晰，结合了 MMLoRA 的 ViT 块的结构被展现在了图 5(c) 中。在多头注意力模块中，LoRA 模块被附加在了 $q$ 和 $v$ 投影层上 [49]。在这一阶段，为了专注捕捉各自单一模态下的特定任务信息，排除了多模态的互动。到了随后的 MLP 层中，MMLoRA 模块才被施加在 MLP 的各线性层上，进而促成了多模态信息的融合。

3) 深度融合模块 (DFM)：多尺度特征在语义分割任务里扮演着至关紧要的角色，这是由于密集型预测不仅需要全局的信息，还要局部的信息。如图 6(a) 所示，两个金字塔模块被用于生成多尺度的特征图，其中每个金字塔都包含一组相互平行的卷积或是反卷积操作。若是从尺度大小为 (1/16) 的基础 ViT 特征图起算，我们采用了步长设定为 $\{ (1/4) , (1/2) , 1 , 2 \}$ 的各项卷积操作来产出尺度依次为 {(1/4), (1/8), (1/16), (1/32)} 的特征图，其中包含分数的步长代表反卷积运算 [48]。这些简易的金字塔模块生成了两组兼备多模态与多尺度的特征，将其表示为 $F _ { x } ^ { i }$ 与 $F _ { y } ^ { i }$，这里的 $i = \{ 1 , 2 , 3 , 4 \}$ 代表对应的尺度索引。随后，四个 SE 融合模块 [33] 被用来对这些多模态特征实施更深一步的交融。值得一说的是，要是配备更为高阶的融合模块，是可以取得进一步拔高效果的。

如图 6(b) 所例示的那样，SE 融合模块以聚合多模态特征中的全局信息起步。针对第 i 个融合模块，在输入通道大小为 $C _ { i }$ 的条件下，squeeze-and-excitation (通道注意力) 过程是通过两次核大小为 $1 \times 1$ 的卷积操作去执行的，紧接其后的是 ReLU 和 Sigmoid 激活函数。随后这些多模态输出被赋予相应的权重，并且在对应位置上逐元素相加合并，由此产生了代表强化的融合特征 $F _ { f } ^ { i }$。这四个 SE 融合模块输出的内容合并形成了多尺度融合特征，用 $F _ { f } ^ { I - 4 }$ 表示，它们会被送入解码器以备后续处理。本工作沿用了 UNetformer [35] 方案里的解码器，它凭借聚焦兼顾全局与局部的资讯，将抽象语义信息重新构建为了分割图像图。

![](images/23edea327a9eb4137d32b1fab5da979300f7667eba2be7672b8ed606ca34d2b6.jpg)

![](images/4c924366a07acbef862a77f5afd0648cae1738e9eeade99163df150b26144758.jpg)  
(b)   
图 6. (Fig. 6.) (a) DFM 示意图。其中包含两个在被送入四个 SE 融合模块汇聚之前，用以将多模态特征扩大并展开成多尺度特征的金字塔组。(b) SE 融合模块结构的示意图 [33]。值得注意的是，我们只是采用了目前现有的一个简易的融合模块，而并没有专门为此设计此部位结构，这恰恰证明了我们成果的主要提升正统得益于这套基于视觉基础模型的多模态微调策略。
"""

with open(filepath, "w") as f:
    f.writelines(new_lines)
    f.write(append_text.strip() + "\n")
