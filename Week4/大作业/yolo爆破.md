# YOLO 系列逐代突破总结

| Model | Anchor | Input | Backbone | Neck | Predict / Train |
|---|---|---|---|---|---|
| YOLOv1 | 无真正意义上的 Anchor；将图像划分为 7×7 网格，每个网格预测 2 个框 | 448×448 | 类 GoogLeNet 风格网络，24 个卷积层 + 2 个全连接层 | 无 | 直接回归 bbox 和类别概率；使用 IoU 相关损失思想；推理时使用 NMS；每个网格主要负责一个类别，定位和小目标能力较弱 |
| YOLOv2 | 有 Anchor；13×13 网格，每个网格预测 5 个 anchor；通过 K-means 聚类得到先验框 | 416×416；支持多尺度训练 | Darknet-19（19 个卷积层 + 5 个最大池化层，广泛使用 BN 和 ReLU） | 无明显独立 Neck；引入 Passthrough Layer 做浅层与深层特征融合 | 基于 Anchor 预测相对偏移；使用 NMS；支持多尺度训练；提升召回率和定位能力 |
| YOLOv3 | 有 Anchor；通常 9 个 Anchor，分配到 3 个尺度上预测 | 416×416 / 608×608 | Darknet-53（53 个卷积层，引入残差连接，使用 BN 和 Leaky ReLU） | FPN（多尺度特征融合） | 多尺度检测；使用 Logistic 分类器替代 Softmax；推理时使用 NMS；对小目标检测更友好 |
| YOLOv4 | 有 Anchor | 常用 608×608 | CSPDarknet53（引入 CSP 结构减少计算冗余，结合 Mish 激活、DropBlock、CmBN 等） | SPP + PANet | 使用 CIoU Loss；推理时常配合 DIoU-NMS；训练中引入 Mosaic、SAT、标签平滑等策略，精度和速度进一步提升 |
| YOLOv5 | 有 Anchor | 常用 640×640 | CSP-based Backbone，早期版本含 Focus 模块，使用 BN 和 Leaky ReLU / SiLU | FPN + PAN（工程实现中常写作 PAN-FPN） | 使用 GIoU / CIoU Loss；自动计算 Anchor；支持 Mosaic、MixUp、自适应缩放等增强；推理时使用 NMS |
| YOLOX | 无 Anchor（Anchor-Free） | 常用 640×640 | CSPDarknet 系列 / Darknet 风格主干（不同版本略有差异） | PAFPN / FPN 风格特征融合 | Anchor-Free；Decoupled Head；使用 SimOTA 标签分配；常配合 IoU Loss 与 NMS；减少 Anchor 设计复杂度 |
| YOLOv6 | 无 Anchor（Anchor-Free） | 常用 640×640 | EfficientRep Backbone（基于重参数化思想优化部署效率） | Rep-PAN Neck | 使用 Efficient Decoupled Head；采用 SimOTA 风格标签分配；常见 SIoU/IoU 类损失；面向工业部署优化 |
| YOLOv7 | 有 Anchor（主流版本） | 常用 640×640 | Extended-ELAN / E-ELAN 结构，结合重参数化与高效层连接设计 | SPPCSPC + PAN | 使用辅助头进行训练；标签分配和损失函数进一步优化；推理时使用 NMS；兼顾速度与精度 |
| YOLOv8 | 无 Anchor（Anchor-Free） | 常用 640×640 | 使用 C2f 模块的 Backbone，增强梯度流与特征表达能力 | FPN + PAN 风格融合结构 | Decoupled Head；使用 TAL（Task-Aligned Assigner）标签分配；采用 DFL + IoU 类损失；推理时使用 NMS，工程易用性强 |