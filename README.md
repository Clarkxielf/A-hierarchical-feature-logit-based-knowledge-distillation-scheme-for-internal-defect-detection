## A-hierarchical-feature-logit-based-knowledge-distillation-scheme-for-internal-defect-detection
This repo is the implementation of the paper : A hierarchical feature-logit-based knowledge distillation scheme for internal defect detection of magnetic tiles.

Abstract
=
Magnetic tiles are the key components of various electrical and mechanical systems in modern industry, and detecting their internal defects holds immense significance in maintaining system performance and ensuring operational safety. Recently, deep learning has emerged as a leading approach in pattern recognition due to its strong capability of extracting latent information. In practical scenarios, there is a growing demand for embedding deep learning algorithms in edge devices to enable real-time decision-making and reduce data communication costs. However, a powerful deep learning algorithm with high complexity is impractical for deployment on edge devices with limited memory capacity and computational power. To overcome this issue, we propose a novel knowledge distillation method, entitled hierarchical feature-logit-based knowledge distillation, to compress deep neural networks for internal defect detection of magnetic tiles. Specifically, it comprises a one-to-all feature matching for disparate feature knowledge distillation, a logit separation for relevant and irrelevant logit knowledge distillation, and a parameter value prediction network for seamlessly fusing feature and logit knowledge distillation. Besides, an ingenious hierarchical distillation mechanism is designed to address the capacity gap issue between the teacher and the student. The extensive experimental results show the effectiveness of our proposed model. The code is available at  https://github.com/Clarkxielf/Hierarchical-feature-logit-based-knowledge-distillation-scheme.git.

Highlights
=
1)	A novel similarity feature matching tensor is constructed for disparate feature knowledge distillation.
2)	We reveal that both of feature and logit distillation are indispensable ingredient to boost the success of knowledge transferring from the teacher to guide the training of the student.
3)	A parameter value prediction network is proposed to search the best values in a practically infinite combination space.
4)	An ingenious hierarchical distillation mechanism is designed to address the capacity gap issue.

Contributions
=
(1)	To fill the semantic gap between teacher and student networks, a similarity feature matching tensor is created along the channel dimension, which can avoid directly calculating the summation of the distance between the teacher feature map and the student feature map in a one-to-one spatial matching fashion. Subsequently, the semantics of the student's features are reconstructed using a multi-head attention model, enabling the model to selectively emphasize information from various representation subspaces across different spatial positions.
(2)	We revisit feature and logit distillation and reveal that both are indispensable ingredients to boost the success of knowledge transfer from the teacher to guide and refine the student training process. Therefore, a novel unified framework for simultaneously distilling features and logits is proposed, and a secondary parameter value prediction network is proposed to balance their importance, which is very powerful for searching for the best values in a practically infinite combination space.
(3)	We propose an ingenious hierarchical distillation mechanism to address the capacity gap issue between teacher and student networks. Consequently, despite the teacher model having a significantly larger parameter size compared to the student network, effective knowledge transfer from the teacher network to the student network can still be achieved.

The overall framework of the proposed method
=
![image](C:\Users\C\Desktop\框架图.png)
