Code for Grouped Knowledge Distillation for Deep Face Recognition (AAAI2023)

The loss.py has been uploaded and it's recommended to replace the vanilla loss function by GKD loss. [Tface](https://github.com/Tencent/TFace/tree/master/recognition/tasks/ekd)

We also provide a concise training code for a quick start. Replace the dataset definition with yours, an optimal definition of dataset is [insightface](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/dataset.py.)

The code for testing on MegaFace is based on [Insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_evaluation_/megaface). Also, I have uploaded the code to obtain the testing results tpr@far=1e-x named [tpr.py](https://github.com/WisonZ/GKD/blob/main/tpr.py).
