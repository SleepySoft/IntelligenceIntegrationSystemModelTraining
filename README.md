# IntelligenceIntegrationSystemModelTraining

Split [IIS](https://github.com/SleepySoft/IntelligenceIntegrationSystem) model training into a standalone project.

## Overview

由于模型训练数据与结果评估数据文件较大，所以将训练单独拆分为一个独立项目。

主项目地址：[IntelligenceIntegrationSystem](https://github.com/SleepySoft/IntelligenceIntegrationSystem)

## Structure and Usage

## Usage

通用的训练流程分成三部分：准备数据集，模型训练，模型评估。分别对应以下三个文档：

[01.PrepareDataset.md](01.PrepareDataset.md)

[02.ModelTraining.md](02.ModelTraining.md)

[03.ResultsEvaluation.md](03.ResultsEvaluation.md)

如果你想开始训练，请从01文档开始阅读。如果你只想浏览模型的训练结果，请直接看03文档。

和具体硬件平台相关的部分在 [TrainingScripts](TrainingScripts) 对应环境的目录中。

#### Data

[Data](Data)目录为已生成好的训练数据，以及新模型的评估结果。大家可以直接使用该数据集进行训练，或者直接查看评估结果。

#### Training Scripts

[TrainingScripts](TrainingScripts)目录为模型训练脚本，不同目录对应着不同的硬件环境。其中的Memo目录存放微调训练的记录。

#### Data Preprocessing

[preprocess1_extract_clean.py](preprocess1_extract_clean.py)

[preprocess2_sampling_split.py](preprocess2_sampling_split.py)

[preprocess3_generate_alpaca.py](preprocess3_generate_alpaca.py)

按照 [01.PrepareDataset.md](01.PrepareDataset.md) 说明，依次调用这三个脚本生成训练数据集。

#### Data Evaluation

[validation1_batch_inference.py](validation1_batch_inference.py)

[validation2_review_app.py](validation2_review_app.py)

[validation3_auto_eval.py](validation3_auto_eval.py)

阅读 [03.ResultsEvaluation.md](03.ResultsEvaluation.md) 获取这三个脚本的使用方法。


## Train Record

#### 20251214

数据：[v1](Data/v1)

记录：[FinetuneRecord_20251214.md](TrainingScripts/Linux-MI50-Duo%2FMemo/FinetuneRecord_20251214.md)

总结：第一次训练，结果并不令人满意。可能是训练集中被判定为“无价值”的样本太多，也可能与我在生成训练集时故意降低文章评分（期望评分更严格）有关。
