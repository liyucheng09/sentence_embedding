# FAQ精排模型

本项目包括FAQ向量精排与FAQ交互式精排。

## FAQ向量精排

FAQ向量精排通过计算余弦相似度选择输入对应的标准问。

获取向量的方式如下：

```
from demo import SentenceEmbedding

model = SentenceEmbedding(model_path, kernel_bias_path='kernel_path/', pool='cls')

demo.get_embeddings(['我不知道', '我是猪'])

>>> [[-0.01397967  0.03157582  0.03531738 ... -0.01741204  0.05552316
   0.09307685]
 [ 0.00039342  0.04293009  0.04313357 ... -0.01089463 -0.04951692
   0.04515894]]
```

具体需要查看`demo.py`中对象的定义和`get_embeddings`方法的参数。

目前最优的方案是对比学习+白化。

对比学习模型`simcse`模型文件在`/cfs/cfs-dtmr08t1/simcse`, 白化所需要的`kernel`和`bais`在`/cfs/cfs-dtmr08t1/simcse/kernel_path`中。

## FAQ交互式精排

FAQ交互式精排将输入与标准问拼接后输入模型，判断其匹配的可能性。

使用方法见`rerank_after_recall.py`

模型文件在`/cfs/cfs-dtmr08t1/rerank`

# 项目的架构

本项目包括了FAQ精排的全流程。

## 模型文件

`glove.py`基于词向量的句向量获取方法。

`model.py`
- `SentencePairEmbedding`: `SentenceBert`模型。
- `SimCSE`: 对比学习
- `rerank`: 交互式精排，即句子对的分类
- `SingleSentenceEmbedding`: 可以通过torch.jit包装的句向量model。用于venus trpc部署。

## 评测

`eval.py`针对句子对数据（学术界STS任务）的评测脚本。

`matching.py`直接使用精排模型检索答案（匹配知识库中所有query，取topk）。

`after_recall.py`在ES检索基础上使用向量精排进一步获得结果。

`rerank_after_recall.py`在ES检索基础上使用交互式精排进一步获得结果。

## 部署

`demo.py`对模型包装后的demo。

`venus.py` Venus TRPC部署的脚本。使用torch.jit保存模型。具体参考我的腾讯文档。

**本项目某些文件依赖于lyc库，请查看我的git**