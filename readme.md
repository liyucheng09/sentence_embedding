# 使用

```
from demo import SentenceEmbedding

demo=SentenceEmbedding(model_path)

demo.get_embeddings(['我不知道', '我是猪'])

>>> [[-0.01397967  0.03157582  0.03531738 ... -0.01741204  0.05552316
   0.09307685]
 [ 0.00039342  0.04293009  0.04313357 ... -0.01089463 -0.04951692
   0.04515894]]
```


# 模型和镜像

- `model_path=/cfs/cfs-dtmr08t1/bert-base-chinese-local`

- cfs挂载路径：`/cfs/cfs-dtmr08t1/`

- 可用的镜像：`venus-c-yuchengli-transformers4.1.1-datasets1.2.0-2`

- 镜像网址：`registry2sz.sumeru.mig/bsimage/venus-c-yuchengli-transformers4.1.1-datasets1.2.0-2:0.1.4`