# 负采样不建树；短语上降频之后，赢家却是层次 softmax

<!-- release-date: 2013-10-16 -->

**本文依据**：`Distributed Representations of Words and Phrases and their Compositionality`，arXiv:1310.4546v1，2013-10-16 提交，letter，9 页。作者 Tomas Mikolov、Ilya Sutskever、Kai Chen、Greg Corrado、Jeffrey Dean，均署 Google Inc., Mountain View。封面未印会议名。页码指这份 PDF 的页序。标「外部补充」的句子不来自本文。

## 一句话

这篇不发明新的词向量结构。它接在 Word2Vec 那篇的 Skip-gram 后面，换掉三件让训练卡住的事：全词表 softmax、高频词占满的共现、以及单词向量表达不了的习语。负采样把「对整张词表归一化」收成 1 个正例加 $k$ 个噪声词；按频率丢掉高频词，速度大约快 2–10 倍，低频词也更规整；再用共现分数把短语收成独立 token（PDF p. 1–2）。

词类比上，负采样赢过层次 softmax。短语类比上，把高频词降频之后，层次 softmax 反而最高；他们拿来报 72% 的那个模型，用的也是层次 softmax，不是负采样（PDF p. 5–7）。

## 先把三个词说清

- **Skip-gram**：中心词预测窗口里的邻词。这篇不重画它的结构。PDF p. 2 的图 1 只是把上一篇的架构重印了一遍：输入是 $w(t)$，投影之后去预测周围的词。
- **层次 softmax**：输出层是一棵二叉树，词在叶子上。算一个词的概率，只走根到叶那一条路径。
- **负采样**：不建树。正例要像真邻居，再从噪声分布里抽 $k$ 个词，要求它们不像邻居。

完整 softmax 不实用，是因为代价随词表 $W$ 涨，$W$ 常在 $10^{5}$–$10^{7}$（PDF p. 3，式 2）。下面这张图只比较两种替代，不比较它们的实验排名。排名在后文的表里，而且两张表的赢家不一样。

![层次 softmax 只沿根到叶计算约 log2(W) 个内部节点，灰色兄弟支不计算；负采样不建树，一个正例配 k 个红色噪声词，代价是 k+1 次 logistic。](/readings/Word2Vec-NegativeSampling/neg-vs-hs.svg)

上图按 PDF p. 3 的两节重画，不是原文插图。原文没有这张对照图。

## 层次 softmax：代价跟着路径走，不跟着词表走

给定词序列 $w_1,\ldots,w_T$，Skip-gram 最大化平均对数概率（PDF p. 2，式 1）：

$$
\frac{1}{T}\sum_{t=1}^{T}\sum_{-c\le j\le c,\,j\neq 0}\log p(w_{t+j}|w_t)
$$

$c$ 是窗口，可以随中心词变。$c$ 越大，训练例越多，时间也越长。基本定义仍是 softmax（PDF p. 3，式 2）：分母对全部 $W$ 个词求和。$v_w$ 是输入向量，$v'_w$ 是输出向量。

层次 softmax 把输出层做成二叉树，词是叶子。每个内部节点存两个子节点的相对概率，从根走到词 $w$ 就是一次随机游走。路径长度 $L(w)$，平均不超过 $\log W$；要评估的节点大约是 $\log_2 W$ 个，而不是 $W$ 个（PDF p. 3）。概率是路径上各次 logistic 的乘积（式 3）：

$$
p(w|w_I)=\prod_{j=1}^{L(w)-1}\sigma\Big([[n(w,j+1)=\mathrm{ch}(n(w,j))]]\cdot {v'_{n(w,j)}}^\top v_{w_I}\Big)
$$

$\sigma(x)=1/(1+e^{-x})$。$[[x]]$ 为真取 1，否则取 $-1$。对全部词求和为 1。和标准 softmax 不同：每个词只留输入向量 $v_w$，输出向量 $v'_n$ 放在内部节点上（PDF p. 3）。

树的形状会影响速度和精度。本文用二叉 Huffman 树，高频词码长短。按频率把词捆在一起，此前已被当作神经网络语言模型的简单加速，不是这篇的新发现（PDF p. 3）。

## 负采样：只要样本，不要噪声概率

NCE 用 logistic 回归把数据从噪声里分开（Gutmann 与 Hyvärinen；Mnih 与 Teh 用到语言模型）。这接近 Collobert 与 Weston 用 hinge 把数据排在噪声之上（PDF p. 3）。NCE 可以近似最大化 softmax 的对数概率。Skip-gram 只关心向量好不好，作者因此把 NCE 再简化。负采样替换目标里的每一个 $\log P(w_O|w_I)$（PDF p. 3，式 4）：

$$
\log\sigma({v'_{w_O}}^\top v_{w_I})+\sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n(w)}\big[\log\sigma(-{v'_{w_i}}^\top v_{w_I})\big]
$$

正例的点积经 sigmoid 要靠近 1；抽来的噪声词，点积取负号之后也要靠近 1。小数据集上 $k$ 取 5–20 有用，大数据集上 2–5 就够（PDF p. 4）。

和 NCE 的差别只剩这一句：NCE 既要样本，也要噪声分布的数值概率；负采样只要样本。近似最大化 softmax 对数概率这件事，对学向量不重要（PDF p. 4）。

噪声分布仍是自由参数。他们试过的选择里，把 unigram $U(w)$ 升到 $3/4$ 次方再归一化，即 $U(w)^{3/4}/Z$，显著好于 unigram 和均匀分布。NCE 与负采样、以及未在文中给数字的语言建模，都是这个结论（PDF p. 4）。语言建模的具体分数本文没有。

## 降频：少看 the，把梯度留给低频词

超大语料里 in、the、a 能出现上亿次。Skip-gram 从 France 与 Paris 的共现里学得到东西，从 France 与 the 里学得少，因为几乎每个词都和 the 同句出现。反过来，高频词自己的向量看过几百万次之后几乎不再变（PDF p. 4）。

每个词 $w_i$ 被丢掉的概率是（式 5，公式从 PDF p. 4 底部排到 p. 5）：

$$
P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}
$$

$f(w_i)$ 是该词的频率，$t$ 通常约 $10^{-5}$（PDF p. 5）。频率高于 $t$ 的词会被狠丢，频率排序仍保留。公式是启发式的。引言里的加速幅度是大约 2–10 倍，写在 PDF p. 2，不是封面摘要。

它改的是哪些共现对进入训练，不是随机删掉半篇语料。

## 词类比：负采样赢，降频主要换时间和语义

评测沿用上一篇的类比：Germany : Berlin :: France : ?，找余弦距离上最接近 $\mathrm{vec}(\text{Berlin})-\mathrm{vec}(\text{Germany})+\mathrm{vec}(\text{France})$ 的词，搜索时丢掉输入词。句法如 quick : quickly :: slow : slowly，语义如国家到首都（PDF p. 5）。

训练数据是 Google 内部新闻，约 10 亿词。出现少于 5 次的词丢掉，词表 692K。下表是 300 维 Skip-gram（PDF p. 5，表 1）。

| 方法 | 时间 [min] | 句法 [%] | 语义 [%] | 总准确率 [%] |
|---|---|---|---|---|
| NEG-5 | 38 | 63 | 54 | 59 |
| NEG-15 | 97 | 63 | 58 | 61 |
| HS-Huffman | 41 | 53 | 40 | 47 |
| NCE-5 | 38 | 60 | 45 | 53 |
| NEG-5，$10^{-5}$ 降采样 | 14 | 61 | 58 | 60 |
| NEG-15，$10^{-5}$ 降采样 | 36 | 61 | 61 | 61 |
| HS-Huffman，$10^{-5}$ 降采样 | 21 | 52 | 59 | 55 |

没有降采样时，NEG-15 的总准确率 61%，高于 HS 的 47% 和 NCE-5 的 53%。加上降采样，NEG-5 从 38 分钟降到 14 分钟，总准确率从 59% 到 60%；HS 的语义从 40% 升到 59%，总准确率从 47% 到 55%，仍低于负采样。作者的判断停在这一档任务上：负采样超过层次 softmax，也略好于 NCE；降频让训练快数倍，表示也更准（PDF p. 5）。

有人会说 Skip-gram 本身是线性的，所以才适合线性类比。作者指向上一篇：高度非线性的 sigmoid 循环网，数据变多后这类任务也会明显变好。那是对「只有 Skip-gram 才会加减」的限制，不是这篇的新实验（PDF p. 5）。

国家到首都的几何，论文用一张 PCA 图说明，不是用上表的百分比说明。

![原文 Figure 2：1000 维 Skip-gram 向量做二维 PCA 后，国家落在左侧、首都落在右侧，国家到首都的偏移大致同向。图注写明训练时没有提供「首都」是什么的监督。](/readings/Word2Vec-NegativeSampling/figure2-pca-capitals.png)

引自原文 Figure 2，PDF p. 4。图注只写 1000 维 Skip-gram，没有写这些向量是负采样训的还是层次 softmax 训的。不要把它读成表 1 的消融图。

## 短语：先收成 token，降频之后层次 softmax 翻盘

许多短语不是词义的简单相加。Canada 和 Air 拼不出 Air Canada；Boston Globe 是报纸，不是波士顿加地球（PDF p. 1–2）。做法是先找出经常一起出现、在别处很少乱配的词串，换成唯一 token。New York Times、Toronto Maple Leafs 会被替换；this is 不动（PDF p. 5）。

分数用 unigram 与 bigram 计数（PDF p. 6，式 6）：

$$
\mathrm{score}(w_i,w_j)=\frac{\mathrm{count}(w_i w_j)-\delta}{\mathrm{count}(w_i)\times\mathrm{count}(w_j)}
$$

$\delta$ 挡住由极低频词凑出来的假短语。分数超过阈值的 bigram 收成短语。通常对语料跑 2–4 遍，阈值递减，让更长的短语也能形成。短语识别文献很多，本文不做对比（PDF p. 6）。

新测试集有 3218 题，五类例子是报纸、NHL、NBA、航空公司和公司高管。典型题是 Montreal : Montreal Canadiens :: Toronto : Toronto Maple Leafs（PDF p. 2、p. 6）。题集在 `questions-phrases.txt`。

同一份约 10 亿词的新闻，先建成短语语料，再训 300 维、窗口 5 的 Skip-gram（PDF p. 6，表 3）。

| 方法 | 维数 | 无降采样 [%] | $10^{-5}$ 降采样 [%] |
|---|---|---|---|
| NEG-5 | 300 | 24 | 27 |
| NEG-15 | 300 | 27 | 42 |
| HS-Huffman | 300 | 19 | 47 |

不降采样时，层次 softmax 最差，19%。降采样之后它变成最好，47%，超过 NEG-15 的 42%。这和表 1 的方向相反。作者把这句话写明了：降采样有时既能加速又能提精度；算法选择是任务相关的（PDF p. 6，结论在 p. 8）。

为冲这一档的精度，他们把数据加到约 330 亿词，用层次 softmax、1000 维、整句当上下文，准确率 72%。减到 60 亿词则是 66%（PDF p. 7）。报 72% 的不是负采样。

表 4 是人工看低频短语的近邻，不是自动指标。带 $10^{-5}$ 降采样时，NEG-15 把 Vasco de Gama 的近邻给成 Lingsugur，层次 softmax 给成 Italian explorer；Lake Baikal 一边是 Great Rift Valley，另一边是 Aral Sea（PDF p. 7）。作者说这和表 3 一致：短语的最好表示来自层次 softmax 加降采样。这是他们看过样本之后的判断。

## 逐元素相加是另一种线性，不是类比的同一件事

类比是平移。另一件是把两个向量逐元素相加。封面附近的例子是 $\mathrm{vec}(\text{Russia})+\mathrm{vec}(\text{river})$ 靠近 Volga River（PDF p. 2）。表 5 用最好的 Skip-gram 列出四个最近邻，查询词并不都和这个例子相同（PDF p. 7）：

| 查询 | 最近的四个 token |
|---|---|
| Czech + currency | koruna；Check crown；Polish zolty；CTK |
| Vietnam + capital | Hanoi；Ho Chi Minh City；Viet Nam；Vietnamese |
| German + airlines | airline Lufthansa；carrier Lufthansa；flag carrier Lufthansa；Lufthansa |
| Russian + river | Moscow；Volga River；upriver；Russia |
| French + actress | Juliette Binoche；Vanessa Paradis；Charlotte Gainsbourg；Cecile De |

Russian + river 的第一名是 Moscow，Volga River 排第二。相加有时有意义，不是每次第一名都对。

作者给的机制是对训练目标的解读，不是单独的定理：词向量与 softmax 之前的输入成线性关系，向量刻画该词出现的上下文分布；这些值与输出层概率成对数关系，两向量之和对应两套上下文分布的乘积。乘积起 AND 的作用。若 Volga River 常与 Russian 和 river 同句出现，这两个向量之和就会靠近 Volga River（PDF p. 7）。对有句法结构的长句，他们把本文定位成递归矩阵–向量方法的补充，不是替代（PDF p. 8）。

## 和已发表向量比，比的是低频词的近邻

Collobert 与 Weston、Turian 等人、Mnih 与 Hinton 的向量可以下载。上一篇已经在词类比上比过，Skip-gram 大幅领先（PDF p. 7）。这篇改看低频词的最近邻。空单元格表示该词不在对方词表里（PDF p. 8，表 6）。

| 模型 | Redmond | Havel | ninjutsu | graffiti | capitulate |
|---|---|---|---|---|---|
| Collobert，50 维，2 个月 | conyers；lubbock；keene | plauen；dzerzhinsky；osterreich | reiki；kohona；karate | cheesecake；gossip；dioramas | abdicate；accede；rearm |
| Turian，200 维，数周 | McCarthy；Alston；Cousins | Jewell；Arzu；Ovitz | — | gunfire；emotion；impunity | — |
| Mnih，100 维，7 天 | Podhurst；Harlang；Agarwal | Pontiff；Pinochet；Rodionov | — | anaesthetics；monkeys；Jews | Mavericks；planning；hesitated |
| Skip-Phrase，1000 维，约 1 天 | Redmond Wash.；Redmond Washington；Microsoft | Vaclav Havel；president Vaclav Havel；Velvet Revolution | ninja；martial arts；swordsmanship | spray paint；grafitti；taggers | capitulation；capitulated；capitulating |

Skip-Phrase 用了短语，训练词超过 300 亿，正文又写成约 300 亿（PDF p. 8）。作者把差距一部分归到数据量：比先前典型规模高两到三个数量级；尽管数据大得多，训练时间只是先前架构的一小部分。这是定性近邻，不是表 1 那套准确率。

同一套技术也可以用于上一篇的 CBOW。开源代码在 `code.google.com/p/word2vec`（PDF p. 8）。

## 资料与阅读边界

最影响性能的决定，作者自己列为：模型架构、向量维度、降采样率、训练窗口（PDF p. 8）。词类比上负采样更强，短语类比在降采样之后层次 softmax 更强，冲 72% 时用的也是层次 softmax。两张表不要合成「负采样全面更好」。

未报告的数字不要补：语言建模上 $U(w)^{3/4}$ 更好，但分数不在本文。短语识别不与已有方法对比。图 2 的 1000 维向量没有写训练目标。NCE 的数值噪声概率被故意丢掉，所以负采样不是一个归一化的生成模型。
