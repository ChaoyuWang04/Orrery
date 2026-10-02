# 统计量进了计算图，层输入才钉得住

<!-- release-date: 2015-02-11 -->

> 本文依据本地 `readings/_src/深度学习基石/BatchNorm.pdf`，即 **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**，arXiv:1502.03167v3、封面日期 2015-03-02，共 11 页。作者 Sergey Ioffe、Christian Szegedy，均署 Google Inc.。页码均指 PDF 自身的页码。文中会区分三件事：**报告明确写了什么**、**我们怎么解释它**、**哪些是外部资料**。

## 读前先把几个词说成人话

这篇难读的地方不是减均值，而是它坚持：减均值必须发生在计算图里面。后面所有加速数字，都踩在这件事上。

- **协变量偏移（covariate shift）**：一个学习系统的输入分布变了，它就得重新适应。通常这是整网的域适应问题。
- **内部协变量偏移（Internal Covariate Shift）**：训练时网络参数一变，内部激活的分布跟着变。论文把整网的那个词推进到子网和单层。
- **白化**：把输入线性变成零均值、单位方差，再去掉相关。完整白化要协方差和逆平方根，每步都做太贵。
- **mini-batch**：一次更新用的那一小批样本。BN 的均值和方差就在这批上估，并且梯度要穿过它们。
- **$\gamma$、$\beta$**：归一化之后再学的缩放和平移。没有它们，标准化会悄悄砍掉一层能表示的函数。
- **饱和非线性**：sigmoid 这类，输入绝对值一大，导数就趋近零。

后来把这一层接到每个卷积之后的做法，见 ResNet 一篇。

## 一句话先说清

深度网难训，不只是因为损失非凸。每一层的输入都受前面所有层的参数影响，小改动会沿深度放大。输入分布一变，后面的层就要不断适应新分布。作者把这种漂叫做内部协变量偏移（PDF p.1–2）。

![训练时在 mini-batch 上减均值、除标准差，再学 γ 和 β；推理时改用多个训练 batch 的总体统计，并合成一条直线。BN 插在仿射之后、非线性之前。](/readings/BatchNorm/bn-train-infer.svg)

加到当时最强的 ImageNet 分类网上，同样精度只需约十四分之一的训练步；再改学习率与正则，单网超过原 Inception。六个 BN 网集成后，验证集 top-5 错误 4.9%，测试集 4.82%（ILSVRC 服务器）。摘要把测试错误写成 4.8%，以正文和 Figure 4 脚注的 4.82% 为准（PDF p.1、p.7–8）。

它解决的不是「再发明一种白化」，而是 2015 年初那条卡死的路：层输入一边学一边改分布，只能用小学率、小心初始化，饱和非线性几乎训不动。把归一化变成可微的一层，分布才钉得住。

### 一条阅读路线

1. **p.2 的偏置例子**：统计量若在梯度外面，更新会被抵消，参数空涨。
2. **p.3 算法 1，p.4 算法 2**：训练用 batch 统计，推理用总体统计，再合成线性。
3. **p.5 第 3.2–3.3 节**：插在非线性之前；卷积按特征图共享；为什么学习率可以加大。
4. **p.5–6 Figure 1**：MNIST 上看分布漂没漂。曲线没有印终点精度。
5. **p.7 Figure 3**：到达 72.2% 的步数，以及各自的最高精度。
6. **p.8 Figure 4**：集成与此前结果。附录 Figure 5 是这座 Inception 的规格，不是准确率。

## 主要矛盾：前面一层一动，后面一层就要重新适应

随机梯度下降按步更新参数，每步用大小为 $m$ 的 mini-batch 估梯度。批量相对单样本有两处好处：梯度更接近全训练集；现代硬件上批量计算更并行（PDF p.1）。

难处在超参。学习率和初始值都要小心调。把网络写成 $\ell = F_2(F_1(u, \Theta_1), \Theta_2)$。学 $\Theta_2$ 时，可以把 $x = F_1(u, \Theta_1)$ 看成子网的输入。对 $F_2$ 做一步梯度下降，和单独训 $F_2$ 完全一样。因此「训练与测试同分布」这类对整网有利的性质，对子网同样有利。$x$ 的分布若能固定，$\Theta_2$ 就不必反复补偿上游的漂移（PDF p.1–2）。

饱和非线性把这件事放大。sigmoid 层 $z = g(Wu + b)$，$|x|$ 一大，导数就趋零。$x = Wu + b$ 又被 $W$、$b$ 和更下层一起推着走，很多维会滑进饱和区。实务上靠 ReLU、小心初始化和小学率硬扛（PDF p.2）。

![若均值在梯度外面算，更新偏置后再减新均值，输出不变、偏置空涨；归一化必须写进计算图，梯度才看得见统计量对参数的依赖。](/readings/BatchNorm/bn-gradient-cancel.svg)

完整白化还贵：要算协方差、逆平方根，以及这些变换的导数。按单样本或按图像位置算统计会丢掉激活的绝对尺度。作者要的是：相对整个训练数据的统计来归一化，从而保住网络里的信息（PDF p.2–3）。

**我们如何解释**。上图左栏是存在性的反例，不是一条训练曲线。它只说明「先更新、再在梯度外重算统计」可以让损失完全不动。后文的加速数字，不能反过来证明内部协变量偏移已被消干净；论文自己写的是朝减少它迈一步（PDF p.2）。

## 两条简化：按维标准化，再用 mini-batch 估

完整白化又贵又不是处处可微，于是两条简化（PDF p.3）。

第一，不联合白化，只对每个标量特征独立做零均值、单位方差。$d$ 维输入 $x = (x^{(1)}, \ldots, x^{(d)})$ 的每一维：

$$
\hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{\mathrm{Var}[x^{(k)}]}}
$$

期望和方差本应在全训练集上算。即便不去相关，这种归一化也能加快收敛（PDF p.3，引 LeCun 等人）。

只标准化会改层能表示的函数。sigmoid 的输入若被钉在零附近，就几乎落在线性区。所以插入的变换必须能表示恒等：

$$
y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}
$$

令 $\gamma^{(k)} = \sqrt{\mathrm{Var}[x^{(k)}]}$、$\beta^{(k)} = \mathbb{E}[x^{(k)}]$，就能还原原始激活，如果那才是最优的（PDF p.3）。

第二，随机优化用不了每步全数据统计。mini-batch 本身就能估每维均值和方差，于是这些统计量完整进入反传。按维方差而不是联合协方差，还有一层实际原因：batch 往往小于激活维数，联合协方差容易奇异，必须再正则（PDF p.3）。

对某个激活，mini-batch $\mathcal{B} = \{x_{1 \ldots m}\}$。算法 1 里 $\epsilon$ 加在方差上，保证数值稳定（PDF p.3）：

$$
\begin{aligned}
\mu_{\mathcal{B}} &\leftarrow \frac{1}{m} \sum_{i=1}^{m} x_i \\
\sigma_{\mathcal{B}}^{2} &\leftarrow \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^{2} \\
\hat{x}_{i} &\leftarrow \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2} + \epsilon}} \\
y_i &\leftarrow \gamma \hat{x}_i + \beta
\end{aligned}
$$

注意除数是 $m$，不是 $m-1$。$y = \mathrm{BN}_{\gamma,\beta}(x)$ 并不按样本独立处理：$x$ 依赖当前样本，也依赖 batch 里别的样本。忽略 $\epsilon$、且 batch 内样本同分布时，$\hat{x}$ 期望为 0、方差为 1。联合分布仍可能变，但固定一二阶矩应能加快子网、从而加快整网（PDF p.3–4）。

训练时要用链式法则把损失 $\ell$ 反传到 $x_i$、$\mu_{\mathcal{B}}$、$\sigma_{\mathcal{B}}^{2}$、$\gamma$、$\beta$。印刷公式在 PDF 第 4 页，化简前为：

$$
\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma
$$

$$
\frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot (x_i - \mu_{\mathcal{B}}) \cdot \frac{-1}{2} (\sigma_{\mathcal{B}}^{2} + \epsilon)^{-3/2}
$$

$$
\frac{\partial \ell}{\partial \mu_{\mathcal{B}}} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_{\mathcal{B}}^{2} + \epsilon}} + \frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}} \cdot \frac{\sum_{i=1}^{m} -2(x_i - \mu_{\mathcal{B}})}{m}
$$

$$
\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_{\mathcal{B}}^{2} + \epsilon}} + \frac{\partial \ell}{\partial \sigma_{\mathcal{B}}^{2}} \cdot \frac{2(x_i - \mu_{\mathcal{B}})}{m} + \frac{\partial \ell}{\partial \mu_{\mathcal{B}}} \cdot \frac{1}{m}
$$

$$
\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i, \qquad \frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i}
$$

BN 因此是可微变换。学到的仿射还能表示恒等，容量保住（PDF p.4）。

### 可迁移启发

凡是「先统计再变换」的层，都要问一句：统计量有没有进反传。没进，就可能出现偏置空涨那种抵消。标准化之后一定要留 $\gamma$、$\beta$，否则非线性可能被钉在线性区。

## 训练用 batch，推理用总体统计

要对网络做 BN，选定一批激活，按算法 1 插入变换；原先吃 $x$ 的层改吃 $\mathrm{BN}(x)$。可用 batch 梯度、或 $m > 1$ 的 SGD，以及 Adagrad 这类变体（PDF p.4）。

训练时依赖 mini-batch 是为了效率；推理既不必要、也不可取，输出应只由输入决定。训完后改用总体统计（PDF p.4）：

$$
\hat{x} = \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}}
$$

忽略 $\epsilon$ 时，与训练时同样是均值 0、方差 1。无偏方差估计为 $\mathrm{Var}[x] = \frac{m}{m-1} \mathbb{E}_{\mathcal{B}}[\sigma_{\mathcal{B}}^{2}]$，期望对大小为 $m$ 的训练 batch 取。也可用滑动平均跟踪训练中的精度。算法 2 的推理步骤用的是上面这个无偏平均，不是滑动平均（PDF p.4）。

均值方差冻结之后，归一化就是对每个激活的线性变换，再与 $\gamma$、$\beta$ 合成一条直线，替换 $\mathrm{BN}(x)$（PDF p.4，算法 2）：

$$
y = \frac{\gamma}{\sqrt{\mathrm{Var}[x] + \epsilon}} \, x + \left( \beta - \frac{\gamma \, \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \right)
$$

其中 $\mathbb{E}[x] \leftarrow \mathbb{E}_{\mathcal{B}}[\mu_{\mathcal{B}}]$，$\mathrm{Var}[x] \leftarrow \frac{m}{m-1} \mathbb{E}_{\mathcal{B}}[\sigma_{\mathcal{B}}^{2}]$。

### 卷积：整张特征图共用一对 $\gamma$、$\beta$

作者聚焦「仿射再逐元非线性」：$z = g(Wu + b)$。BN 紧挨非线性之前，归一化的是 $x = Wu + b$。也可以归一化 $u$，但 $u$ 往往已是非线性输出，分布形状会变，只钉一二阶矩消不掉协变量偏移。$Wu + b$ 更对称、更不稀疏、更像高斯，归一化后分布更稳（PDF p.4–5）。

偏置 $b$ 会被减均值消掉，角色由 $\beta$ 接管。于是写成 $z = g(\mathrm{BN}(Wu))$（PDF p.5）。

卷积还要遵守卷积性质：同一特征图不同位置用同一套归一化。mini-batch 里该特征图所有位置的激活合在一起当成 $\mathcal{B}$。batch 大小 $m$、特征图 $p \times q$ 时，有效大小 $m' = m \cdot p q$。每张特征图只学一对 $\gamma^{(k)}$、$\beta^{(k)}$，不是每个空间位置一对。推理时对给定特征图的每个激活用同一线性变换（PDF p.5）。

## 为什么学习率可以加大，Dropout 可以减弱

传统深度网学习率太大，梯度会爆或消失，也会卡在差的局部极小。BN 钉住各层激活，避免参数的小改动被放大成激活和梯度上的大改动，也不那么容易进饱和区（PDF p.5）。

对参数尺度也更稳。对标量 $a$，

$$
\mathrm{BN}(Wu) = \mathrm{BN}((aW)u)
$$

$$
\frac{\partial \mathrm{BN}((aW)u)}{\partial u} = \frac{\partial \mathrm{BN}(Wu)}{\partial u}, \qquad \frac{\partial \mathrm{BN}((aW)u)}{\partial (aW)} = \frac{1}{a} \cdot \frac{\partial \mathrm{BN}(Wu)}{\partial W}
$$

尺度不影响层的雅可比，因而不影响梯度传播；权重越大，对权重的梯度越小，参数增长被稳住（PDF p.5）。

作者进一步猜想：BN 可能让层雅可比的奇异值靠近 1。若相邻两层输入都已归一化，变换近似线性，且归一化向量近似高斯、不相关，则 $JJ^{T} = I$，奇异值全为 1。真实变换非线性，归一化值也不保证高斯或独立。精确影响留待后续（PDF p.5）。这是猜想，不是定理。

正则方面：训练时一个样本总是和 batch 里别人一起出现，网络对单样本不再给出确定性值。实验里这对泛化有利。Dropout 通常用来减过拟合；BN 网里可以去掉或减弱（PDF p.5）。

## MNIST：分布漂没漂，图上看得见，终点读不出

为验证内部协变量偏移以及 BN 能否压住它，作者在 MNIST 上做数字分类。网络很简单：28×28 二值图，三个全连接隐层各 100 个激活，sigmoid，$W$ 用小高斯初始化，最后 10 维加交叉熵。训 50000 步，每 batch 60 个样本。BN 加在每个隐层。目标不是刷 MNIST 纪录，只比有无 BN（PDF p.5–6）。

Figure 1(a)：BN 网测试正确率更高、升得更快。Figure 1(b)(c) 取最后隐层一个典型 sigmoid 输入，画训练过程中第 15、50、85 百分位。无 BN 时均值和方差显著漂移；有 BN 时分布稳得多（PDF p.5–6）。这张图没有印出最终百分比。

## ImageNet：先加 BN，再把配方改到匹配它

BN 加到 Inception 的一个新变体上，任务是 ImageNet 1000 类分类。相对原 Inception，5×5 卷积换成两层连续 3×3，滤波器最多 128。参数 $13.6 \cdot 10^{6}$，除顶上 softmax 外没有全连接层。细节在附录（PDF p.6）。

训练用带动量的 SGD，mini-batch 32，大规模分布式架构，类似 Dean 等人 2012。训练过程中用单裁剪、验证集 top-1 跟踪。所有变体都把 BN 按卷积方式加在每个非线性之前，其余结构不动（PDF p.6）。

只加 BN 吃不透方法。配套改动是（PDF p.6）：

| 改动 | 论文给的理由 |
|---|---|
| 提高学习率 | 第 3.3 节允许，没有不良副作用 |
| 去掉 Dropout | 加快训练，过拟合没有加重 |
| L2 权重正则减弱 5 倍 | 验证精度反而更好 |
| 学习率衰减加快 6 倍 | 网训得更快，指数衰减要跟得上 |
| 去掉局部响应归一化 | 有 BN 后不必 |
| 更彻底地打乱训练样本 | shard 内 shuffle，验证精度大约再升 1% |
| 减弱光度扭曲 | 每个样本被看见的次数更少，让训练更盯更真实的图 |

对照网都在 LSVRC2012 训练集上训、验证集上测（PDF p.7）。

| 模型 | 初始学习率 | 相对 Inception | 非线性 |
|---|---:|---:|---|
| Inception | 0.0015 | 1 | ReLU |
| BN-Baseline | 同 Inception，只加 BN | 1 | ReLU |
| BN-x5 | 0.0075 | 5 | ReLU |
| BN-x30 | 0.045 | 30 | ReLU |
| BN-x5-Sigmoid | 0.0075 | 5 | sigmoid |

原 Inception 用同样的 5 倍学习率，参数会到机器无穷。原 Inception 换 sigmoid，一直停在相当于随机猜的水平（PDF p.7）。

Figure 2 是单裁剪验证精度对训练步，曲线上没有印出可抄的终点。到达 Inception 最高精度 72.2% 所需步数，以及各网自己的最高精度，在 Figure 3（PDF p.7）。

| 模型 | 到达 72.2% 的步数 | 最高精度 |
|---|---:|---:|
| Inception | $31.0 \cdot 10^{6}$ | 72.2% |
| BN-Baseline | $13.3 \cdot 10^{6}$ | 72.7% |
| BN-x5 | $2.1 \cdot 10^{6}$ | 73.0% |
| BN-x30 | $2.7 \cdot 10^{6}$ | 74.8% |
| BN-x5-Sigmoid | | 69.8% |

Sigmoid 那一格的步数原文即空，不补。只加 BN（BN-Baseline），不到一半步数就打平 Inception。再改配方，BN-x5 用 Inception 的十四分之一步到达 72.2%。我们按表核对：$31.0 / 2.1 \approx 14.8$，论文写成 14；$2.1 / 31.0 \approx 6.8\%$，引言写成 7% 的训练步。两句说的是同一对比（PDF p.2、p.7）。

BN-x30 初期略慢，最终更高：约 $6 \cdot 10^{6}$ 步到 74.8%，步数是 Inception 到 72.2% 所需的五分之一。$31.0 / 6 \approx 5.2$，论文写成 5。Figure 3 里它到达 72.2% 是 $2.7 \cdot 10^{6}$ 步，与 $6 \cdot 10^{6}$ 步到自己的峰值不是同一个数（PDF p.7）。

BN-x5-Sigmoid 达到 69.8%。没有 BN 的 sigmoid Inception 从未好过 1/1000，也就是随机猜（PDF p.7）。这是「内部协变量偏移减少后，饱和非线性也能训」的直接证据。它不意味着今天该退回 sigmoid。

## 集成：验证 4.9%，测试 4.82%

当时 ImageNet 竞赛最好结果来自 Deep Image 的集成，以及 He 等人 2015 的集成。后者经 ILSVRC 服务器测得 top-5 错误 4.94%。这份 PDF 的参考文献把那篇标成整流器论文，不是后来的残差网。论文报告验证集 top-5 错误 4.9%、测试错误 4.82%，并写它超过 Russakovsky 等人 2014 估计的人类标注精度（PDF p.7）。

集成用 6 个网，都基于 BN-x30，再各自改一部分：卷积层初始权重加大；Dropout 概率 5% 或 10%（原 Inception 是 40%）；模型最后隐层用非卷积、按激活的 BN。每个网大约 $6 \cdot 10^{6}$ 步到达自己的最高精度。集成预测是各类概率的算术平均。多裁剪与集成细节类似 Szegedy 等人 2014（PDF p.7）。

Figure 4 在 50000 张验证集上与此前结果对照。脚注：BN-Inception 集成在 100000 张测试集上由测试服务器报 4.82% top-5（PDF p.7–8）。

| 模型 | 分辨率 | 裁剪数 | 模型数 | Top-1 错误 | Top-5 错误 |
|---|---:|---:|---:|---:|---:|
| GoogLeNet ensemble | 224 | 144 | 7 | — | 6.67% |
| Deep Image low-res | 256 | — | 1 | — | 7.96% |
| Deep Image high-res | 512 | — | 1 | 24.88 | 7.42% |
| Deep Image ensemble | 可变 | — | — | — | 5.98% |
| BN-Inception 单裁剪 | 224 | 1 | 1 | 25.2% | 7.82% |
| BN-Inception 多裁剪 | 224 | 144 | 1 | 21.99% | 5.82% |
| BN-Inception 集成 | 224 | 144 | 6 | 20.1% | 4.9% |

Deep Image high-res 的 top-1 原文印 24.88，未加百分号，上表照抄。空单元格保持为空。验证集 4.9% 与测试集 4.82% 是两套集合，摘要的 4.8% 是测试错误的四舍五入（PDF p.1、p.8）。

## 和标准化层的差别，以及没做的事

力量来自两处：归一化激活，以及把归一化写进架构，让任何优化器都正确处理它。每个激活只多两个参数，表示能力保住。得到的网可以用饱和非线性，更能容忍大学习率，常常不必靠 Dropout 正则（PDF p.7–8）。

方法与 Gülçehre 与 Bengio 2013 的标准化层有相似处，但目标与做法不同。BN 要的是训练全程激活分布稳定，实验里加在非线性之前。对方加在非线性之后，激活更稀疏。作者在大规模图像分类里，无论有无 BN，都没看到非线性输入稀疏。其他差别：BN 有可学的缩放平移以表示恒等；处理卷积；推理确定、不依赖 mini-batch；对每个卷积层都做 BN（PDF p.8）。

没做的：循环网，那里内部协变量偏移和梯度消失、爆炸更严重。传统意义上的域适应，是否只需重算总体均值方差就能更容易泛化到新分布。进一步的理论分析（PDF p.8）。原件没有 CIFAR 实验，也没有机器数、墙钟时间或开源仓库。并行切分没写。

## 附录：这座 Inception 相对 GoogLeNet 改了什么

Figure 5 是架构规格，不是准确率。读表方式见 Szegedy 等人 2014。作者列出的要点（PDF p.9–10）：

- 5×5 卷积换成连续两层 3×3。最大深度增加 9 个权重层；参数约增 25%，计算代价约增 30%。
- 28×28 的 Inception 模块从 2 个增到 3 个。
- 模块内有时平均池、有时最大池。
- 模块之间没有贯穿式池化；在模块 3c、4e 的滤波器拼接前使用 stride-2 的卷积或池化。
- 第一层卷积用深度乘数 8 的可分离卷积，降低计算、增加训练时显存。

下表按原表抄。3c、4e 行印了 stride 2，输出尺寸栏仍印着 28×28×576 与 14×14×1024，不改成「理应变成的下一档尺寸」（PDF p.11）。

| 类型 | patch / stride | 输出尺寸 | depth | 1×1 | 3×3 reduce | 3×3 | 双 3×3 reduce | 双 3×3 | Pool |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| convolution | 7×7/2 | 112×112×64 | 1 | | | | | | |
| max pool | 3×3/2 | 56×56×64 | 0 | | | | | | |
| convolution | 3×3/1 | 56×56×192 | 1 | | 64 | 192 | | | |
| max pool | 3×3/2 | 28×28×192 | 0 | | | | | | |
| inception (3a) | | 28×28×256 | 3 | 64 | 64 | 64 | 64 | 96 | avg + 32 |
| inception (3b) | | 28×28×320 | 3 | 64 | 64 | 96 | 64 | 96 | avg + 64 |
| inception (3c) | stride 2 | 28×28×576 | 3 | 0 | 128 | 160 | 64 | 96 | max + pass through |
| inception (4a) | | 14×14×576 | 3 | 224 | 64 | 96 | 96 | 128 | avg + 128 |
| inception (4b) | | 14×14×576 | 3 | 192 | 96 | 128 | 96 | 128 | avg + 128 |
| inception (4c) | | 14×14×576 | 3 | 160 | 128 | 160 | 128 | 160 | avg + 128 |
| inception (4d) | | 14×14×576 | 3 | 96 | 128 | 192 | 160 | 192 | avg + 128 |
| inception (4e) | stride 2 | 14×14×1024 | 3 | 0 | 128 | 192 | 192 | 256 | max + pass through |
| inception (5a) | | 7×7×1024 | 3 | 352 | 192 | 320 | 160 | 224 | avg + 128 |
| inception (5b) | | 7×7×1024 | 3 | 352 | 192 | 320 | 192 | 224 | max + 128 |
| avg pool | 7×7/1 | 1×1×1024 | 0 | | | | | | |

## 可以带走的

1. **统计量必须进计算图**。在梯度外面减均值，会出现更新被归一化抵消、参数空涨。BN 的核心不是减均值本身，而是让 $\mu$、$\sigma^{2}$ 参与反传。
2. **标准化之后一定要留 $\gamma$、$\beta$**。否则非线性可能被钉在线性区，容量被悄悄砍掉。
3. **训练统计和推理统计要分开**。训练用 batch 是为了可微和正则；推理按算法 2 用总体均值和无偏方差，合成一个线性变换。滑动平均是跟踪训练精度的替代，不是这条默认路径。
4. **BN 改变的是可训练域，配方要跟着改**。大学率、弱 Dropout、弱 L2、更快衰减、更强 shuffle，都是同一机制的配套。
5. **卷积按特征图共享**。位置之间共用均值方差和一对 $\gamma$、$\beta$，既省参数，也保住平移性质。

## 关键词回看

- **内部协变量偏移**：训练时前面层参数一变，后面层输入分布跟着漂。
- **Batch Normalization**：对 mini-batch 按维标准化，再学缩放平移；训练可微，推理改用总体统计。
- **$\gamma$、$\beta$**：让 BN 能表示恒等。
- **卷积 BN**：整张特征图、整个 batch 合在一起估统计，每通道一对参数。
- **BN-x5 / BN-x30**：在 BN 上把学习率提到原 Inception 的 5 倍或 30 倍，并配套改正则与衰减。

## 参考资料

- 原件：`readings/_src/深度学习基石/BatchNorm.pdf`（arXiv:1502.03167v3，11 页）
- 文内对照都来自这份 PDF 的引用，不另引实验数字：Inception / GoogLeNet（Szegedy et al. 2014）、Dropout（Srivastava et al. 2014）、标准化层（Gülçehre and Bengio 2013）

## 资料与阅读边界

本文只依据这份 v3 原件。封面日期是 2015-03-02；首发日仍是 v1 提交日。
