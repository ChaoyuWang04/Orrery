# FlashAttention-3：H100 的新硬件是异步的，内核也得学会同时干几件事

<!-- release-date: 2024-07-11 -->

> 本文依据 **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**（Jay Shah、Ganesh Bikshandi、Ying Zhang、Vijay Thakkar、Pradeep Ramani、Tri Dao）的 NeurIPS 2024 正式版，共 28 页（正文 p. 1–10，参考文献 p. 11–14，附录 A–C p. 15–22，NeurIPS 自查表 p. 23–28）。对应的 arXiv 编号是 2407.08608。页码均指这份 PDF 自身的页码。文中会区分三件事：**论文明确写了什么**、**我们怎么解释它**、**哪些是外部资料或本文推算**。

## 读前先把几个词说成人话

- **Hopper / H100**：NVIDIA 的 Hopper 架构，代表产品是 H100。本文的 H100 指 SXM5 版，132 个 SM（PDF p. 4、p. 19）。
- **warpgroup**：4 个相邻的 warp，共 128 个线程。线程层级从细到粗是线程、warp（32 个线程）、warpgroup、线程块（CTA）、线程块簇（Hopper 新增）、网格（PDF p. 3）。
- **TMA（Tensor Memory Accelerator）**：Hopper 上专门在显存与共享内存之间异步搬数据的硬件单元（PDF p. 3）。
- **WGMMA**：Hopper 的 warpgroup 级矩阵乘指令。它是**异步**的，而且可以直接从共享内存取操作数（PDF p. 3）。
- **warp 特化（warp specialization）**：把线程块里的 warp 分成生产者和消费者两种角色，一种只发搬数据的指令，一种只做计算（PDF p. 3）。
- **setmaxnreg**：Hopper 允许在 warpgroup 之间动态重新分配寄存器，让做矩阵乘的 warp 多拿、只发 TMA 的 warp 少拿（PDF p. 3）。
- **MUFU（多功能单元）**：算指数这类特殊函数的专用单元；softmax 的指数由它执行（PDF p. 4–5）。
- **FP8（e4m3）**：8 位浮点，4 位指数、3 位尾数。Hopper 上 FP8 矩阵乘的吞吐是 FP16/BF16 的 2 倍（PDF p. 3、p. 7）。
- **SS-GEMM 与 RS-GEMM**：WGMMA 的第一个操作数来自共享内存（S）还是寄存器（R）（PDF p. 4）。

前两代见 FlashAttention 与 FlashAttention-2 两篇。这里只需要记住：第一代用分块和在线 softmax 让 $N\times N$ 的矩阵不落到显存；第二代把 $Q$ 放在外层、沿序列长度并行，并尽量少做非矩阵乘运算。

## 先讲矛盾：同一套算法，在 H100 上只用了三分之一

FlashAttention-2 在 A100 上能达到理论峰值的 73%（见 FlashAttention-2 一篇）。可搬到 H100，它只有约 35% 的利用率，而优化好的矩阵乘能到 80–85%（PDF p. 1）。

论文分两层解释（PDF p. 1–2）：

- **实现层**：FlashAttention-2 没有用 Hopper 专用的指令，还在用 Ampere 的那一套。ThunderKittens、cuDNN 9 等工作已经表明，换上 Hopper 指令就能提速。
- **算法层**，这才是根本：FlashAttention-2 遵循的是一个简化的同步模型，设计里没有显式利用**异步**和**低精度**。

为什么这两样重要？论文的说法是：异步来自硬件专门化，Tensor Core 专做矩阵乘，TMA 专做搬数，它们与做逻辑、整数和浮点运算的其余 CUDA 核心是分开的；低精度（Pascal 的 FP16、Ampere 的 BF16、Hopper 的 FP8、Blackwell 的 FP4）是在同样功耗和面积下把吞吐翻倍、翻四倍的成熟手段（PDF p. 2）。

还有一笔账把矛盾说得更尖锐。H100 SXM5 的 FP16 矩阵乘峰值是 989 TFLOPs/s，而指数这类特殊函数只有 3.9 TFLOPs/s：每个 SM 每时钟 16 次，乘以 132 个 SM 和 1830 MHz（PDF p. 4，脚注 5）。头维 128 的 FP16 前向里，矩阵乘的 FLOP 是指数运算的 512 倍，可指数的吞吐低 256 倍，于是**指数运算要占掉矩阵乘时间的约 50%**。FP8 更糟：矩阵乘吞吐翻倍，指数吞吐不变（PDF p. 4）。

矛盾于是是：

> 硬件把矩阵乘、搬数、指数交给了三种能同时工作的单元。内核如果还是一步等一步地走，Tensor Core 就要花大量时间等数据、等 softmax。怎样重排算法，让它们同时忙起来？再加上 FP8，布局和精度两道关怎么过？

## 一句话先说清

FlashAttention-3 是 FlashAttention-2 在 Hopper 上的重新设计，提出并综合了三件事（PDF p. 2）：

1. **生产者 / 消费者异步**：用 warp 特化和共享内存环形缓冲，把搬数和计算分给不同的 warp，藏住访存和发指令的延迟。
2. **把 softmax 藏在异步矩阵乘底下**：warpgroup 之间打乒乓，warpgroup 内部再做两级流水，让 softmax 与 WGMMA 重叠。
3. **FP8 矩阵乘**：解决 FP8 WGMMA 的布局约束，并用分块量化和非相干处理压住量化误差。

结果：BF16 前向比 FlashAttention-2 快 1.5–2.0 倍，最高 840 TFLOPs/s，约为峰值的 85%；反向快 1.5–1.75 倍；FP8 前向最高约 1.3 PFLOPs/s；FP8 的数值误差比按张量量化的基线低 2.6 倍（PDF p. 1–2）。

### 一条阅读路线

1. **p. 3 Table 1 与第 2.2 节**：Hopper 的存储层级、线程层级和异步单元。
2. **p. 4–5 第 3.1 节、Algorithm 1 与 Figure 1**：warp 特化与乒乓调度。
3. **p. 6–7 第 3.2 节、Figure 2 与 Algorithm 2**：warpgroup 内部的两级流水。
4. **p. 7–8 第 3.3 节**：FP8 的布局与精度。
5. **p. 8–10 第 4 节**：Figure 5–7、Table 2 消融、Table 3 数值误差。
6. 附录 **B（p. 16–21）**：反向、SASS 分析、三级流水、变长序列、持久化内核、FP8 的寄存器交换与 V 转置、推理。

## 背景：Hopper 多了什么

Table 1（PDF p. 3）把 H100 SXM5 的存储层级列成了这样：

| 硬件层级 | 并行单位 | 存储 | 容量与带宽 |
|---|---|---|---|
| 整块芯片 | 网格 | 显存（GMEM，即 HBM） | 80 GiB，3.35 TB/s |
| GPC | 线程块簇 | L2 缓存 | 50 MiB，12 TB/s |
| SM | 线程块（CTA） | 共享内存（SMEM） | 每 SM 228 KiB，全卡 31 TB/s |
| 线程 | 线程 | 寄存器（RMEM） | 每 SM 256 KiB |

共享内存的 31 TB/s 是论文按第三方测得的每 SM 每时钟 128 字节，乘以 132 个 SM 和 1830 MHz 推出来的（PDF p. 3，脚注 4）。每个线程最多 256 个寄存器（PDF p. 3）。

与 A100 相比，对这篇论文要紧的是三样异步能力（PDF p. 3）：

1. **TMA**：一个线程就能发起一整块数据的异步搬运；
2. **异步 WGMMA**：发出矩阵乘之后，warp 可以去做别的，之后再等它完成；
3. **setmaxnreg**：按角色重新分配寄存器。

另外，FP8 WGMMA 只接受 k-major 的共享内存操作数，即沿收缩维 $K$ 连续；FP16 两种都接受。背靠背的两个矩阵乘里，FP32 累加器的布局和 FP8 操作数的布局又互相冲突（PDF p. 4）。这两条后面会变成 FP8 的两道关。

## 设计一：warp 特化，搬的只管搬，算的只管算

![一个线程块里，生产者 warpgroup 让出寄存器，只用 TMA 把 Qᵢ 和逐块的 Kⱼ、Vⱼ 读进 s 格环形缓冲；消费者 warpgroup 多拿寄存器，只做 QKᵀ、softmax、PV 并释放缓冲格；反向多一个 dQ 写手 warp，专门把本地 dQ 原子加到显存。](/reports/FlashAttention-3/figure-warp-specialization.svg)

### 旧问题：一个 warp 既要搬又要算

FlashAttention-2 的每个 warp 既发加载指令也做计算，两者串行排在同一条指令流里。Hopper 上搬数和矩阵乘都是异步单元干的，同步写法让它们互相等（PDF p. 2）。

### 新设计

前向在 batch、头数和查询序列长度上仍是完全并行的，所以只需看一个线程块怎么处理一块 $Q_i$（PDF p. 4）。线程块里分出两种 warpgroup（PDF p. 4–5，Algorithm 1）：

- **生产者**：先让出一部分寄存器，然后发起 TMA 读 $Q_i$；再按 $j$ 逐块等环形缓冲的第 $(j\bmod s)$ 格被消费完，往里读 $K_j$、$V_j$，读完通知消费者。
- **消费者**：多拿寄存器，等 $Q_i$ 和 $K_j$ 到位，用 SS-GEMM 算 $S_i^{(j)}=Q_iK_j^\top$；做在线 softmax 并缩放 $O_i$；等 $V_j$ 到位，用 RS-GEMM 算 $O_i\mathrel{+}=\tilde P_i^{(j)}V_j$；释放这一格。循环结束后除以 $\ell_i$、算 logsumexp，写回显存。

发 TMA 不必等其他加载完成；缓冲前 $s$ 轮装填时，生产者也不用等（PDF p. 4）。

**我们怎么解释它。** 这是把「预取」从程序员手排的指令顺序，变成了两种角色之间的队列。生产者尽可能跑在前面，消费者只要队列里有数据就不停。寄存器按角色分，是因为发 TMA 只需要一个线程，而矩阵乘和 softmax 很吃寄存器。

### 反向：多一个 dQ 写手

反向同样做 warp 特化，但多了一个角色：**dQ 写手**（PDF p. 16，附录 B.1）。每个线程块算出的是 $dQ$ 的局部贡献，要累加到全局的 $dQ$ 上，许多线程块会写同一个位置，产生竞争。把这件事交给单独的 warp，用信号量做原子加，其余 warp 就不必被它挡住，可以接着做下一次矩阵乘（PDF p. 16）。

反向算法的其余部分与 FlashAttention-2 一致：每个线程块固定一块 $K_j$、$V_j$，内层扫 $Q_i$；重算 $S$、$P$，算 $dP$、$dS$，累加 $dV_j$、$dK_j$（PDF p. 16，Algorithm 3）。

### 可迁移启发

当硬件有独立的搬数单元时，把「搬」和「算」拆给不同的执行者，中间用有界队列连接，比在同一条指令流里手排重叠更稳。资源（这里是寄存器）也跟着角色分配，而不是平均分。

## 设计二：warpgroup 之间打乒乓

![理想化时间线：warpgroup 1 与 warpgroup 2 轮流发 GEMM1、GEMM0，一方做 softmax 时另一方占着 Tensor Core；最下一行显示 Tensor Core 每一格都被某个 warpgroup 占用。](/reports/FlashAttention-3/figure1-pingpong.svg)

### 旧问题：softmax 占了矩阵乘一半的时间

上面那笔账：头维 128 的 FP16 前向里，指数要占矩阵乘时间的约 50%（PDF p. 4）。指数由 MUFU 执行，是和 Tensor Core 分开的单元。理想情况是矩阵乘在跑的时候，MUFU 同时在算另一批指数（PDF p. 5）。

### 新设计：用屏障排定两个 warpgroup 的先后

用同步屏障（`bar.sync`）强制 warpgroup 1 的两个矩阵乘（本轮的 GEMM1 即 $PV$，下一轮的 GEMM0 即 $QK^\top$）排在 warpgroup 2 的矩阵乘之前。于是 warpgroup 2 占着 Tensor Core 时，warpgroup 1 在做 softmax；然后角色互换（PDF p. 5）。

论文承认实际调度没有图里那么干净，但这通常能提升性能：头维 128、序列 8192 的 FP16 前向，从 570 提到 620–640 TFLOPs/s（PDF p. 5）。

MQA 与 GQA 沿用 FlashAttention-2 的做法，通过下标映射避免在显存里复制 $K$、$V$（PDF p. 5）。

### 可迁移启发

两种单元耗时相当、又彼此依赖时，单条流水只能串行。给它两条互相独立的流（这里是两个 warpgroup 各算各的查询行），再用屏障排定先后，让两条流错开半拍。

## 设计三：warpgroup 内部的两级流水

![上半是理想化时间线：第 j 轮的 softmax 进行时，第 j−1 轮的 PV 已经在 Tensor Core 上跑；下半是 Algorithm 2 主循环的六步，先连发 QKᵀ 与上一轮的 PV，再用新的 S 做 softmax，最后等上一轮 PV 完成并缩放 O。](/reports/FlashAttention-3/figure2-two-stage.svg)

### 旧问题：一轮之内的顺序依赖

在一轮迭代里，softmax 依赖第一个矩阵乘的输出 $S$，第二个矩阵乘又依赖 softmax 的输出 $\tilde P$。Algorithm 1 第 17、21 行的等待，把 softmax 和两个矩阵乘串成了一条线（PDF p. 6）。

### 新设计：跨轮流水

在寄存器里多放一份缓冲，打破轮内依赖：第 $j$ 轮里，先发出本轮的 $S_{\text{next}}=Q_iK_j^\top$，再发出**上一轮**的 $O_i\mathrel{+}=\tilde P_{\text{cur}}V_{j-1}$，两个都不等；然后等 $S_{\text{next}}$ 完成、做本轮的 softmax；再等上一轮的 $PV$ 完成、缩放 $O_i$（PDF p. 6，Algorithm 2 第 8–16 行）。论文的原话是：第 $j$ 轮的第二个 WGMMA 与第 $j+1$ 轮的 softmax 重叠（PDF p. 6）。

Algorithm 2 替换 Algorithm 1 里的消费者部分，两者合起来就是 FP16 下完整的 FlashAttention-3 前向（PDF p. 6）。

### 三个实际问题

论文专门列了三点（PDF p. 7；p. 17–18）：

1. **编译器会重排指令。** 伪代码是理想顺序，NVCC 可能打乱。作者看了生成的 SASS：softmax 被排到最前面，第一个 WGMMA 与 softmax、FP32 到 FP16 的类型转换交错执行，说明矩阵乘与非矩阵乘确实并行了；第二个 WGMMA 没有和其他指令交错，这符合预期（PDF p. 17，附录 B.2）。
2. **寄存器压力。** 多存一份 $S_{\text{next}}$，每个线程块多用 $B_r\times B_c\times 4$ 字节寄存器。这与「用更大的块」这种同样吃寄存器的优化冲突，要按实测取舍（PDF p. 7）。
3. **三级流水反而更差。** 作者试了再多一级，让第二个 WGMMA 也和 softmax 重叠。结果编译器只让第一个 WGMMA 与 softmax 重叠，第二个仍然没有，原因作者也说不清；再加上寄存器更紧、只能用更小的块，所以性能不如两级（PDF p. 17–18，附录 B.3）。

### 消融：两处改动各有贡献

Table 2（PDF p. 9）固定 batch 4、序列 8448、16 头、头维 128，不带因果掩码的 FP16 前向：

| 配置 | 时间 | TFLOPs/s |
|---|---:|---:|
| FlashAttention-3 | 3.538 ms | 661 |
| 去掉 GEMM-softmax 流水，保留 warp 特化 | 4.021 ms | 582 |
| 保留 GEMM-softmax 流水，去掉 warp 特化 | 4.105 ms | 570 |

**我们怎么解释它。** 两项各自去掉都会掉约 12–14%（本文推算）。这张表只有一个形状，也没有「两项都去掉」的一行，所以只能说两者各有用，不能说它们的贡献可以相加。

### 可迁移启发

理想的流水图不是执行计划：编译器会重排，寄存器会不够。每多一级流水就多一份缓冲，流水深度由状态预算决定。验证重叠是否真的发生，要看生成的机器码。

## 设计四：FP8，布局一关、精度一关

![左：第一个 WGMMA 的 FP32 累加器里，第 0 行每 2 列一格，T0 到 T3 轮流拿；第二个 WGMMA 要的 FP8 操作数里，每个线程拿连续 4 列，所以要用 byte_perm 和 shfl_sync 交换；右：FP8 WGMMA 只接受沿收缩维连续的操作数，PV 要求 V 沿序列连续，所以生产者在共享内存里用 LDSM、STSM 转置 V。](/reports/FlashAttention-3/figure-fp8-layout.svg)

### 布局关

两处冲突（PDF p. 7；附录 B.7–B.8，p. 19–21）：

1. **寄存器归属不一致。** 第一个 WGMMA 的 FP32 累加器降成 FP8 之后，要当第二个 WGMMA 的操作数，可两种布局里同一列数据属于不同线程。做法是用 `byte_perm` 在线程内重排字节、`shfl_sync` 在每 4 个线程之间交换，再按线程号 `byte_perm` 一次。
2. **V 的连续方向不对。** FP8 WGMMA 要求 k-major，于是 $Q$、$K$ 要沿头维连续，$V$ 要沿序列连续；而 $V$ 通常沿头维存放，TMA 又不能改变连续方向。做法是在核内转置：生产者 warpgroup 在 TMA 读完一块 $V$ 后，用 LDSM、STSM 指令在共享内存里转置到单独的缓冲。这两条指令很省寄存器，生产者让出寄存器后仍能执行；FP8 本身省下的共享内存正好放得下这份缓冲（PDF p. 20）。

### 精度关：分块量化与非相干处理

FP8（e4m3）只有 3 位尾数，误差本来就比 FP16/BF16 大；大模型里又常有比多数值大得多的离群值，让量化更难。常见做法是按张量缩放，每个张量一个缩放因子（PDF p. 7）。FlashAttention-3 用了两招（PDF p. 7–8）：

1. **分块量化（block quantization）**：$Q$、$K$、$V$ 按 $B_r\times d$ 或 $B_c\times d$ 的块各存一个缩放因子。量化可以融合进注意力之前的操作（比如旋转位置编码），因为那本来就受访存限制，不增加耗时；FlashAttention 本来就按块算，给 $S$ 的每块乘上对应的缩放因子也不花额外计算。
2. **非相干处理（incoherent processing）**：量化前给 $Q$、$K$ 都乘上同一个随机正交矩阵 $M$。因为 $MM^\top=I$，

$$
(QM)(KM)^\top = QMM^\top K^\top = QK^\top
$$

注意力输出不变。而 $QM$ 的每个元素是 $Q$ 一行元素的随机加权和，离群值被「摊开」，量化误差就小了。$M$ 取 ±1 随机对角矩阵与 Hadamard 矩阵的乘积，乘法只需 $O(d\log d)$，也能融合进旋转位置编码（PDF p. 8）。这个技巧借自 QuIP 与 QuIP# 的量化工作（PDF p. 8、p. 15）。

### 数值误差：Table 3

测试数据模拟离群值：每个元素取 $\mathcal N(0,1)$，再以 0.1% 的概率加上一个标准差 10 的正态项。与 FP64 参考实现比较均方根误差（PDF p. 9–10）：

| 方法 | RMSE |
|---|---:|
| 标准实现 FP16 | $3.2\times10^{-4}$ |
| FlashAttention-2 FP16 | $1.9\times10^{-4}$ |
| FlashAttention-3 FP16 | $1.9\times10^{-4}$ |
| 标准实现 FP8（按张量缩放） | $2.4\times10^{-2}$ |
| FlashAttention-3 FP8 | $9.1\times10^{-3}$ |
| FlashAttention-3 FP8，去掉分块量化 | $9.3\times10^{-3}$ |
| FlashAttention-3 FP8，去掉非相干处理 | $2.4\times10^{-2}$ |

FP16 下两代 FlashAttention 都比标准实现误差低 1.7 倍，因为 softmax 的中间结果保持在 FP32；FP8 的标准基线是矩阵乘累加用 FP32、softmax 中间结果用 FP16（PDF p. 9–10）。

**我们怎么解释它。** 2.6 倍（$2.4\times10^{-2}$ 对 $9.1\times10^{-3}$）几乎全部来自非相干处理：去掉它，误差回到基线水平；去掉分块量化，误差只从 $9.1$ 升到 $9.3\times10^{-3}$。在这组人造的离群值分布上，「把离群值摊开」比「缩小量化粒度」重要得多。这是一组合成数据，真实模型上的效果论文没有测。

### 可迁移启发

1. **低精度先拆成两个问题**：能不能跑（布局、指令约束），和准不准（误差）。
2. **先找不改结果的恒等变换**：正交旋转不改 $QK^\top$，却能改变数值分布；这类变换还可以搭访存受限算子的便车，几乎免费。

## 工程细节：附录 B 里的几件事

- **变长序列**（PDF p. 18–19）：TMA 直接处理变长要改 tensormap，开销大。前向读 $Q$ 时让 TMA 总是读满一块，越界的行填零，再由掩码处理；写 $O$ 用普通的合并写。反向用预处理内核给每条序列的 $dQ$ 等张量补 128 个元素，以便用 TMA 写。
- **线程块簇与 TMA 多播**（PDF p. 19）：固定长度时用大小为 2 的簇，两个处理同一条序列的线程块合读 KV。变长、因果、滑窗时有的线程块会提前退出，无法合作；不用簇会让变长情形慢约 2%。
- **掩码只在需要时做**（PDF p. 19）：因果和变长只对最后一个 K 块施加掩码，局部注意力只对首尾几块施加。
- **持久化内核**（PDF p. 19）：启动与 SM 数相同的线程块（H100 SXM5 上 132 个），由调度器分配工作，让上一块的收尾与下一块的开头重叠，Tensor Core 少闲。
- **推理**（PDF p. 21，附录 B.9）：解码时查询只有一两个 Token，注意力变成受访存限制，要比的是读 KV cache 的带宽而不是 Tensor Core 利用率；按查询长度并行又不够。做法有两个：沿 KV 长度切分（即 Flash-Decoding），再用单独的内核合并；以及 GQA 打包，把同一个 KV 头对应的多个查询头装进一块 $Q$，论文说最多比不打包快 $N$ 倍，$N$ 是 GQA 比例。还支持 PagedAttention，由 Kai Londenberg 贡献。这一节没有给实测数字。

## 实验：先看清测了什么

### 设置

H100 80GB SXM5（700W），时钟固定在 1830 MHz，每项重复 10 次取平均；库版本取写作时（2024 年 10 月）的最新版：CUDA 12.3、cuDNN 9.5.0.50、CUTLASS 3.6、FlashAttention 2.6.3、Triton 3.1、PyTorch 2.5.0（PDF p. 21–22）。

序列长 512 到 16K，batch 调到总 Token 数 16K；隐藏维 2048，头维 64、128 或 256（即 32、16 或 8 个头）。FLOPs 公式与 FlashAttention-2 相同：前向 $4\cdot\text{seqlen}^2\cdot\text{头维}\cdot\text{头数}$，因果掩码除以 2，反向乘 2.5（PDF p. 8）。对照有标准实现、FlashAttention-2、Triton 版（用了 H100 专用指令）和 cuDNN（PDF p. 8）。

### BF16 前向（Figure 5，PDF p. 9，TFLOPs/s）

| 设置 | 序列长 | FlashAttention-2 | Triton | cuDNN | FlashAttention-3 |
|---|---:|---:|---:|---:|---:|
| 头维 64，不带因果 | 512 | 226 | 299 | 387 | 351 |
| | 16K | 312 | 423 | 533 | 566 |
| 头维 128，不带因果 | 512 | 232 | 314 | 514 | 482 |
| | 16K | 364 | 451 | 681 | 760 |
| 头维 128，因果 | 512 | 145 | 202 | 334 | 304 |
| | 16K | 328 | 415 | 629 | 697 |
| 头维 256，不带因果 | 512 | 228 | 223 | 501 | 512 |
| | 16K | 337 | 305 | 758 | 842 |

### BF16 反向（Figure 6，PDF p. 10，TFLOPs/s，不带因果）

| 头维 | 序列长 | FlashAttention-2 | cuDNN | FlashAttention-3 |
|---|---:|---:|---:|---:|
| 64 | 512 | 198 | 275 | 288 |
| | 16K | 291 | 475 | 552 |
| 128 | 512 | 214 | 304 | 317 |
| | 16K | 322 | 576 | 615 |

读这两张表要知道四件事：

1. **峰值与利用率**：842 TFLOPs/s（头维 256、16K）除以 989，是 85%，对上摘要。
2. **相对 FlashAttention-2 的倍数超出了论文的区间。** 按柱顶数字算，前向在头维 64 时是 1.4–1.8 倍，头维 128 时稳定在约 2.1 倍，头维 256 时 2.0–2.5 倍；反向是 1.4–2.0 倍（本文推算）。论文写的是前向 1.5–2.0 倍、反向 1.5–1.75 倍（PDF p. 8）。附录说基准用的是 2024 年 10 月的库版本，图很可能在定稿时重测过，而文字区间没有跟着更新；这是我们的推测。
3. **对 cuDNN**：论文说 1K 及以上 FlashAttention-3 超过 cuDNN（PDF p. 8）。头维 128、256 时大体如此；头维 64 时 1K 仍落后（不带因果 422 对 452），带因果 4K 也落后（444 对 450）。512 时几乎所有配置都输给 cuDNN。反向在所有长度上都领先 cuDNN。
4. **短序列最弱。** 512 时每个线程块的主循环只有几轮，流水线的装填与收尾占比大；持久化内核针对的正是这一点（PDF p. 19）。这是我们的解释。

### FP8 前向（Figure 7 与附录 Figure 10，PDF p. 10、p. 22）

头维 256、不带因果时，FlashAttention-3 从 512 的 761 升到 16K 的 1322 TFLOPs/s，cuDNN 是 686–1139；带因果时 438–1132，cuDNN 是 304–1099，4K 时 cuDNN 反超（1015 对 972）。头维 128、不带因果时两者互有胜负（FlashAttention-3 为 525–999，cuDNN 为 617–1001）。论文的概括是「FP8 与 cuDNN 持平」（PDF p. 2）。

**本文推算**：FP8 的理论峰值按「FP16 的 2 倍」算是约 1979 TFLOPs/s，1322 约为其 67%，比 BF16 的 85% 低一截。论文没有报告 FP8 的利用率。

## 论文承认的限制

第 5 节写了两条未来工作：针对大模型推理做优化，以及理解低精度注意力在大规模训练中的影响（PDF p. 10）。作者也说，虽然聚焦 Hopper，但预期这些技术适用于其他有足够异步与低精度能力的加速器（PDF p. 2、p. 10）。

我们读完还要补几条：

- **FP8 只测了前向。** 没有 FP8 反向，也没有用 FP8 注意力训练模型的结果。
- **数值误差只在合成数据上测。** 离群值分布是人造的，没有真实模型的激活。
- **消融只有一个形状**，也没有两项都去掉的对照。
- **推理只描述了做法**，没有解码场景的实测。
- **因果掩码的反向**在 Figure 6 里没有出现。

## 论文之后

以下是外部补充，不是 PDF 内容。

- 下一代 Blackwell（B200）的 Tensor Core 吞吐又翻了一倍，而指数单元与共享内存带宽没变。FlashAttention-3 用的 Hopper MMA 指令在 Blackwell 上不向前兼容，无法运行；FlashAttention-4 为此重写了内核，沿用两个 Q 块打乒乓的思路，并把一部分指数挪到 FMA 单元上用多项式算，见 FlashAttention-4 一篇。
- 本文用到的几个 Hopper 数字，与 FlashAttention-4 论文对 Hopper 的描述一致：每 SM 每时钟 16 次指数、128 字节共享内存读带宽、每线程最多 256 个寄存器。

## 可迁移启发

1. **换硬件之后，先量利用率，不是量加速比。** FlashAttention-2 在 A100 上 73%，在 H100 上只有 35%，说明算法假设与新硬件对不上。
2. **把内循环碰到的每种硬件单元列出来，看它们能否同时忙。** 矩阵乘、搬数、指数分属三种单元，串行执行就等于浪费两种。
3. **角色分离比平均分工更根本。** 生产者、消费者、dQ 写手各管一件事，资源按角色分配。
4. **流水线深度由状态预算决定。** 两级比三级快，原因是寄存器和编译器，而不是算法。
5. **理想图不是执行计划。** 用机器码验证重叠是否发生。
6. **低精度先分成「能不能跑」和「准不准」。**
7. **先找不改结果的恒等变换。** 正交旋转不改 $QK^\top$，却能压住离群值。
8. **基准的每个人为选择都要写下来。** 库版本、时钟、FLOPs 公式、测了哪些配置，决定了一个加速比能被怎样引用。

## 关键词回看

- **异步**：Tensor Core、TMA、MUFU 各自独立工作，内核要让它们同时忙。
- **warp 特化**：生产者只发 TMA，消费者只做 WGMMA 与 softmax，中间是 $s$ 格环形缓冲。
- **setmaxnreg**：按角色重新分配寄存器。
- **乒乓调度**：两个 warpgroup 轮流占 Tensor Core，一方做 softmax 时另一方做矩阵乘。
- **两级流水**：本轮的 softmax 与上一轮的 $PV$ 重叠，代价是多存一份 $S$。
- **dQ 写手**：反向里专门做 $dQ$ 原子加的 warp。
- **k-major**：FP8 WGMMA 要求操作数沿收缩维连续，因此 $V$ 要在核内转置。
- **分块量化**：每块一个缩放因子。
- **非相干处理**：$Q$、$K$ 乘同一个随机正交矩阵，摊开离群值。
- **840 TFLOPs/s、85%、1.3 PFLOPs/s、2.6 倍**：BF16 峰值、BF16 利用率、FP8 峰值、FP8 误差降幅。

## 资料与阅读边界

### 版本与首发日

- 依据版本：NeurIPS 2024 正式版（camera-ready），首页页脚印有「38th Conference on Neural Information Processing Systems (NeurIPS 2024)」，与 [NeurIPS 2024 会议论文集](https://proceedings.neurips.cc/paper_files/paper/2024/file/7ede97c3e082c6df10a8d6103a2eebd2-Paper-Conference.pdf) 上的文件逐字节相同。arXiv 上的最新版是 [2407.08608v2](https://arxiv.org/abs/2407.08608)（2024-07-12），早于会议定稿，没有更新的版本。会议版附录写明基准用的是 2024 年 10 月的库版本，数字以会议版为准。
- `release-date` 取 **2024-07-11**：官方仓库 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 的提交「FA3 initial code release」与 arXiv v1 都在 2024-07-11。FlashAttention-3 是一项技术，按技术首次官方公开日取。
- 署名：Colfax Research、Meta、NVIDIA、佐治亚理工、普林斯顿大学、Together AI（PDF p. 1），没有斯坦福署名。按「同一系列放同一目录」与前两代同放 `Stanford/`。

### 论文覆盖了、本文覆盖了的部分

摘要与第 1 节；第 2 节注意力公式、Hopper 存储与线程层级、异步与 warp 特化、低精度与布局约束；第 3.1 节 warp 特化、Algorithm 1、乒乓调度与 Figure 1；第 3.2 节两级流水、Figure 2、Algorithm 2 与三个实际问题；第 3.3 节 FP8 布局、分块量化、非相干处理与 Figure 3–4；第 4 节设置、Figure 5–7、Table 2–3；第 5 节限制；附录 A 相关工作的定位；附录 B.1–B.9；附录 C 系统配置与 FP8 全量结果。自查表（p. 23–28）是会议流程材料，不引用。

### 外部资料补充（不是 PDF 原文）

- Blackwell 与 FlashAttention-4 的相关说法取自站内 FlashAttention-4 一篇所依据的论文，见「论文之后」。
- 站内同系列（纯文本篇名）：FlashAttention、FlashAttention-2、FlashAttention-4。

### 原文没有公开、本文也不补的缺口

- FP8 反向与 FP8 训练的结果；
- 真实模型激活上的数值误差；
- 更多形状的消融，以及两项都去掉的对照；
- 推理优化（Flash-Decoding、GQA 打包、PagedAttention）的实测数字；
- 因果掩码下的反向吞吐；
- 头维 64、128 下 FP8 相对 BF16 的逐项比较（附录只给了 FP8 的图）。
