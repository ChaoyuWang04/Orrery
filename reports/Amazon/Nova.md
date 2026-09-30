# Amazon Nova：一份不写架构的报告，拿什么让人相信它的分数

<!-- release-date: 2024-12-03 -->

> 本文依据本地 `papers/Amazon/Nova.pdf`，即 Amazon Artificial General Intelligence 的 **The Amazon Nova Family of Models: Technical Report and Model Card**，arXiv:2506.12103v1（2025-03-17 提交，唯一版本），共 48 页。页码均指 PDF 自身的页码。文中区分三件事：**报告明确写了什么**、**我们怎么解释它**、**哪些是外部资料或本文推算**。

## 阅读前先认识几个词

- **智能档位（intelligence tier）**：报告用来代替参数量的说法。它从不说模型多大，只说某个模型「在它所属的档位里」最强或最快（PDF p. 3）。档位怎么划，全文没有定义。
- **tok/sec 与三项运行时指标**：TTFT（Time to First Token，首 Token 延迟）是从发出请求到收到第一个 Token 的秒数；OTPS（Output Tokens per Second）是之后每秒吐出多少 Token；Total Response Time 是从提交到生成结束的总秒数（PDF p. 13）。报告各精度表里的 tok/sec 一列就是 OTPS。
- **置信区间（CI）**：因为评测样本有限，一个分数会带一个「上下可能漂多少」的宽度。报告给了算它的公式，下文会用这条公式反推样本量。
- **潜在扩散模型（latent diffusion）**：先用 VAE（变分自编码器）把图像或视频帧压成一小张潜变量，扩散模型只在潜变量上从噪声一步步去噪，最后由 VAE 解码回像素。Nova Canvas 与 Reel 用的是这一类（PDF p. 4）。
- **红队（red teaming）与 false refusal**：主动构造对抗提示，看模型会不会越界；false refusal 指正常请求被误判为有害而拒答。
- **Goodput 与 MTTR**：goodput 是训练时间里真正推进训练的份额；MTTR（Mean Time to Restart）是训练中断后恢复到稳态的平均耗时（PDF p. 21）。

## 一句话先说清

这份报告不回答「Nova 是怎么训出来的」，它回答的是「**凭什么按档位选它、它快不快、亚马逊替你查过哪些风险**」。

模型本体它几乎什么都没给：Pro、Lite、Micro 是 Transformer，预训练用多语言、多模态数据，之后经过 SFT、奖励模型、DPO 与 PPO，合计一段话（PDF p. 3）。参数量、层数、Token 总量、超参、并行方式一项都没有。

篇幅花在三处：

- **评测口径**：置信区间公式、每个基准的提示词全文、哪些对照分是亚马逊自己在对手 API 上重跑的，都写出来了（PDF p. 5–14、30–38）；
- **Responsible AI**：八个维度、双审核模型、307 种攻击技术的分类、四家外部红队各自测了什么（PDF p. 17–21）；
- **训练基础设施**：一页半，却是全文唯一有工程数字的地方——检查点开销、恢复时间、自研的激活检查点方案（PDF p. 21–22）。

如果只记一句：

> **架构保密时，能被外人核对的只剩口径。这份报告的价值不在分数本身，而在它把口径写到了可以反算的程度。**

## 先看全景：48 页花在哪里

![Nova 报告 48 页里，讲模型本体的只有 p. 3 一段话；评测与评测提示词占了二十多页，Responsible AI 约五页，训练基础设施约一页半。](/reports/Nova/figure-page-map.svg)

这张分布就是这份报告的性格。标题同时写着 Technical Report 和 Model Card，实际比例更像一份扩写的模型卡：绿色格只有 p. 3–4 和附录 A，其中 p. 4 与附录 A 讲的还是 Canvas、Reel 的功能清单。

家族里的五个模型，报告能给出的规格就这么多（模态读自 Figure 1，PDF p. 1；上下文 PDF p. 10；速度 PDF p. 6；生成规格 PDF p. 3–4）：

| 模型 | 输入 | 输出 | 上下文 | 生成速度（tok/sec） | 报告给的机制 |
|---|---|---|---|---:|---|
| Nova Pro | 文本、图像、视频、代码、文档 | 文本、代码 | 300K | 100 | Transformer，一段话 |
| Nova Lite | 同上 | 文本、代码 | 300K | 157 | 同上 |
| Nova Micro | 文本、代码 | 文本、代码 | 128K | 210 | 同上 |
| Nova Canvas | 文本、可选参考图 | 图像 | — | — | 潜在扩散 + VAE + 文本编码器 |
| Nova Reel | 文本、可选参考图 | 视频（6 秒，720p，24 fps） | — | — | 同上 |

几条补充：

- Canvas 的正文写水平分辨率 512 到 2K、宽高比 1:4 到 4:1、最多 4.2M 像素（PDF p. 3）；附录 A 写的是 512×512 到 2K×2K（PDF p. 28），两处说法口径不同。Reel 支持 20 多种文字驱动的镜头运动（PDF p. 4）。
- Canvas 与 Reel 都经过预训练加微调两阶段，数据过滤、去重与增强管线跑在 AWS EMR 与 AWS Batch 上（PDF p. 4）。这是全文仅有的数据工程描述。
- Pro、Lite、Micro 可以在 Bedrock 上做多模态或文本微调，也支持把大模型蒸馏到小模型（PDF p. 3）。报告只说能做，没给任何微调或蒸馏的实验。
- Figure 1 在 Pro、Lite、Micro 三个框上各带一个「A/文」图标（PDF p. 1），这是常见的翻译图标，结合 p. 3 的多语言描述，本文理解为「多语言」标记；报告没有解释它。

**报告里没有的**：Nova Premier。摘要只列五个模型（PDF p. 1），全文没有出现 Premier。外部补充：同日官方新闻稿把 Premier 列为家族成员，写明 Micro、Lite、Pro 当天可用、Premier 在 2025 年第一季度提供；价格也不在报告里，新闻稿给的是「比各自档位里表现最好的模型至少便宜 75%」这类相对说法。链接见文末。

## 第一层矛盾：闭源商品的自报分数，凭什么可信

### 旧问题

一家公司说自己的模型 MMLU 85.9，外人没法复现，只能整体信或整体不信。对一个不开源、不公开架构的商品，这个问题更尖锐。

### Nova 的做法：把口径一条条写出来

报告做了五件具体的事。

**1. 给误差棒，也给算误差棒的公式。** 当分数是二元得分的平均时，报告假设样本服从高斯分布，把 95% 置信区间近似为（PDF p. 5，式 1）：

$$
CI(S) = 1.96 \times \sqrt{\frac{S \times (1 - S)}{N}}
$$

$S$ 是测得的分数，$N$ 是样本量，1.96 是 95% 置信度对应的标准正态分位数。

**2. 标出哪些对照分是自己测的。** 各表用上标 M 标「measured by us」。脚注 2 说明通道：Claude 与 Meta 模型走 Bedrock API，OpenAI 与 Gemini 走各自的 API，都在 Nova 训练完成之后测（PDF p. 5）。

**3. 说明引用的是「最高公开分」。** 能找到时，对照模型取官方报告与官网里最高的公开数字（PDF p. 7）。

**4. 每个基准的提示词全文进附录。** 附录 B 占 p. 30–38，MMLU、ARC-C、DROP（连 6 个示例一起）、IFEval、BBH、GPQA、MATH、Flores、SQuALITY、各多模态基准、FinQA 与 CRAG 的判分提示词都在（PDF p. 30–38）。Flores 甚至写明 Nova、Llama 用一套提示，Gemini、GPT 用另一套（PDF p. 34），对应 Table 2 里带星号的格子。

**5. 运行时数据交给第三方。** Figure 3 的三项指标取自 Artificial Analysis，1000 输入 / 100 输出，取数日 2024-11-29（PDF p. 13–14）。

### 我们如何验算：用 ± 号反推样本量

式 1 可以倒过来用：已知分数和 ±，就能解出 $N$。报告在正文里给了部分基准的样本量，拿来对一遍（**以下算术为本文所做**）：

| 基准 | 报告写的 $N$ | Nova Pro 分数 | 式 1 算出的 CI | 表里印的 CI |
|---|---:|---:|---:|---:|
| GSM8K | 1,319（PDF p. 5） | 94.8 | 1.20 | ±1.2 |
| ChartQA | 2,500（PDF p. 7） | 89.2 | 1.22 | ±1.2 |
| LVBench | 1,549（PDF p. 10） | 41.6 | 2.45 | ±2.5 |
| HumanEval | 164（PDF p. 12） | 89.0 | 4.79 | ±4.8 |
| FinQA | 8,281（PDF p. 12） | 77.2 | 0.90 | ±0.9 |
| CRAG | 2,706（PDF p. 12） | 50.3 | 1.88 | ±1.9 |

六项全在四舍五入内对上。**式 1 确实就是表里 ± 号的来源**，于是它可以当工具用，帮读者看出报告没明说的两件事：

- **MATH 用的是 5000 题全集。** 报告写的是「MATH5k set」（PDF p. 5）。Nova Pro 76.6 ±1.2 反解出 $N \approx 4782$，与 5000 相符；若是常见的 500 题子集，同一分数的 CI 应约为 ±3.7。所以这个 MATH 分不能和别家的 MATH500 直接比，报告没做这个提醒。
- **IFEval 的分母是指令，不是提示。** 报告说 IFEval 有 541 条提示（PDF p. 5），但 92.1 ±1.8 反解出 $N \approx 863$，与表头「instruction-level loose accuracy」一致（PDF p. 6）——一条提示里可能有好几条可验证指令。

也有核不上的：ARC-C 的 ±1.3 反解出约 1121，DROP 的 ±0.7 约 9775，EgoSchema 的 ±5.4 约 265，报告没给这三个基准的样本量。其中 DROP 报的是 F1，不是二元得分的平均，按式 1 自己的适用条件本不该套这个公式，表里仍然印了 ±（PDF p. 6）。

### 口径仍然不齐的地方

口径写出来，读者才看得到它不齐。逐条列：

- **对照组的提示方式各不相同。** Table 1 在每组模型下面单印一行提示方式（PDF p. 6）：Nova 的 ARC-C 是 0-shot、GSM8K 是 0-shot CoT；Claude 组的 ARC-C 是 25-shot；Gemini 组的 MATH 是 4-shot、GSM8K 是 11-shot。同一格子里，Nova 和对手用的示例预算并不相同。
- **Nova 在 ChartQA 上没用 CoT，对手都用了。** Table 3 的 ChartQA 列头带备注 (C)：「除 Amazon Nova 外所有模型都用 CoT」（PDF p. 8）。MMMU 列头则标着 (CoT)，与 p. 7 正文「MMMU 用 CoT 提示」一致。
- **正文与表格对不上两处。** §2.1.1 说 DROP 用 0-shot CoT、ARC-C 用 0-shot CoT（PDF p. 5），而 Table 1 写的是 DROP 6-shot CoT、ARC-C 0-shot（PDF p. 6），附录 B 的 DROP 模板也确实带 6 个示例（PDF p. 30）。表格与附录一致，正文有误。
- **Claude 的 IFEval 分另有星号。** 取自 Anthropic 的报告，报告注明其计分方式未说明（PDF p. 6）。
- **Llama 3.2 的文本分借自 Llama 3.1。** Table 4 备注 (A) 说，因为共享同一个文本模型，BFCL 上 Llama 3.2 11B、90B 直接用 Llama 3.1 8B、70B 的榜单结果（PDF p. 9）。Table 1 里 Llama 3.2 11B 与 Llama 3.1 8B 除 tok/sec 和 GPQA（32.8 对 30.4）外五列数字完全相同（PDF p. 6），也符合这个做法，但 Table 1 没有加注。
- **「四个领域」只列了三个。** §2.4 说挑了四个领域，紧接着列出软件工程、金融分析、RAG（PDF p. 11）。
- **同一模型两处速度不同。** Gemini 1.5 Pro (002) 在 Table 1 是 58 tok/sec，在 Table 2 是 57（PDF p. 6–7）。

这些都不是造假，但每一条都会影响「某格 Nova 比对手高几分」能不能按字面读。

### 这一节可以带走什么

- 公开不了机制时，**把口径公开到能被反算**，是最便宜的信任建设：误差棒配公式、提示词进附录、标明哪些对照分是自己重跑的、走的哪条通道。
- 读别人的带 CI 的表，可以**反解样本量**，判断一个「高 2 分」是不是噪声，也能发现 MATH5k 与 MATH500 这类口径差。

## 六个评测面：读表要知道的几件事

完整对照表在原文 Table 1–7。下面每一面只列 Nova 三行和必要的对照，外加读表时容易读错的地方。

### 核心文本能力（Table 1，PDF p. 6）

| 模型 | tok/sec | MMLU | ARC-C | DROP | GPQA | MATH | GSM8K | IFEval | BBH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nova Pro | 100 | 85.9 | 94.8 ±1.3 | 85.4 ±0.7 | 46.9 ±4.6 | 76.6 ±1.2 | 94.8 ±1.2 | 92.1 ±1.8 | 86.9 |
| Nova Lite | 157 | 80.5 | 92.4 ±1.5 | 80.2 ±0.8 | 42.0 ±4.6 | 73.3 ±1.2 | 94.5 ±1.2 | 89.7 ±2.1 | 82.4 |
| Nova Micro | 210 | 77.6 | 90.2 ±1.7 | 79.3 ±0.8 | 40.0 ±4.5 | 69.3 ±1.3 | 92.3 ±1.4 | 87.2 ±2.3 | 79.5 |
| Claude 3.5 Sonnet (Oct) | 57 | 89.3 | 96.3 | 88.3 | 58.0 | 78.3 | 96.5 | 90.2 | 93.2 |
| GPT-4o | 163 | 88.7 | 96.2 | 83.4 | 48.4 | 76.6 | 92.6 | 89.8 | 83.0 |
| Gemini 1.5 Pro (002) | 58 | 85.9 | 95.4 | 74.9 | 55.1 | 86.5 | 90.8 | 91.7 | 89.2 |
| Gemini 1.5 Flash 8B (001) | 283 | 68.1 | 88.7 | 68.1 | 33.5 | 58.7 | 84.5 | 86.1 | 69.5 |

（对照行省去了 ± 与上标，原表都有。）

- 家族内部是干净的单调关系：**越慢越准**，Micro 到 Pro 速度差约 2.1 倍，MMLU 差 8.3 分。这是「档位」的实际含义——延迟与精度曲线上的几个工作点，不是参数量的档位。
- Nova Pro 的 MMLU 与 Gemini 1.5 Pro (002) 同为 85.9，低于 Claude 3.5 Sonnet 3.4 分；GPQA 46.9 明显低于 Claude 的 58.0 与 Gemini 的 55.1；MATH 与 GPT-4o 同为 76.6。
- 报告说 Micro 在它的档位里多项领先（PDF p. 3）。Micro 的 210 tok/sec 在整张表里仅次于 Gemini 1.5 Flash 8B 的 283，而各项分数都高于后者，这一格的说法站得住。

翻译用 Flores200，14 种语言与英语互译，0-shot（PDF p. 7）。Nova Pro 的 en→Set1 spBleu 43.4 是 Table 2 最高，COMET22 89.1 则略低于 Claude 3.5 Sonnet 的 89.4 与 GPT-4o 的 89.2；Set1→en 方向的 spBleu 最高是 Gemini 1.5 Pro 的 45.6，Nova Pro 44.4（PDF p. 7）。翻译是 Nova 与头部对手差距最小的一项。

### 多模态理解（Table 3，PDF p. 8）

| 模型 | MMMU (CoT) | ChartQA | DocVQA | TextVQA | VATEX | EgoSchema |
|---|---:|---:|---:|---:|---:|---:|
| Nova Pro | 61.7 ±3.2 | 89.2 ±1.2 | 93.5 | 81.5 | 77.8 | 72.1 ±5.4 |
| Nova Lite | 56.2 ±3.2 | 86.8 ±1.3 | 92.4 | 80.2 | 77.8 | 71.4 ±5.4 |
| Claude 3.5 Sonnet (Oct) | 70.4 ±3.0 | 90.8 ±1.1 | 94.2 | 61.7 M | — | — |
| Gemini 1.5 Pro (001) | 65.9 ±3.1 | 87.2 ±1.3 | 93.1 | 78.7 | 64.6 | 72.2 ±5.4 |
| GPT-4o (May) | 69.1 ±3.0 | 85.7 ±1.4 | 92.8 | 77.2 | — | 72.2 ±5.4 |
| Llama 3.2 90B | 60.3 ±3.2 | 85.5 ±1.4 | 90.1 | 80.7 M | — | — |

原表还有几条备注：Gemini 1.5 Pro 的 MMMU 实为 (002)、DocVQA 用了外部 OCR、VATEX 是 4-shot；GPT-4o 的 TextVQA 实为 Nov 版且由亚马逊自测（PDF p. 8）。

- 报告说 ChartQA 与 VATEX 上 Nova「排第一或第二」（PDF p. 8）。对表：ChartQA 上 Nova Pro 89.2 次于 Claude 3.5 Sonnet 的 90.8，是第二；VATEX 上 77.8 表内最高。但 ChartQA 这一列 Nova 没用 CoT、对手都用了（见上一节）。
- TextVQA 差距最大：Nova Pro 81.5，Claude 3.5 Sonnet 61.7，后者是亚马逊自测的。
- MMMU 上 Nova Pro 落后 Claude、GPT-4o、Gemini 1.5 Pro 4–9 分；EgoSchema 上与 Gemini 1.5 Pro、GPT-4o 基本打平（72.1 对 72.2）。

### Agentic：函数调用与网页 agent（Table 4–5，PDF p. 9–10）

文本函数调用用 BFCL v3，取 2024-11-17 的榜单与仓库状态（PDF p. 9）。BFCL 的几个判法先说清：AST 比对函数名与参数签名是否与人工标注一致；Execution 不看签名，真的执行调用，比对返回值；Relevance 测「有合适工具时知道该调」；Irrelevance 测「没有合适工具时知道不调」，报告把后者当作幻觉率的度量（PDF p. 9）。

| 模型 | 总分 | 延迟（秒） | 非实时 AST | 非实时 Exec | 实时总分 | 多轮 | Relevance | Irrelevance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Nova Pro | 68.4 | 1.0 | 90.1 | 89.8 | 71.5 | 45.1 | 95.1 | 65.1 |
| Nova Lite | 66.6 | 0.6 | 87.5 | 86.4 | 66.0 | 50.3 | 97.6 | 49.1 |
| Nova Micro | 56.2 | 0.5 | 87.2 | 89.7 | 67.4 | 15.5 | 87.8 | 57.6 |
| Claude Sonnet 3.5 (Jun) | 61.3 | 3.9 | 70.0 | 66.3 | 74.7 | 40.0 | 68.3 | 74.6 |
| Claude Haiku 3 | 40.4 | 1.5 | 41.7 | 47.5 | 57.7 | 20.6 | 97.6 | 29.4 |
| Gemini 1.5 Pro (002) | 59.8 | 3.0 | 88.0 | 91.4 | 74.3 | 16.3 | 75.6 | 75.1 |
| GPT-4o (Aug) | 68.9 | 1.5 | 85.9 | 85.6 | 75.4 | 45.3 | 63.4 | 82.9 |

报告说 Nova 在 AST、Execution、Relevance 与总分上相对可比模型表现突出（PDF p. 9）。对表要打折：总分 Nova Pro 68.4 略低于 GPT-4o 的 68.9；实时总分 Nova 三个都低于 Claude 3.5 Sonnet、Gemini 1.5 Pro 和 GPT-4o；Irrelevance 一列 Nova Lite 49.1、Pro 65.1，明显低于 GPT-4o 的 82.9 和 Gemini 两款的 75 上下。**Nova 很愿意调工具，也更容易在不该调的时候调。** Lite 与 Micro 的延迟确实是表内最低（PDF p. 9）。

网页 agent 三项都以网页截图为输入（PDF p. 9–10）：

| 模型 | VisualWebBench | MM-Mind2Web 步骤准确率 | GroundUI-1K |
|---|---:|---:|---:|
| Nova Pro | 79.7 | 63.7 | 81.4 |
| Nova Lite | 77.7 | 60.7 | 80.2 |
| Claude 3.5 Sonnet (Oct) | 76.7 M | 61.6 M | 16.3 |
| GPT-4o (Nov) | 77.5 M | 55.0 M | 13.4（May 版） |
| Gemini 1.5 Pro (002) | 76.4 M | 58.4 M | 35.2（001 版） |
| Gemini 1.5 Flash (002) | 76.1 M | 46.2 M | 59.9 M |

- §2.2 开头说 Pro 与 Lite 在这三个多模态基准上「创下新的最先进水平」（PDF p. 9）。Pro 三项都是表内第一；Lite 在 VisualWebBench 与 GroundUI-1K 上也高于所有对照，但 MM-Mind2Web 的 60.7 低于 Claude 3.5 Sonnet 的 61.6。这句话对 Lite 只成立三分之二。
- GroundUI-1K 的差距最夸张：Nova Pro 81.4，Claude 3.5 Sonnet 16.3，GPT-4o 13.4。这个基准给一张截图和一条指令，要求输出目标 UI 元素的二维位置，落在标注框内才算对（PDF p. 10），提示词要求用 0–1000 归一化的坐标框作答（PDF p. 38）。**本文的解读**：差距主要反映「是否专门训过输出屏幕坐标」，而不是通用视觉能力差五倍；Gemini 1.5 Flash 实测 59.9 也说明对照模型之间差别很大。报告没有解释这个差距的成因。

### 长上下文（Figure 2 与 Table 6，PDF p. 10–11）

![Figure 2：Nova Micro、Lite、Pro 的文本大海捞针热力图。横轴是上下文长度，纵轴是针的位置，所有格子都在绿色区间，最浅的是 Lite 在 256K、针位 40% 那一格。](/reports/Nova/figure2-niah.png)

上下文上限是 Micro 128K、Lite 与 Pro 300K，测试从 32K 起（PDF p. 10）。热力图没有标数字，只能按色标读：全部格子都在色标 75 以上的绿色段，没有黄色或红色；最浅的几格集中在 Lite 的 200K–300K、针位 30%–50%，以及 Micro 的 128K、针位 90%（PDF p. 11，本文读色）。正文说模型能「从任意深度」检索信息（PDF p. 11），按图看大体成立，但不是每格都满。

| 模型 | SQuALITY ROUGE-L | LVBench 准确率 |
|---|---:|---:|
| Nova Pro | 19.8 ±8.7 | 41.6 ±2.5 |
| Nova Lite | 19.2 ±8.6 | 40.4 ±2.4 |
| Nova Micro | 18.8 ±8.6 | — |
| Gemini 1.5 Pro (002) | 19.1 ±8.6 M | — |
| GPT-4o | 18.8 ±8.6 | 30.8 ±2.3 |
| Gemini 1.5 Pro (001) | — | 33.1 ±2.3 |

（PDF p. 11）SQuALITY 的 ± 宽到 8 分以上，19.8 与 18.8 在统计上分不开。LVBench 是 99 个视频、1549 道题的长视频问答（PDF p. 10），Nova Pro 41.6 高出对照约 8–11 分，对照分取自该基准的排行榜（PDF p. 11）。

### 领域任务（Table 7，PDF p. 12）

| 模型 | HumanEval pass@1 | FinQA | CRAG |
|---|---:|---:|---:|
| Nova Pro | 89.0 ±4.8 | 77.2 ±0.9 | 50.3 ±1.9 |
| Nova Lite | 85.4 ±5.4 | 73.6 ±0.9 | 43.8 ±1.9 |
| Nova Micro | 81.1 ±6.0 | 65.2 ±1.0 | 43.1 ±1.9 |
| Claude 3.5 Sonnet (Oct) | 93.7 ±3.7 | 77.3 ±0.9 M | 52.6 ±1.8 M |
| GPT-4o | 90.2 ±4.6 | 71.1 ±1.0 M | 52.0 ±1.9 M |

CRAG 的设置值得看清（PDF p. 12）：用 Task 1 设定，每题给 5 个预选网页；按官方仓库的做法用 BeautifulSoup 清洗 HTML、切成不超过 1000 字符的块，用 `sentence-transformers/all-MiniLM-L6-v2` 编码，取最相似的 20 块作为上下文；判分由 `gpt-4-turbo-2024-04-09` 按附录 B.3.2 的提示词完成。对照模型的 CRAG 都是亚马逊按同一管线自测的。所以这一列测的是「在这条固定检索管线下读材料答题的能力」，不是一套 RAG 系统的整体水平。

### 运行时（Figure 3，PDF p. 14）

Figure 3 是三张柱状图，按原图数据标签转成表（Artificial Analysis，2024-11-29，1000 输入 / 100 输出）：

| 模型 | TTFT（秒，越低越好） | OTPS（越高越好） | 总响应时长（秒，越低越好） |
|---|---:|---:|---:|
| Llama 3.2 11B | 0.29 | 124 | 1.1 |
| **Nova Micro** | 0.32 | 210 | 0.8 |
| Gemini 1.5 Flash 8B | 0.35 | 283 | 0.7 |
| Gemini 1.5 Flash (Sep) | 0.35 | 190 | 0.9 |
| Mixtral 8x7B | 0.36 | 115 | 1.3 |
| Llama 3.1 8B | 0.36 | 157 | 1.0 |
| **Nova Lite** | 0.37 | 157 | 1.0 |
| **Nova Pro** | 0.38 | 100 | 1.4 |
| Llama 3.1 70B | 0.42 | 73 | 1.7 |
| GPT-4o | 0.42 | 163 | 1.2 |
| Llama 3.2 90B | 0.46 | 40 | 2.9 |
| Mistral Large 2 (Nov) | 0.53 | 35 | 3.4 |
| GPT-4o mini | 0.62 | 113 | 1.5 |
| Llama 3.1 405B | 0.72 | 29 | 4.0 |
| Claude 3.5 Haiku | 0.72 | 64 | 2.4 |
| Claude 3.5 Sonnet (Oct) | 0.87 | 57 | 2.7 |
| Gemini 1.5 Pro (Sep) | 0.98 | 58 | 2.8 |

- 三个 Nova 的首 Token 都在 0.4 秒以内，但首 Token 快不等于端到端快：Nova Pro 的 TTFT 0.38 秒比 GPT-4o 的 0.42 略快，OTPS 却只有 100 对 163，总时长 1.4 秒反而慢于 GPT-4o 的 1.2 秒。
- 只输出 100 个 Token 时总时长已经被 OTPS 主导；输出越长，OTPS 的权重越大。选型时应按自己的输出长度看，而不是只看 TTFT。
- 报告的结论是三者「都在各自档位里最快之列」（PDF p. 13）。Micro 的 TTFT 其实次于 Llama 3.2 11B，OTPS 与总时长次于 Gemini 1.5 Flash 8B，「之列」这个措辞是准确的。

## Canvas 与 Reel：没有客观判分器时怎么评

**Canvas 的自动指标**（Table 8，PDF p. 15）：ImageReward 在 MSCOCO-2014 验证集随机抽 10k 条提示上算；TIFA 用 TIFA-v1.0 预选的 4k 条提示，通过视觉问答检查图文是否一致。

| 模型 | TIFA | ImageReward |
|---|---:|---:|
| Nova Canvas | 0.897 | 1.250 |
| DALL.E 3 | 0.863 | 1.052 |
| Stable Diffusion 3.5 Large | 0.891 | 1.082 |
| Stable Diffusion 3 Medium | 0.881 | 0.952 |
| Flux Pro 1.0 | 0.875 | 1.075 |
| Flux Schnell | 0.882 | 0.999 |

TIFA 上 Nova 与 SD 3.5 Large 只差 0.006，ImageReward 上领先较明显。

**Canvas 的人评**（PDF p. 15–16）：约 1000 条提示，取自 MSCOCO、DrawBench、OpenParti、DALL.E 3 Eval、DOCCI 等，少数提示被随机重复以多取数据点；所有图都是 1k×1k；单盲、成对、顺序随机；交给第三方标注商，先按指南培训。**任一方的安全过滤器挡住、没有出图的提示直接丢掉，不给标注员看。**

| Nova Canvas 对 | 维度 | 胜 | 平 | 负 |
|---|---|---:|---:|---:|
| DALL.E 3 | 整体偏好（图像质量） | 54.5 | 6.4 | 39.1 |
| DALL.E 3 | 指令跟随（图文对齐） | 39.4 | 22.5 | 38.1 |
| Imagen 3 | 整体偏好（图像质量） | 48.2 | 5.3 | 46.5 |
| Imagen 3 | 指令跟随（图文对齐） | 38.4 | 28.1 | 33.5 |

（PDF p. 16，单位 %）整体偏好上对 DALL.E 3 领先明显，对 Imagen 3 接近打平；指令跟随维度平局率升到 22%–28%，胜负差缩到 1–5 个点。**「看着更好」和「更听话」是两件事，后者的优势小得多。** 报告正文只说 Canvas 胜率更高（PDF p. 16），没有单独讨论这一行。

**Reel 的人评**（PDF p. 16–17）：约 700 条提示，全部来自开源基准，分六大类，刻意覆盖镜头运动、主体随时间变化的动态属性、多个动作组合的运动绑定三类难点；720p、不同随机种子；单盲成对比较，两个轴——视频质量（单帧质量、运动质量、图文对齐、运动与文本对齐）与视频一致性（主体与背景在时间上是否稳定）。每对比较由 3 名标注员独立判、多数投票；每批随机抽 5%–10% 由专家复核。

| Nova Reel 对 | 维度 | 胜 | 平 | 负 |
|---|---|---:|---:|---:|
| Runway Gen3 Alpha | 视频质量 | 56.4 | 9.9 | 33.7 |
| Runway Gen3 Alpha | 视频一致性 | 67.0 | 9.1 | 23.9 |
| Luma 1.6 | 视频质量 | 51.1 | 3.4 | 45.5 |
| Luma 1.6 | 视频一致性 | 74.7 | 5.1 | 20.2 |

（PDF p. 17，单位 %）一致性上的优势明显大于质量上的优势，对 Luma 1.6 的质量项 51.1 对 45.5 已接近打平。报告没有解释原因。

还有一处要留意：自动指标比的是 DALL.E 3、SD 3.5 Large、SD 3 Medium 与 Flux，人评比的是 DALL.E 3 与 Imagen 3（PDF p. 15–16），两组对手不是同一批，Imagen 3 没有自动指标，SD 与 Flux 没有人评。

## Responsible AI：给了过程，没给结果

§5 从 p. 17 写到 p. 21，是评测之外最厚的一节。结构是三段：定目标 → 保证达标的手段 → 评测与红队。

### 目标与承诺

八个维度各一句定义（Table 11，PDF p. 18）：公平（Fairness）、可解释（Explainability）、隐私与安全（Privacy and security）、安全（Safety）、可控（Controllability）、真实与鲁棒（Veracity and robustness）、治理（Governance）、透明（Transparency）。目标还参照法规、自愿框架与客户承诺，并列了参与的外部机制：Frontier Model Forum、Partnership on AI、NIST 组织的论坛、白宫自愿承诺、英国与首尔的 AI 安全峰会、G7 广岛进程行为准则；另与 METR 合作，用来充实「可控」这一维度的设计目标（PDF p. 18）。

### 手段：对齐之外还有两道运行时防线

- **对齐这一层**：管模型行为的五个维度（安全、公平、真实与鲁棒、可控、隐私与安全）从预训练数据筛选开始，再用 SFT 与 RLHF 对齐。按每个维度造了多语言的单轮与多轮 RAI 示范，用有用性与有害性研究决定 SFT 的数据配比；RLHF 里另有一个 RAI 专用奖励模型；离线评测与红队暴露的风险，会收集语义相近的样例进下一轮 SFT 与 RLHF（PDF p. 18）。
- **运行时这一层**：单独训练了输入审核模型与输出审核模型，作为第一道和最后一道防线。输入侧拦恶意、不安全、违法或试图绕过对齐（提示注入、越狱）的提示；输出侧检查内容是否符合 RAI 目标。理由是这样「能更快响应新发现的威胁或对齐缺口」（PDF p. 18）。**本文的解读**：改模型要重训，改审核只需重新上线一个小模型，两者的响应周期不在一个量级。
- **鲁棒性按后果组织**：面向开发者与终端用户的风险归成四类——敏感数据外泄、执行未授权动作、运行时服务可用性下降、恶意内容生成（PDF p. 19）。
- **隐私**：训练数据、权重与模型版本有访问控制；可行时对训练数据里的某些个人数据去标识或删除（PDF p. 19）。
- **透明**：图像与视频生成时嵌入不可见水印，加固了对旋转、缩放、颜色反转、翻转的鲁棒性；视频逐帧嵌入并能扛住 H.264 压缩；Canvas 所有产物加 C2PA 元数据；检测输出置信度分数而不是单一二值判断，能反映内容被编辑的程度，检测 API「发布后不久」提供（PDF p. 19）。

### 评测与红队

静态评测用 BOLD、RealToxicityPrompts、MM-SafetyBench 等公开基准，加上一批持续更新的私有基准，由内部标注团队按维度造例、安全与可控等领域的专家补对抗提示，随红队结果不断扩充，覆盖多语言、多模态、单轮与多轮（PDF p. 19）。

报告把静态基准和红队分得很清楚：静态基准测的是「明说要违规内容」的直白意图，红队测的是**掩盖意图**的技巧（PDF p. 19）。内部红队通过语言、结构、模态三类变异扩充人工对抗提示，共识别和开发了 300 多种技术，单独测也串联组合测，还设计了跨模态攻击，例如把对抗内容藏进看似无害的图片（PDF p. 19）。Figure 4 把这些技术按三层展开，根节点是 307（PDF p. 20）：

| 一级类别 | 数量 | 二级细分（数量） |
|---|---:|---|
| Prompt injections（直接与间接） | 40 | Obfuscation 8（其下 Cipher-based 3、Text based obfuscation 5）、Code injection 9、Recursive injection 1、Virtualization 9、Defined dictionary attack 1、Payload splitting 6、Token smuggling 6 |
| Jailbreak | 24 | one-shot 12、many-shot 12 |
| Multiple languages | 33 | Multilingual prompting 14、Translation requests 8、Mixed language requests 11 |
| Context-based | 32 | In-Context Learning 14、Context switching 18（其下 Syntactic separators 8、Semantic separators 10） |
| Persuasion | 119 | Instruction repetition 19、Completion Compliance 13、Affirmative Suffixes 13、One-sided arguments 12、Refusal suppression 10、Chain of utterances 10、Socratic Questioning Technique 15、Personification 10、Task constraints 17 |
| Obfuscations | 59 | Veiled Expressions 16、Output constraints 20、Euphemisms via Ciphers 11、Decoding Manipulation 11、Macaronic prompt 1 |

表由本文从桑基图逐项读出。六个一级类别合计 307，与根节点一致；每类的子项也各自加回父项。**Persuasion（说服）一类就有 119 种，接近四成**，子项几乎都是话术：重复指令、肯定式后缀、单边论证、压制拒答、苏格拉底式追问、拟人化。按技术数量看，这份分类里越狱的主要面是话术，不是编码花招。

外部红队四家（PDF p. 20–21）：

| 机构 | 领域 | 报告给出的规模与方法 |
|---|---|---|
| ActiveFence | 仇恨言论、政治虚假信息、极端主义等内容安全，以及安全向测试 | 150 多名领域专家；9700 多条对抗提示，分布在 20 多个类别 |
| Deloitte Consulting（原 Gryphon Scientific） | 生物 | 一套 30 道题，测可能助长生物武器研发或使用的科学知识与推理；按科学准确性和对作恶者的有用度评分，再对最初给出可疑信息的题深挖 |
| Gomes Group（卡内基梅隆大学） | 化学 | 两项非自动评测考察通过采购与远程化学混合场景的聚合攻击；自动评测用两套数据：39 种危险化学品（含 DEA 一类、二类与化学战剂）、362 种常见化学品；其中 NFPA 菱形分级评测 1810 条提示，单轮与多轮准确率一致 |
| Nemesys Insights | 放射性与核 | 两个场景（非国家行为体获取并使用钴-60；获取并跨国运输高浓铀）；8 名领域专家，2 队 × 2 场景，6 小时规划周期 |

自动红队改造自家的 FLIRT 框架：从人工判定可能违规的种子提示出发，用专门的 red-LM 通过上下文学习批量生成新提示，评估回复，把真正触发违规的提示留作下一轮的种子，迭代若干轮；同一套机制也用来测 false refusal，覆盖多轮、多语言与多种输入输出模态（PDF p. 21）。

### 这一节缺的恰恰是结果

整节没有一个拒答率、有害输出率、偏见分或毒性分。给了基准的名字、私有基准的建法、红队的**工作量**，没有**结果**。外部机构测完得出了什么结论，报告也只写「据反馈迭代改进」（PDF p. 20）。

外部补充：AWS 的 AI Service Card（适用于截至 2025-04-30 的版本，含 Premier）给过一部分结果数字，例如在 2.4k 条诱导有害内容的私有集上平均超过 90% 的提示得到安全回答、BOLD 上 99.9% 以上的补全被判为公平。那是另一份文档、覆盖的模型版本也不同，不属于这份报告，链接见文末。

### 这一节可以带走什么

- **把「做了多少测试」和「测出来多少」分开读。** 读别人的安全章节先找可比较的分数；写自己的，别用工作量代替结论。
- **审核模型与主模型解耦**，让新攻击能在上线小模型的周期里被挡住，不必等下一轮对齐训练。
- **按后果给 agent 风险分类**（数据外泄、未授权动作、可用性下降、恶意内容），比按关键词分更稳。

## 训练基础设施：全文唯一的工程数字

### 硬件与集群

Nova 家族在三种加速器上训练：亚马逊自研的 Trainium1（TRN1）、NVIDIA A100（P4d 实例）与 H100（P5 实例）（PDF p. 21）。报告最值得注意的一句是：与 SageMaker 一起搭起 NVIDIA GPU 与 TRN1 两类集群，**并行跑训练以确保模型性能一致**，同时分别优化两套栈的吞吐（PDF p. 21）。也就是说，自研芯片不是只拿来省钱的备选路径，而是被要求在结果上与 NVIDIA 路径等价。报告没有给出一致性验证的任何数字。

其余环境（PDF p. 21）：所有集群用 PB 级无阻塞 EFA 网络，报告称它比其他网络传输协议更不易丢包，并且在 H100 上提供 EC2 所有实例类型里最高的网络带宽；分布式训练跑在 SageMaker 托管的 EKS 集群上；数据与检查点 IO 用 FSx 与 S3——FSx 给大任务提供高性能、方便的存储，S3 让大规模多模态数据与检查点能低成本扩展。

### 恢复时间：压的是协调

![一次重启从中断到回到稳态的环节：重启组件、找到最新检查点（3 分钟降到 5 秒）、各节点只加载自己 rank 的文件、数据加载初始化 205 毫秒；常驻的异步观察进程预先把检查点文件映射到节点。TRN1 集群 MTTR 目标 9 分钟，实际平均 6.5 分钟。](/reports/Nova/figure-restart-path.svg)

报告说预训练的周平均 goodput 最高达到 97%，靠三路优化：降低任务失败率、压检查点开销、压 MTTR（PDF p. 21）。MTTR 的口径写得很老实：从中断前最后一次成功检查点算起，包括重启各组件、从检查点恢复到稳态训练的全部时间（PDF p. 21）。

图上的每一格都对应报告的一句话（PDF p. 21–22）：

- 优化器状态与权重完全分片、去掉检查点持久化的一切阻塞开销后，存一次检查点的开销降到 H100 集群约 1 秒、TRN1 集群约 0.1 秒。两者为什么差十倍，报告没说。
- 优化训练启动时的节点通信初始化，并用一个**异步观察进程**缩短检查点加载：它持续把每个最新检查点文件映射到对应节点，恢复时每个节点只加载自己 rank 的那部分，于是「找到最新检查点」从 3 分钟降到 5 秒。
- 缓存并复用数据索引，数据加载初始化降到每次重启 205 毫秒。
- 合起来，TRN1 集群的平均 MTTR 是 6.5 分钟，低于 9 分钟的目标。

**本文的解读**：3 分钟到 5 秒这一段省下的不是读盘带宽，而是「最新检查点在哪、哪份是我的」这类元数据协调。把它交给一个常驻进程提前算好，关键路径上就只剩真正的读取。

### 显存与通信

- **SSC（Super-Selective Activation Checkpointing）**：自研的激活检查点方案，在显存受限的环境里尽量少做重算。与 NVIDIA 的 Selective Checkpointing 相比，**显存占用降低约 50%，重算开销只增加约 2%**（PDF p. 22）。它具体挑哪些激活保存、哪些重算，报告没写。
- **两处默认行为**：默认的梯度规约行为导致通信重叠不理想，调整规约的顺序与频率后，大部分数据并行通信被藏进计算里；默认 PyTorch 内存分配器的同步性质，会在集合通信里造成掉队者（straggler），拖停多个 worker（PDF p. 22）。报告只说发现了分配器问题，没说怎么改的。

分配器这一条尤其值得记：单卡上完全看不出来，只有几千个 rank 要在一个集合通信点会合时才显形，而且表现为「大家都慢」而不是「谁报错」。

### 这一节可以带走什么

- 重启时间先按「谁在等」拆开，**元数据与协调常常比读盘更贵**；能提前算好的映射交给常驻进程。
- 引入新硬件栈时，把**结果等价**写成验收条件并真的并行跑一遍，比吞吐数字更能建立信任。
- 性能问题要在目标规模上看：分配器、梯度规约这类默认行为，只有在大规模集合通信里才暴露。

## 报告没有公开的

- 参数量、层数、隐藏维度、头数、是否 MoE、注意力与位置编码、词表与分词器；
- 预训练 Token 总量、数据配比与各来源占比、去重与质量过滤的细节、污染检查、知识截止日；
- 优化器、学习率、batch size、步数、并行拓扑、训练时长、芯片数量与总算力；
- 后训练各阶段的顺序与数据规模、DPO 与 PPO 各用在哪一段、奖励模型的规模与训练数据；
- Canvas 与 Reel 的网络结构、VAE 与文本编码器是什么、训练规模；
- 任何价格数字；
- 任何安全评测的结果分数，以及外部红队的结论；
- SSC 的具体做法，TRN1 与 NVIDIA 两条路径「性能一致」的验证数据；
- Nova Premier 的一切；
- 权重与代码。附录 B 给的 `huggingface.co/amazon-agi`（PDF p. 30）是评测补充材料的入口，不是权重。

## 最后的判断

这份文件名为 Technical Report and Model Card，实际是一份**选型与合规文档**：它回答按档位该选哪个、快不快、风险查过哪些、训练管线靠不靠得住，把架构、数据与超参整段留白。这是闭源商品的体裁选择，不是疏漏。

证据强度分三层：

- **有表有数、口径可查的**：三个理解模型在各基准上的分数（Table 1–7），家族内部越慢越准的单调关系；式 1 与表中 ± 号自洽（本文用六个基准验算）；Canvas 与 Reel 的人评胜负率；训练管线的检查点开销、MTTR、找检查点的耗时、SSC 的两个百分比。
- **只是作者的定性说法**：「在各自智能档位中领先」——档位从未定义（PDF p. 3）；「首批在 Bedrock 上提供视频理解」（PDF p. 3）；「从任意深度检索」——热力图没有数字（PDF p. 11）；TRN1 与 NVIDIA 路径「性能一致」（PDF p. 21）。
- **需要自己打折的**：BFCL 上「总分突出」其实略低于 GPT-4o；Lite 在 MM-Mind2Web 上不是第一；ChartQA 上 Nova 与对手的提示方式不同；GroundUI-1K 的五倍差距反映的是输出接口，不是视觉能力。

如果只带走一句：

> **读一份不写架构的报告，要看它选择把哪些东西写到可核对——那就是它希望你据以决策的部分；其余的空白，不要自己去填。**

## 可以带回自己项目的几条

1. **规格表用调用方的语言写。** 延迟、单价、目标任务上的分数三列并排，比参数量有用得多；Nova 在每张精度表里放 tok/sec，成本几乎为零。
2. **自报分数配能反算的误差棒，提示词全文进附录，标明哪些对照分是自己重跑的、走的哪条通道。** 不泄露机密，却能显著降低外人质疑的成本。
3. **读 CI 表时反解样本量**，顺手检查 MATH5k 与 MATH500 这类口径差。
4. **说「第一」之前先对一遍同页的表。** Nova 的 BFCL 总分与 Lite 的 Mind2Web 都是被自己的表削掉的。
5. **审核层与模型层解耦**；**agent 风险按后果分类**。
6. **重启时间按「谁在等」分解**，把元数据协调移出关键路径。
7. **激活检查点的取舍粒度越细越好**，SSC 的「约 50% 显存换约 2% 重算」可作参照点。
8. **新硬件栈以结果等价为验收条件**，并行跑一遍再说。

不该照搬的：完全不写架构是闭源商品的选择，想要社区信任与学术引用的项目不该学；运行时数据借第三方，前提是有现成的跨托管商口径。

## 关键词回看

- **智能档位**：报告用来代替参数量的组织单位，未定义。
- **TTFT / OTPS / 总响应时长**：首 Token 延迟、每秒输出 Token 数、端到端时长；短输出看 TTFT，长输出看 OTPS。
- **式 1 置信区间**：$CI(S)=1.96\sqrt{S(1-S)/N}$，适用于二元得分的平均，可反解样本量。
- **上标 M**：亚马逊自己在对手 API 上重跑的对照分。
- **BFCL 的 AST / Execution / Relevance / Irrelevance**：签名比对、执行比对、该调时调、不该调时不调。
- **GroundUI-1K**：给截图与指令，输出目标 UI 元素的位置。
- **TIFA / ImageReward**：图像生成的两个自动指标，前者用视觉问答查图文一致性。
- **双审核模型**：输入侧与输出侧各一个，与主模型解耦。
- **FLIRT**：自动红队框架，red-LM 从种子提示迭代生成对抗提示，兼测 false refusal。
- **307 种攻击技术**：Figure 4 的三层分类，Persuasion 占 119。
- **不可见水印与 C2PA**：生成内容的两层溯源。
- **Goodput / MTTR**：推进训练的时间占比 / 从上次成功检查点到回到稳态的时间。
- **SSC**：Super-Selective Activation Checkpointing，约 50% 显存换约 2% 重算。

## 资料与阅读边界

**原始依据**

- 本地 `papers/Amazon/Nova.pdf`，即 arXiv:2506.12103v1（2025-03-17 提交），48 页。arXiv 页面只有 v1 这一个版本，与本地一致。编号前缀是 2506 而提交日是 3 月，arXiv 页面没有解释，本文以页面记录的提交日为准。报告附录 D 要求引用时以「Amazon AGI」为唯一作者，bibtex 年份写 2024，正式 URL 在 Amazon Science（PDF p. 43）。

**首发日证据**

- AWS What's New 发布日志 [Announcing Amazon Nova foundation models available today in Amazon Bedrock](https://aws.amazon.com/about-aws/whats-new/2024/12/amazon-nova-foundation-models-bedrock/) 标注 Posted on: Dec 3, 2024，写明 Micro、Lite、Pro、Canvas、Reel 当天在 Amazon Bedrock 可用。
- 同日官方新闻稿 [Introducing Amazon Nova](https://press.aboutamazon.com/2024/12/introducing-amazon-nova-a-new-generation-of-foundation-models) 与 [AWS News Blog](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-frontier-intelligence-and-industry-leading-price-performance/) 与之一致。
- 因此 `release-date` 取 **2024-12-03**，即家族最早对外可用日，而不是 arXiv 提交日（2025-03-17）。

**外部补充（不是报告内容）**

- Nova Premier 与价格的相对说法，来自上面的官方新闻稿。
- 安全评测的结果数字见 AWS 的 [Amazon Nova AI Service Card](https://docs.aws.amazon.com/ai/responsible-ai/nova-micro-lite-pro/overview.html)，覆盖的模型版本与本报告不同。
- 报告依赖的两处外部口径：Berkeley Function Calling Leaderboard（2024-11-17 快照，PDF p. 9）与 [Artificial Analysis 方法说明](https://artificialanalysis.ai/methodology)（2024-11-29 取数，PDF p. 13–14）。

**本文的推算与读图**

- 式 1 的 CI 验算与样本量反解是本文的算术；Figure 2 的深浅、Figure 3 与 Figure 4 转成的表格是本文读图所得，Figure 2 原图没有数字。
