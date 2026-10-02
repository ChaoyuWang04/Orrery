# alphaXiv 扫描记录

下划线开头,不出现在网页。每日扫描 alphaXiv 热榜的历史:一篇论文在首发 7 天内**第一次**达到票数线的那天记一行,之后不再重复记。规则见 [10-材料解读流程](../docs/10-材料解读流程.md) 第二节「来源二」。

- `npm run papers:feed -- --write` 追加新行,判定与去向先填脚本初判;核实后由维护者或定时任务改成终判
- **票数**是达标那天扫到的 `public_total_votes`(网页点赞按钮上的数),之后不回写。它是两个解读库「按时间」视图里同一时段内的排序依据
- **去向**写 `reports/<公司>/<材料>` 或 `readings/<方向>/<材料>` 时,页面按这个路径把票数挂到卡片上;没入库的写 `待定`、`C · 待拍板` 或 `不收:理由`
- 判定取值:`A · 名单内` / `B · 名单外过闸门三` / `C · 待拍板` / `D · 已在库` / `待核 · 名单外`(脚本初判,核实后改成 B 或 C)

| 达标日 | 编号 | 标题 | 机构 | 票数 | 判定 | 去向 |
|---|---|---|---|---|---|---|
| 2026-09-23 | 2609.19969 | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | Deepseek | 570 | D · 已在库 | reports/DeepSeek/DeepSeek-V4.1-Flash |
| 2026-09-23 | 2609.mimo-scaling-reinforcement-learning | MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement | Xiaomi | 174 | A · 名单内(Xiaomi) | reports/Xiaomi/MiMo-V2.6 |
| 2026-09-23 | 2609.18207 | Reinforcement Learning for Real-Time Vision-Language-Action Policies | Stanford University | 168 | A · 名单内(Stanford) | reports/Stanford/Real-Time-EXPO-FT |
| 2026-09-23 | 2609.20807 | Score Centering Stabilizes Off-policy Reinforcement Learning | Together AI | 158 | A · 名单内(TogetherAI) | reports/TogetherAI/Score-Centering |
| 2026-09-23 | 2609.20519 | SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness | NVIDIA、Nanyang Technological University、Massachusetts Institute of Technology | 127 | A · 名单内(NVIDIA) | reports/NVIDIA/SoL-Pi |
| 2026-09-23 | 2609.20800 | JEPA-Anything: Learning Predictive Models across Different Worlds | Phi AI Labs | 118 | A · 名单内(CUHK) | reports/CUHK/JEPA-Anything |
| 2026-09-23 | 2609.19138 | In-Context Robot Learning with VLM Agents | Morphi Robot、Shanghai Innovation Institute、Huazhong University of Science and Technology、Fudan University | 97 | B · 名单外 | readings/世界模型与 Agent/GPT-Policy |
| 2026-09-23 | 2609.20612 | What Does Privileged Information Add to On-Policy Self-Distillation? | National University of Singapore | 65 | A · 名单内(NUS) | reports/NUS/Privileged-Info-OPSD |
| 2026-09-23 | 2609.21561 | On Repulsive and Attractive Teachers: Separating Correctness from Behavior in Self-Distillation | ETH Zürich、Max Planck Institute for Intelligent Systems | 58 | A · 名单内(ETH) | reports/ETH/Repulsive-Self-Distillation |
| 2026-09-23 | 2609.stable-unstable-singularities-navier-stokes | Stable and Unstable Singularities in Navier-Stokes | (alphaXiv 未标) | 54 | 不收 | 不收:数学分析论文(Navier-Stokes 奇点理论),非计算机研究 |
| 2026-09-23 | 2609.20784 | RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning | Zhejiang University、Alibaba Group | 53 | A · 名单内(ZJU) | reports/ZJU/RetireOPD |
| 2026-09-23 | 2609.reinforcing-agents-collective-skills | Reinforcing Agents with Collective Skills | NVIDIA | 52 | A · 名单内(NVIDIA) | reports/NVIDIA/Skill2Env |
| 2026-09-23 | 2609.19107 | How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents | New York University、Q Labs | 52 | A · 名单内(NYU) | reports/NYU/Model-Growth-Scaling-Exponents |
| 2026-09-23 | 2609.19134 | ScienceIDE: Turning World's Scientific Codebase into Agent Learnable Environments | AItonomyFoundation、PhAI-Labs | 47 | A · 名单内(Oxford) | reports/Oxford/ScienceIDE |
| 2026-09-23 | 2609.22068 | CodeMidas: Scaling Agentic Coding RL Environments from Code Itself | Xiaomi、Peking University、University of Hong Kong、Renmin University of China | 46 | A · 名单内(Xiaomi) | reports/Xiaomi/CodeMidas |
| 2026-09-23 | 2609.ier-opd | 1% of Tokens Can Be Enough: On Gradient Estimation in On-Policy Distillation | Ant Group、MBZUAI | 44 | A · 名单内(MBZUAI) | reports/MBZUAI/IER-OPD |
| 2026-09-23 | 2609.20804 | An Empirical Study of Harness Design for Coding Agents | University of Massachusetts Amherst、Emory University、Zoom Video Communications | 43 | B · 名单外 | readings/Agent 训练与工具使用/Coding-Harness-Design |
| 2026-09-23 | 2609.18708 | Rethinking Critic Learning in PPO: Understanding and Mitigating Value Flattening | Shanghai Jiao Tong University、Shanghai Artificial Intelligence Laboratory、Westlake University、Nanjing University | 41 | A · 名单内(SJTU) | reports/SJTU/Value-Flattening |
| 2026-09-23 | 2609.20794 | PosteriorBench: From Point Estimates to Posterior Matching in Evaluating Generative Inverse Solvers | California Institute of Technology、National Taiwan University、Lawrence Berkeley National Laboratory、Stanford University | 36 | A · 名单内(Caltech) | reports/Caltech/PosteriorBench |
| 2026-09-23 | 2609.20649 | DexTouch-WM: Learning Action-Conditioned Tactile World Models from Human Touch for Dexterous Robot Manipulation | The Hong Kong University of Science and Technology (Guangzhou)、Xspark AI、Peking University、University of Hong Kong | 34 | A · 名单内(HKUST) | reports/HKUST/DexTouch-WM |
| 2026-09-23 | 2609.19101 | Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations | Goodfire | 32 | B · 名单外 | readings/可解释性与对齐/Reward-Hacking-Probes |
| 2026-09-23 | 2609.19644 | ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI | Google Cloud AI Research、University of Waterloo | 30 | A · 名单内(Google) | reports/Google/ScientistTwo |
| 2026-09-23 | 2609.24972 | RRSI: Regularized Recursive Self-Improvement of Agent Harnesses | Google Cloud AI Research、University of North Carolina at Chapel Hill、Stanford University、Washington University in St. Louis | 30 | A · 名单内(Google) | reports/Google/RRSI |
| 2026-09-24 | 2609.24984 | WorldCrafter: Consistent Video World Model with Implicit 3D-aware Memory | Tencent、Peking University | 40 | A · 名单内(Tencent) | reports/Tencent/WorldCrafter |
| 2026-09-24 | 2609.22682 | Self-Organizing Agent Teams Learn to Reason Together | Stanford University、Together AI、Emory University | 36 | A · 名单内(Stanford) | reports/Stanford/Self-Organizing-Agent-Teams |
| 2026-09-24 | 2609.2609-opus-5-5 | Claude Opus 5.5 System Card | Anthropic | 35 | A · 名单内(Anthropic) | reports/Anthropic/Claude-Opus-5.5 |
| 2026-09-24 | 2609.23986 | Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents | University of Texas at Dallas | 33 | B · 名单外 | readings/Agent 训练与工具使用/Jev-Mem |
| 2026-09-24 | 2609.20744 | Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation | University of California, Berkeley、Impossible, Inc.、University of Texas at Austin | 32 | A · 名单内(Berkeley) | reports/Berkeley/Video-DeltaNet |
| 2026-09-24 | 2609.20820 | Workspace Models: Lightweight Robotic Memory via Saliency-Driven Supervision | Massachusetts Institute of Technology、Carnegie Mellon University | 32 | A · 名单内(MIT) | reports/MIT/Workspace-Models |
| 2026-09-24 | 2609.23881 | MotionJEPA: Preventing Temporal Feature Collapse by Capturing Visual Changes in Latent Space | University of Oxford、vivo Tech Research GmbH、Bielefeld University、Slater Labs | 31 | A · 名单内(Oxford) | reports/Oxford/MotionJEPA |
| 2026-09-24 | 2609.24974 | Harness-Zero: Harness Distillation via Agent-as-Harness | Peking University、Google、Hong Kong University of Science and Technology | 31 | A · 名单内(Peking) | reports/Peking/Harness-Zero |
| 2026-09-24 | 2609.kl-regularized-policy-optimization-agentic-rl | KL-Regularized Policy Optimization for Critic-Free Agentic Reinforcement Learning | (alphaXiv 未标) | 31 | B · 名单外 | readings/训练方法与强化学习/KLPO |
| 2026-09-24 | 2609.24981 | GAE: Learning a Geometry-Native Latent Space for 3D-Consistent World Generation | Hong Kong University of Science and Technology、Tencent、University of Hong Kong、University of Texas at Austin | 30 | A · 名单内(HKUST) | reports/HKUST/Geometric-AutoEncoder |
| 2026-09-25 | 2609.22978 | DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale | Deepseek、Tsinghua University | 45 | D · 已在库 | reports/DeepSeek/DSec |
| 2026-09-25 | 2609.26368 | HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing | Xiaomi | 40 | A · 名单内(Xiaomi) | reports/Xiaomi/HySparse2 |
| 2026-09-25 | 2609.28399 | Memory Attention | (alphaXiv 未标) | 38 | B · 名单外 | readings/注意力与长上下文/Memory-Attention |
| 2026-09-25 | 2609.24352 | Few-Shot Demonstrations Elicit the Use of In-Context World Representations in LLMs | University of Tokyo、Harvard University、Prior Computers | 38 | B · 名单外 | readings/可解释性与对齐/Few-Shot-World-Representations |
| 2026-09-25 | 2609.ai-agents-discover-reverse-transcriptases | Autonomous AI Agents Discover Reverse Transcriptases with Tandem Repeat Arrays | Anthropic | 34 | A · 名单内(Anthropic) | reports/Anthropic/ART-Discovery |
| 2026-09-25 | 2609.26550 | JEV-as-a-Judge: Accept When Confident, Escalate When Unsure | Carnegie Mellon University | 34 | A · 名单内(CMU) | reports/CMU/JEV-as-a-Judge |
| 2026-09-25 | 2609.26457 | Recursive self-improvement of AI research agents | Weco AI | 30 | D · 已在库 | reports/Weco/AIDE2 |
| 2026-09-25 | 2609.24919 | PixelDiT2: Representation-Grounded Pixel Diffusion Transformers | NVIDIA、University of Rochester | 30 | A · 名单内(NVIDIA) | reports/NVIDIA/PixelDiT2 |
| 2026-09-26 | 2609.24170 | An Unexpected Robot Policy: Early Evaluations of GPT-6 Astra on RoboDojo and Beyond | RoboProbe、RoboDojo、University of Hong Kong、Tsinghua University | 31 | B · 名单外 | readings/世界模型与 Agent/LLM-as-Policy |
| 2026-09-27 | 2609.30063 | Self-Play Pretraining with Zero Data | Tel Aviv University、Stanford University、LAPTh、USMB | 49 | B · 名单外 | readings/训练方法与强化学习/Self-Play-Pretraining |
| 2026-09-27 | 2609.27656 | InternW0: A Foundational Physical World Model for Efficient Real-World Interactions | Shanghai Artificial Intelligence Laboratory | 38 | A · 名单内(ShanghaiAILab) | reports/ShanghaiAILab/InternW0 |
| 2026-09-27 | 2609.26781 | Agensh: Scaling Organizational Intelligence to 1,024 Agents | Microsoft Research | 32 | A · 名单内(Microsoft) | reports/Microsoft/Agensh |
| 2026-09-28 | 2609.agents-covert-communication-test-time | Despite Instructions: Frontier Agents Improvise Covert Channels at Test Time | Arizona State University、Cornell University、University of California, Davis、University of Pennsylvania | 36 | B · 名单外 | readings/可解释性与对齐/Test-Time-Covert-Channels |
| 2026-09-28 | 2609.28258 | Generalizable Robotic Insertion with World Models | NVIDIA、University of California, San Diego、University of Southern California | 30 | A · 名单内(NVIDIA) | reports/NVIDIA/InsertionWM |
| 2026-09-29 | 2609.30266 | LLM Agents Can Easily Tamper With Their Own Traces | ELLIS Institute Tübingen、Max Planck Institute for Intelligent Systems、Tübingen AI Center、Exponential Security Labs | 41 | A · 名单内(MaxPlanck) | reports/MaxPlanck/Agent-Trace-Tampering |
| 2026-09-29 | 2609.25518 | Matryoshka attribution: Learning to attribute language model outputs to representations and weights | Stanford University | 40 | A · 名单内(Stanford) | reports/Stanford/Matryoshka-Attribution |
| 2026-09-29 | 2609.29171 | Representation World Model: Learning States, Transition and Executable Plans in Representation | Tsinghua University | 40 | A · 名单内(Tsinghua) | reports/Tsinghua/Representation-World-Model |
| 2026-09-29 | 2609.26891 | Harness as a Language: A Minimalist Agent Framework With Maximal Expressivity | MIT Computer Science and Artificial Intelligence Laboratory | 37 | A · 名单内(MIT) | reports/MIT/JAZ |
| 2026-09-29 | 2609.25611 | Qwen3.8-Omni: Towards Native Omni-Modal Agents | Qwen | 37 | A · 名单内(Alibaba) | reports/Alibaba/Qwen3.8-Omni |
| 2026-09-29 | 2609.25627 | MachEmbodied-U0: Unified Understanding and Generation Model for Embodied Intelligence | Li Auto Inc. | 34 | A · 名单内(LiAuto) | reports/LiAuto/MachEmbodied-U0 |
| 2026-09-29 | 2609.26796 | Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs | Mohamed bin Zayed University of Artificial Intelligence | 33 | A · 名单内(MBZUAI) | reports/MBZUAI/Flash-dLLM |
| 2026-09-29 | 2609.29845 | Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs | (alphaXiv 未标) | 31 | B · 名单外 | readings/深度学习基石/Linear-Superposition |
| 2026-09-29 | 2609.28654 | Training Object Permanence in World Models | University of Southern California、Carnegie Mellon University、University of Michigan、Johns Hopkins University | 31 | B · 名单外 | readings/世界模型与 Agent/WROP |
| 2026-09-29 | 2609.28466 | The Past Frames the Future: Memory for Autoregressive Video Generation | Hong Kong University of Science and Technology、City University of Hong Kong、Fudan University、ZODA | 31 | A · 名单内(HKUST) | reports/HKUST/AR-Video-Memory-Survey |
| 2026-09-29 | 2609.28236 | EmbodiedMemory-Bench: Benchmarking Embodied Memory for Long-Horizon Embodied Tasks | Zhejiang University、Central South University、Institute of Software, Chinese Academy of Sciences | 31 | A · 名单内(ZJU) | reports/ZJU/EmbodiedMemory-Bench |
| 2026-09-29 | 2609.28473 | On the Diffusibility of High-Dimensional Latents | Cornell University、Adobe、Virginia Tech、University of Washington | 31 | A · 名单内(Cornell) | reports/Cornell/High-Dim-Latent-Diffusibility |
| 2026-09-29 | 2609.endpoint-constrained-trajectory-optimization | Guiding End-to-End Driving Models with Endpoint-Constrained Trajectory Optimization | (alphaXiv 未标) | 30 | A · 名单内(Toronto) | reports/Toronto/ECO |
| 2026-09-30 | 2609.31394 | InternW0-$Δ$: A World Action Model Bridging Predictive Dynamics and Actions with 20K+ Hours of Open Data | Shanghai Artificial Intelligence Laboratory | 36 | A · 名单内(ShanghaiAILab) | reports/ShanghaiAILab/InternW0-Delta |
| 2026-09-30 | 2609.30652 | Recursive Self-Improvement via On-Policy Distillation for Reasoning | Meta AI、University of California, Riverside | 34 | A · 名单内(Meta) | reports/Meta/Recursive-OPSD |
| 2026-09-30 | 2609.27308 | EmbodiedSWE: Coding Agents for Long Horizon Dexterous Robotics | Bytedance Seed、Yale University、Princeton University、Carnegie Mellon University | 33 | A · 名单内(ByteDance) | reports/ByteDance/EmbodiedSWE |
| 2026-09-30 | 2609.30222 | TrackEverything: Long Horizon Dense Tracking via De-Duplicating 3D Scene Representations | Carnegie Mellon University、Meta | 32 | A · 名单内(CMU) | reports/CMU/TrackEverything |
| 2026-09-30 | 2609.30247 | Rolling-WAM: World Action Models with Rolling Imagination | University of Southern California、Brown University、Fudan University、Toyota Research Institute | 32 | B · 名单外 | readings/世界模型与 Agent/Rolling-WAM |
| 2026-09-30 | 2609.28145 | RL Starts before RL: On Policy Distillation for Better Reinforcement Learning | Fudan University、SII、Jingdong、Peking University | 31 | A · 名单内(Fudan) | reports/Fudan/OPD-before-RL |
| 2026-09-30 | 2609.31093 | Block Sparse Attention with Log-Linear Complexity | Shanghai Jiao Tong University、Bytedance Seed | 30 | A · 名单内(SJTU) | reports/SJTU/PISA |
| 2026-09-30 | 2609.30221 | WanPE: Towards Cinematic Prompt Enhancement for Modern Text-to-Video Generation | Nanjing University、Alibaba Group、University of Science and Technology of China、Fudan University | 30 | A · 名单内(Alibaba) | reports/Alibaba/WanPE |
| 2026-10-01 | 2609.33439 | Raven: The Harness of Harnesses for Composable Agentic Intelligence | EverMind AI | 121 | B · 名单外 | readings/自进化系统/Raven |
| 2026-10-01 | 2609.32701 | Despite Instructions: Frontier Agents Improvise Covert Channels at Test Time | Arizona State University、Cornell University、University of California, Davis、University of Pennsylvania | 54 | D · 已在库 | readings/可解释性与对齐/Test-Time-Covert-Channels |
| 2026-10-01 | 2609.35690 | Agent Priors-guided Policy Learning | National University of Singapore | 47 | A · 名单内(NUS) | reports/NUS/APPL |
| 2026-10-01 | 2609.37053 | MatToolBench: Benchmarking Multimodal Agents in Real-World Materials Science Workflows | Shanghai Jiao Tong University、Suzhou Laboratory、Shanghai Artificial Intelligence Laboratory、BIGAI | 43 | A · 名单内(SJTU) | reports/SJTU/MatToolBench |
| 2026-10-01 | 2609.33150 | Generalization Dynamics of LM Pre-training | University of California, Berkeley、Stanford University、Google DeepMind | 41 | A · 名单内(Berkeley) | reports/Berkeley/LM-Generalization-Dynamics |
| 2026-10-01 | 2609.29429 | Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures | Griffith University、Nanyang Technological University、UNSW、Deakin University | 37 | B · 名单外 | readings/可解释性与对齐/Just-Ask-Jev |
| 2026-10-01 | 2609.35553 | Simplex Diffusion Models | Google DeepMind、École Polytechnique Fédérale de Lausanne、UCL Gatsby | 36 | A · 名单内(Google) | reports/Google/Simplex-Diffusion |
| 2026-10-01 | 2609.29421 | Rufus-Air: An Open LLM Post-Training Recipe | Amazon | 34 | A · 名单内(Amazon) | reports/Amazon/Rufus-Air |
| 2026-10-01 | 2609.29812 | FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates | ELLIS Institute Tübingen、Max Planck Institute for Intelligent Systems、Tübingen AI Center | 33 | A · 名单内(MaxPlanck) | reports/MaxPlanck/FlashLoop |
| 2026-10-01 | 2609.29769 | JEV vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places | University of Pennsylvania | 33 | B · 名单外 | readings/评测与 Benchmark/JEV-vs-LLM-Judges |
| 2026-10-01 | 2609.35318 | DexAgent: An Agentic Human2Sim2Robot Framework for Dexterous Manipulation with Self-Evolving Tool Library | Stanford University、Columbia University | 32 | A · 名单内(Stanford) | reports/Stanford/DexAgent |
| 2026-10-01 | 2609.30249 | RAPID: Robot Agentic Programming from Demonstrations | Massachusetts Institute of Technology、National University of Singapore、University of Pennsylvania、NVIDIA | 32 | A · 名单内(MIT) | reports/MIT/RAPID |
| 2026-10-01 | 2609.tokenization-survey-modern-nlp | Tokenization: A Survey for Modern NLP | (alphaXiv 未标) | 31 | A · 名单内(Google) | reports/Google/Tokenization-Survey |
| 2026-10-01 | 2609.30027 | Synthetic Hospital: An Open, Verifiable, Physician-Validated Longitudinal EHR Benchmark | Carnegie Mellon University | 31 | A · 名单内(CMU) | reports/CMU/Synthetic-Hospital |
| 2026-10-01 | 2609.34085 | AD-E2E-JEPA: A Joint-Embedding Predictive Architecture For End-to-End Autonomous Driving | New York University、Advanced Machine Intelligence Labs | 30 | A · 名单内(NYU) | reports/NYU/AD-E2E-JEPA |
| 2026-10-02 | 2609.37725 | Context Language Models | University of Washington、Meta Superintelligence Labs、Massachusetts Institute of Technology、Trillium Labs | 66 | A · 名单内(Washington) | reports/Washington/Context-Language-Models |
| 2026-10-02 | 2609.35738 | Harness Learning Enables Generalizable Test-Time Adaptation | Carnegie Mellon University、Johns Hopkins University、Stanford University | 35 | A · 名单内(JohnsHopkins) | reports/JohnsHopkins/Harness-Learning |
| 2026-10-02 | 2609.31947 | On-Policy Attention Linearization | Carnegie Mellon University、Cornell University | 35 | A · 名单内(CMU) | reports/CMU/OPAL |
| 2026-10-02 | 2609.35718 | Hard Vision, Easy Vision: What GPT-6 Astra Reveals Across Computer Vision | Mohamed bin Zayed University of Artificial Intelligence、Apertix | 34 | A · 名单内(MBZUAI) | reports/MBZUAI/Frontier-Vision |
| 2026-10-02 | 2609.31620 | FuseReg: Regularizing Layer Fusion Mitigates the Reconstruction-Generation Gap in Representation Autoencoders | USC PSI Lab、Brown University、Rice University、University of Aberdeen | 33 | B · 名单外 | readings/图像、视频与 3D 生成/FuseReg |
| 2026-10-02 | 2609.32457 | Write Back the $Δ$: Revisiting the Same Tokens with Fresh Representations | Tongji University、Shanghai Jiao Tong University、Shanghai Innovation Institute | 31 | B · 名单外 | readings/推理服务与架构探索/ReFlux |
| 2026-10-02 | 2609.35748 | Improving Test-Time Scaling with Adaptive Looped Transformers | Tsinghua University、Yale University | 30 | A · 名单内(Tsinghua) | reports/Tsinghua/TaH2 |
