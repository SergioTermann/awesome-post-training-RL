# Awesome Post-Training RL for LLMs

A curated list of papers on **reinforcement learning post-training** for large language models — from the foundations of RLHF, through preference optimization (DPO family) and RL training frameworks (GRPO family), to the reasoning-RL milestones and frontier-model technical reports.

> 本清单整理了 35 篇 LLM 后训练（post-training）强化学习论文，按主题分类，附 arXiv 链接、完整作者与会议信息。目录结构与本仓库的论文文件夹一一对应。

---

## Contents

1. [Foundations of RLHF（RLHF 奠基）](#1-foundations-of-rlhfrlhf-奠基)
2. [Preference Optimization · DPO Family（偏好优化 · DPO 系）](#2-preference-optimization--dpo-family偏好优化--dpo-系)
3. [RL Training Frameworks · GRPO Family（RL 训练框架 · GRPO 系）](#3-rl-training-frameworks--grpo-familyrl-训练框架--grpo-系)
4. [Reasoning RL Milestones（推理 RL 里程碑）](#4-reasoning-rl-milestones推理-rl-里程碑)
5. [Frontier Model Technical Reports（前沿模型技术报告）](#5-frontier-model-technical-reports前沿模型技术报告)
6. [Accepted at Top Conferences（已录用顶会）](#6-accepted-at-top-conferences已录用顶会)

---

## 1. Foundations of RLHF（RLHF 奠基）

The canonical recipe — supervised fine-tuning, reward modeling, and RL — that started it all.

- **[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)** — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe. *NeurIPS 2022*. The seminal **InstructGPT** work introducing the RLHF pipeline: SFT → reward modeling → PPO to align GPT-3 with human instructions.
- **[Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)** — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, Jared Kaplan. *2022*. Trains a harmless assistant via **RLAIF** — AI feedback governed by a written "constitution" of principles instead of human labels.
- **[RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)** — Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, Sushant Prakash. *ICML 2024*. Shows AI-generated preferences can rival human preferences for RL, making alignment scalable beyond human annotation.

## 2. Preference Optimization · DPO Family（偏好优化 · DPO 系）

Direct / closed-form alignment objectives that sidestep the reward model + PPO machinery.

- **[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)** — Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. *NeurIPS 2023*. **DPO** replaces reward-model + PPO with a single closed-form objective over preference pairs.
- **[Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335)** — Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, Quanquan Gu. *ICML 2024*. **SPIN** iteratively refines a model using its own generated responses as the losing half of preference pairs.
- **[Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)** — Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, Jason Weston. *2024*. Trains LMs to act as their own judge (**LLM-as-a-Judge**), iteratively improving both instruction following and reward modeling.
- **[KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)** — Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela. *ICML 2024*. Aligns from binary good/bad signals via prospect theory rather than paired preferences, cutting data requirements.
- **[ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)** — Jiwoo Hong, Noah Lee, James Thorne. *ACL 2024*. A reference-free objective that fuses SFT and preference alignment into one training stage.
- **[SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)** — Yu Meng, Mengzhou Xia, Danqi Chen. *NeurIPS 2024*. Reference-free reward via average log-probability, plus length normalization and a target reward margin.
- **[Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704)** — NVIDIA. *2024*. Open 340B base/instruct/reward models; demonstrates HelpSteer2 reward modeling and iterative DPO for alignment.
- **[GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization](https://arxiv.org/abs/2601.05242)** — Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, Pavlo Molchanov. *2026*. Decouples multi-reward signals per group and normalizes them before optimization to stabilize multi-objective RL.

## 3. RL Training Frameworks · GRPO Family（RL 训练框架 · GRPO 系）

Critic-free policy-gradient methods, training systems, and efficiency/stability tricks built around GRPO and its relatives.

- **[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)** — Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo. *2024*. Introduces **GRPO** (Group Relative Policy Optimization) for math reasoning, removing the critic model.
- **[Teaching Large Language Models to Reason with Reinforcement Learning](https://arxiv.org/abs/2403.04642)** — Alex Havrilla, Yuqing Du, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, Sainbayar Sukhbaatar, Roberta Raileanu. *2024*. "Back to Basics" — highlights **RLOO** as a simple low-variance alternative to PPO, and explores RL without a cold-start SFT stage.
- **[Tulu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)** — Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, Hannaneh Hajishirzi. *2024*. A fully open post-training recipe (SFT → DPO → **RLVR**) with infrastructure, data, and evaluation for open models.
- **[REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization](https://arxiv.org/abs/2501.03262)** — Jian Hu, Jason Klein Liu, Haotian Xu, Wei Shen. *2025*. Stabilizes critic-free policy optimization via global advantage normalization, improving GRPO-style training.
- **[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)** — Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, Mingxuan Wang. *2025*. Open large-scale RL system with decoupled clip-higher and dynamic sampling to fix GRPO's entropy-collapse issues.
- **[GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning](https://arxiv.org/abs/2506.16141)** — Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Junhao Cheng, Ying Shan, Xihui Liu. *2025*. Adds a consistency-aware objective to GRPO for multimodal reasoning, mitigating hallucination.
- **[STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens](https://arxiv.org/abs/2602.15620)** — Shiqi Liu, Zeyu He, Guojian Zhan, Letian Tao, Zhilong Zheng, Jiang Wu, Yinuo Wang, Yang Guan, Kehua Sheng, Bo Zhang, Keqiang Li, Jingliang Duan, Shengbo Eben Li. *2026*. Stabilizes RL by masking rare, high-variance spurious tokens during policy updates.
- **[Do Post-Training Algorithms Actually Differ? A Controlled Study Across Model Scales Uncovers Scale-Dependent Ranking Inversions](https://arxiv.org/abs/2603.19335)** — Xiaoyi Li. *2026*. Controlled study showing post-training algorithm rankings can invert across model scales.
- **[Efficient RL Training for LLMs with Experience Replay](https://arxiv.org/abs/2604.08706)** — Charles Arnal, Vivien Cabannes, Taco Cohen, Julia Kempe, Remi Munos. *2026*. Brings experience replay to LLM RL for sample-efficient reuse of past rollouts.
- **[DACA-GRPO: Denoising-Aware Credit Assignment for Reinforcement Learning in Diffusion Language Models](https://arxiv.org/abs/2605.16342)** — Amin Karimi Monsefi, Dominic Culver, Nikhil Bhendawade, Lokesh Boominathan, Manuel R. Ciosici, Yizhe Zhang, Irina Belousova. *2026*. Denoising-aware credit assignment for GRPO in diffusion language models.
- **[PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play](https://arxiv.org/abs/2605.16727)** — Roger Creus Castanyer, Geoffrey Bradway, Lorenz Wolf, Maxwill Lin, Augustine N. Mavor-Parker, Matthew James Sargent. *2026*. Co-evolves a population of LoRA modules through self-play for reasoning.
- **[How Off-Policy Can GRPO Be? Mu-GRPO for Efficient LLM Reinforcement Learning](https://arxiv.org/abs/2605.17570)** — Minghao Tian, Yunfei Xie, Chen Wei. *2026*. Investigates how far GRPO can go off-policy; reuses rollouts across multiple updates for efficiency.
- **[Spend Your Rollouts Where It Counts: Rollout Allocation for Group-Based RL Post-Training](https://arxiv.org/abs/2605.26606)** — Woojeong Kim, Ziyi Yang, Jing Nathan Yan, Jialu Liu. *2026*. Dynamically allocates the rollout budget toward where it improves group-based RL the most.
- **[Distilled Reinforcement Learning for LLM Post-training](https://arxiv.org/abs/2607.17247)** — Chen Wang, Zhaochun Li, Jionghao Bai, Yining Zhang, Hexuan Deng, Ge Lan, Yue Wang. *2026*. Distills RL post-training signals into efficient LLM alignment.

## 4. Reasoning RL Milestones（推理 RL 里程碑）

Landmark models showing RL alone can unlock emergent reasoning.

- **[Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)** — Kimi Team. *2025*. Scales RL (long-CoT, RLVR, curriculum, long2short) to rival frontier reasoning models.
- **[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)** — DeepSeek-AI. *2025*. Pure-RL "aha moment" via **R1-Zero**, then the multi-stage **R1** pipeline with distilled open models.
- **[Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model](https://arxiv.org/abs/2503.24290)** — Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, Heung-Yeung Shum. *2025*. Open-source reproduction of R1-Zero-style base-model RL scaling, achieving strong reasoning.

## 5. Frontier Model Technical Reports（前沿模型技术报告）

Recent frontier-model reports describing post-training at scale.

- **[GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)** — GLM-5-Team (Zhipu AI & Tsinghua). *2026*. Frontier agentic LLM emphasizing agentic engineering and tool-use post-training.
- **[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)** — DeepSeek-AI. *2026*. Frontier report on a highly efficient million-token-context model.
- **[Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)** — Kimi Team. *2026*. Open frontier intelligence model report.

## 6. Accepted at Top Conferences（已录用顶会）

Papers accepted at top-tier venues.

- **[Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning](https://arxiv.org/abs/2509.25300)** — Zelin Tan, Hejia Geng, Xiaohang Yu, Mulei Zhang, Guancheng Wan, Yifan Zhou, Qiang He, Xiangyuan Xue, Heng Zhou, Yutao Fan, Zhongzhi Li, Zaibin Zhang, Guibin Zhang, Chen Zhang, Zhenfei Yin, Philip Torr, Lei Bai. *ACL 2026 Main Conference*. Empirical scaling study of RL post-training for mathematical reasoning.
- **[Nudging the Boundaries of LLM Reasoning](https://arxiv.org/abs/2509.25666)** — Justin Chih-Yao Chen, Becky Xiangyu Peng, Prafulla Kumar Choubey, Kung-Hsiang Huang, Jiaxin Zhang, Mohit Bansal, Chien-Sheng Wu. *ICLR 2026*. **NuRL** — nudges LLM reasoning boundaries via nudging-style RL interventions.
- **[Training Large Reasoning Models Efficiently via Progressive Thought Encoding](https://arxiv.org/abs/2602.16839)** — Zeliang Zhang, Xiaodong Liu, Hao Cheng, Hao Sun, Chenliang Xu, Jianfeng Gao. *ICLR 2026*. Progressive thought encoding for efficient training of large reasoning models.
- **[Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models](https://arxiv.org/abs/2604.25011)** — Dan Shi, Zhuowen Han, Simon Ostermann, Renren Jin, Josef van Genabith, Deyi Xiong. *ACL 2026 Main Conference*. Feature-level mechanistic study of why RL generalizes in LLM post-training.

---

## Notes

- All links point to arXiv abstracts; IDs follow the `YYMM.NNNNN` scheme.
- **Venue** is shown where the paper is peer-reviewed at a top venue (NeurIPS / ICML / ICLR / ACL); unmarked entries are arXiv preprints / technical reports.
- Institutional mega-collaborations are credited to the team name (DeepSeek-AI, Kimi Team, NVIDIA, GLM-5-Team) as on the papers themselves.
- Corresponding PDFs are stored in the numbered subfolders of this repository (folder names match the section titles).
- Feel free to open an issue or PR to add missing papers.
