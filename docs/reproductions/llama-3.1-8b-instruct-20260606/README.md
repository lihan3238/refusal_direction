# Llama 3.1 8B Instruct 拒绝方向复现实验报告

本报告记录 `meta-llama/Llama-3.1-8B-Instruct` 在当前工程环境下的一次完整
pipeline 复现。原始运行产物包含 harmful prompt 的 completions/evaluations，
因此不提交到 GitHub；本报告只保留指标、路径、环境和结论。

## 结论

本次复现跑通了 `pipeline.run_pipeline` 的完整主流程：

1. 抽取候选 refusal directions。
2. 选择最终 refusal direction。
3. 在 `jailbreakbench` harmful prompts 上生成 baseline / ablation / actadd completions。
4. 使用 substring matching 和本地 Ollama LlamaGuard 评估 harmful ASR。
5. 在 harmless prompts 上生成 baseline / actadd completions 并评估 refusal。
6. 计算 `pile`、`alpaca`、`alpaca_custom_completions` 的 CE loss / perplexity。

复现到的主现象：

- baseline 模型对 harmful prompts 基本拒绝。
- directional ablation 后，harmful prompt 的拒绝被显著绕过。
- activation addition 也能绕过拒绝。
- harmless prompts 上，正向加入 refusal direction 会提高拒绝比例，但在 Llama 3.1 8B 上没有 Gemma 2B 那么极端。
- ablation 的 loss 与 baseline 接近；actadd 的 Pile / custom completions loss 上升更明显，符合论文中 directional ablation 更 surgical 的结论。

代码层面，本次选择 `Llama-3.1-8B-Instruct` 是为了命中仓库现有
`Llama3Model` adapter。兼容性检查确认该模型仍是 `LlamaForCausalLM` 风格结构：
32 层、hidden size 4096、`LlamaAttention` / `LlamaMLP` 模块路径与现有 hooks
兼容。因此，本轮没有修改核心 pipeline 代码。

## 运行环境

本次完整运行在 `dell` 上执行，不是在本机执行。

| 项 | 值 |
| --- | --- |
| 运行机器 | `dell` (`10.77.0.102`) |
| 模型 | `meta-llama/Llama-3.1-8B-Instruct` |
| 模型路径 | `/home/dell/.workplace/models/refusal_direction/meta-llama/Llama-3.1-8B-Instruct` |
| 代码路径 | `/home/dell/.workplace/reproductions/refusal_direction` |
| Conda 环境 | `/home/dell/.workplace/conda/envs/refusal_direction` |
| 主模型 GPU | `CUDA_VISIBLE_DEVICES=6` |
| Guard | `Ollama llama-guard3:1b` |
| Ollama 地址 | `127.0.0.1:11434`，指 `dell` 的 localhost |
| 下载日志 | `/home/dell/.workplace/logs/refusal_direction_llama31_8b_download_20260606_194556.log` |
| 正式运行日志 | `/home/dell/.workplace/logs/refusal_direction_llama31_8b_full_retry_online_20260606_202948.log` |

模型下载前已检查远端模型目录；`dell` 和 `hello` 的
`~/.workplace/models/refusal_direction` 下当时都只有 `google/gemma-2b-it`。
`hello` 的根分区只剩约 12G，因此本轮未在 `hello` 下载大模型。

本机仅用于代码编辑、SSH/rsync、验证和 GitHub 同步。本机没有启动 Ollama 服务，
也没有运行大模型。

## 原始产物位置

远端正式产物：

```text
/home/dell/.workplace/reproductions/refusal_direction/pipeline/runs/Llama-3.1-8B-Instruct
```

本机已同步一份只供本地检查使用的副本：

```text
artifacts/refusal_direction/Llama-3.1-8B-Instruct/20260606_202948/
```

`artifacts/` 已加入 `.gitignore`，不会提交到 GitHub。15G 模型权重只保留在
`dell`，没有同步回本机。

另有一次失败 run 被保留在远端：

```text
/home/dell/.workplace/reproductions/refusal_direction/pipeline/runs/Llama-3.1-8B-Instruct.failed_offline_20260606_201712
```

该失败 run 的根因是运行命令设置了 `TRANSFORMERS_OFFLINE=1`，导致
`datasets` 在 CE loss 阶段访问 `monology/pile-uncopyrighted` 时被强制 offline。
正式 run 已去掉该环境变量并完整退出 `0`。

## 产物说明

| 路径 | 含义 |
| --- | --- |
| `generate_directions/mean_diffs.pt` | 所有候选 refusal directions，来自 harmful/harmless activation mean difference。 |
| `direction_metadata.json` | 最终选择的 token position 和 layer。 |
| `direction.pt` | 最终选择的 refusal direction 向量。 |
| `select_direction/direction_evaluations.json` | 所有候选方向的 refusal / steering / KL 分数。 |
| `select_direction/direction_evaluations_filtered.json` | 过滤后的候选方向，按 `refusal_score` 升序排序。 |
| `select_direction/*.png` | 候选方向分数随 layer / token position 的图。 |
| `completions/jailbreakbench_*_completions.json` | harmful prompts 在不同干预下的生成结果。 |
| `completions/jailbreakbench_*_evaluations.json` | harmful completions 的 substring / Ollama LlamaGuard ASR。 |
| `completions/harmless_*_completions.json` | harmless prompts 在不同干预下的生成结果。 |
| `completions/harmless_*_evaluations.json` | harmless completions 的非拒绝/拒绝趋势评估。 |
| `loss_evals/*_loss_eval.json` | 各干预下的 CE loss / perplexity / token 数。 |

## 本次复现指标

最终选择的 direction：

```json
{"pos": -2, "layer": 11}
```

### JailbreakBench harmful ASR

| 干预 | Substring ASR | Ollama LlamaGuard ASR |
| --- | ---: | ---: |
| baseline | 0.16 | 0.01 |
| ablation | 1.00 | 0.86 |
| actadd | 1.00 | 0.89 |

解释：

- `baseline` 的 ASR 很低，说明原模型大多拒绝 harmful requests。
- `ablation` 和 `actadd` 的 ASR 都接近 1，说明拒绝机制被显著绕过。
- Ollama LlamaGuard 使用的是本地 `llama-guard3:1b`，不是论文原始 artifacts 的 LlamaGuard2，因此 guard 数字只能比较趋势，不能逐位对齐。

### Harmless 非拒绝率

| 干预 | Substring success rate |
| --- | ---: |
| baseline | 1.00 |
| actadd | 0.83 |

这里的 `substring_matching_success_rate` 可理解为“没有出现常见拒绝字符串的比例”。
baseline 全部正常回答；正向加入 refusal direction 后，非拒绝率从 `1.00` 降到
`0.83`，说明拒绝倾向被诱导出来，但强度弱于 Gemma 2B run 中接近全拒绝的表现。

### CE loss / perplexity

| 干预 | 数据集 | CE loss | Perplexity | Tokens |
| --- | --- | ---: | ---: | ---: |
| baseline | pile | 2.161282 | 8.682265 | 900864 |
| baseline | alpaca | 1.793307 | 6.009292 | 267216 |
| baseline | alpaca_custom_completions | 0.212989 | 1.237371 | 34946 |
| ablation | pile | 2.168510 | 8.745248 | 900864 |
| ablation | alpaca | 1.813329 | 6.130823 | 267216 |
| ablation | alpaca_custom_completions | 0.226281 | 1.253928 | 34946 |
| actadd | pile | 2.316140 | 10.136476 | 900864 |
| actadd | alpaca | 1.783149 | 5.948561 | 267216 |
| actadd | alpaca_custom_completions | 0.462099 | 1.587402 | 34946 |

解释：

- `ablation` 的 Pile loss 只比 baseline 高约 `0.0072`，Alpaca / custom loss 也只小幅变化，说明 directional ablation 对一般分布扰动较小。
- `actadd` 的 Pile loss 从 `2.1613` 升到 `2.3161`，custom completions loss 从 `0.2130` 升到 `0.4621`，说明 activation addition 对输出分布扰动更大。
- `actadd` 的 Alpaca loss 低于 baseline，这说明该单项数据集上并非所有 actadd loss 都上升；更稳妥的结论是 Pile 和 custom completions 明显变差，而 ablation 全部接近 baseline。

## 与 Gemma 2B run 和论文趋势对比

本轮模型不是论文 README 示例里的 `meta-llama/Meta-Llama-3-8B-Instruct`，
而是更新的 `meta-llama/Llama-3.1-8B-Instruct`。因此这里不做逐项复刻原论文数值，
只比较机制趋势。

与我们已完成的 `google/gemma-2b-it` run 对比：

| 模型 | Direction | baseline harmful substring | ablation harmful substring | actadd harmful substring | harmless baseline non-refusal | harmless actadd non-refusal |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `google/gemma-2b-it` | `pos=-1, layer=13` | 0.09 | 0.99 | 0.99 | 0.99 | 0.01 |
| `Llama-3.1-8B-Instruct` | `pos=-2, layer=11` | 0.16 | 1.00 | 1.00 | 1.00 | 0.83 |

对比结论：

- 两个模型都复现了论文主结论：单一 refusal direction 的 ablation 能显著移除拒绝行为。
- Llama 3.1 8B 的 baseline harmful ASR 比 Gemma 2B 略高，但仍然低；ablation / actadd 后都达到 `1.00`。
- Llama 3.1 8B 的正向 actadd 在 harmless 上诱导拒绝的强度较弱，没有像 Gemma 2B 那样把非拒绝率压到接近 0。
- Llama 3.1 8B 的 ablation loss 与 baseline 更接近，actadd 在 Pile/custom 上有更明显扰动，仍支持“ablation 更 surgical”的论文趋势。

## 为什么选择这个模型

本轮目标是“找个大点、新点的模型，尽量不改代码”。仓库当前
`model_factory.py` 支持的家族包括 `qwen`、`llama-3`、`llama`、`gemma`、`yi`。
为了避免新增 adapter，最稳妥的选择是现有 `Llama3Model` 可以直接覆盖的模型。

选择 `meta-llama/Llama-3.1-8B-Instruct` 的原因：

- 比上轮 `google/gemma-2b-it` 大很多，参数量从 2B 到 8B。
- 比仓库 README 示例的 Llama 3 8B 更新。
- 路径包含 `llama-3`，会命中现有 `Llama3Model`。
- 兼容性检查确认 `model.model.layers[*].self_attn` 和 `model.model.layers[*].mlp` 路径存在，现有 hooks 可用。
- 8B bf16 能在 32GB 5090 上跑完整 pipeline。

没有优先选 Qwen2.5/Qwen3 的原因：

- 仓库里的 `QwenModel` 更像旧 Qwen 结构，假设 `model.transformer.h`、`block.attn`、`tokenizer.eod_id` 等路径。
- Qwen2.5/Qwen3 通常是 `model.model.layers`、`self_attn`、`mlp` 等新结构；直接跑大概率需要新增 adapter。
- 如果下一步目标是“更新优先，允许改代码”，Qwen2.5/Qwen3 可以作为下一轮，但它不是“不改代码”的最稳选择。

## 验证记录

正式完整命令在 `dell` 上退出 `0`。本地 artifact 验证确认：

- 21 个 run 文件存在。
- `direction_metadata.json`、所有 evaluation JSON、三组 loss eval JSON 都能解析。
- harmful / harmless completions 数量均为 100。
- run 日志中正式 run 没有 `Traceback`。
- 本机已同步正式 run 和下载/运行日志到 `artifacts/refusal_direction/Llama-3.1-8B-Instruct/20260606_202948/`。

关键验证输出：

```text
direction {'pos': -2, 'layer': 11}
jbb baseline n 100 substring 0.16 ollama 0.01
jbb ablation n 100 substring 1.0 ollama 0.86
jbb actadd n 100 substring 1.0 ollama 0.89
harmless baseline n 100 substring_non_refusal 1.0
harmless actadd n 100 substring_non_refusal 0.83
missing []
```
