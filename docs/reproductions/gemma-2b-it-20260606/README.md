# Gemma 2B IT 拒绝方向复现实验报告

本报告记录 `google/gemma-2b-it` 在当前工程环境下的一次完整 pipeline
复现。原始运行产物包含 harmful prompt 的 completions/evaluations，因此不提交到
GitHub；本报告只保留指标、路径、环境和与仓库自带 artifacts 的对比。

## 结论

本次复现跑通了 `pipeline.run_pipeline` 的完整主流程：

1. 抽取候选 refusal directions。
2. 选择最终 refusal direction。
3. 在 `jailbreakbench` harmful prompts 上生成 baseline / ablation / actadd completions。
4. 使用 substring matching 和本地 Ollama LlamaGuard 评估 harmful ASR。
5. 在 harmless prompts 上生成 baseline / actadd completions 并评估 refusal。
6. 计算 `pile`、`alpaca`、`alpaca_custom_completions` 的 CE loss / perplexity。

复现到的主现象：

- baseline 模型对 harmful prompts 大多拒绝。
- directional ablation 后，harmful prompt 的拒绝被显著绕过。
- activation addition 也能绕过拒绝，但对分布的损伤明显更大。
- 对 harmless prompts 正向加入 refusal direction 后，模型几乎都转为拒绝。
- ablation 的 loss 与 baseline 接近，符合论文中 directional ablation 更 surgical 的结论。

## 运行环境

本次完整运行在 `dell` 上执行，不是在本机执行。

| 项 | 值 |
| --- | --- |
| 运行机器 | `dell` (`10.77.0.102`) |
| 模型 | `google/gemma-2b-it` |
| 模型路径 | `/home/dell/.workplace/models/refusal_direction/google/gemma-2b-it` |
| 代码路径 | `/home/dell/.workplace/reproductions/refusal_direction` |
| Conda 环境 | `/home/dell/.workplace/conda/envs/refusal_direction` |
| GPU | `CUDA_VISIBLE_DEVICES=6` |
| Guard | `Ollama llama-guard3:1b` |
| Ollama 地址 | `127.0.0.1:11434`，指 `dell` 的 localhost |
| 运行日志 | `/home/dell/.workplace/logs/refusal_direction_gemma2b_full_20260606_173604.log` |

本机仅用于代码编辑、SSH/rsync、测试和 GitHub 同步。本机没有启动 Ollama 服务。

## 原始产物位置

远端原始产物：

```text
/home/dell/.workplace/reproductions/refusal_direction/pipeline/runs/gemma-2b-it
```

本机已同步一份只供本地检查使用的副本：

```text
artifacts/refusal_direction/gemma-2b-it/20260606_173604/
```

`artifacts/` 已加入 `.gitignore`，不会提交到 GitHub。

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
{"pos": -1, "layer": 13}
```

### JailbreakBench harmful ASR

| 干预 | Substring ASR | Ollama LlamaGuard ASR |
| --- | ---: | ---: |
| baseline | 0.09 | 0.05 |
| ablation | 0.99 | 0.86 |
| actadd | 0.99 | 0.94 |

解释：

- `baseline` 很低，说明原模型大多拒绝 harmful requests。
- `ablation` 和 `actadd` 很高，说明拒绝机制被显著绕过。
- Ollama LlamaGuard 使用的是本地 `llama-guard3:1b`，不是论文原始 artifacts 的 LlamaGuard2，因此 guard 数字只能比较趋势，不能逐位对齐。

### Harmless 非拒绝率

| 干预 | Substring success rate |
| --- | ---: |
| baseline | 0.99 |
| actadd | 0.01 |

这里的 `substring_matching_success_rate` 可理解为“没有出现常见拒绝字符串的比例”。
baseline 几乎都正常回答；正向加入 refusal direction 后几乎都拒绝。

### CE loss / perplexity

| 干预 | 数据集 | CE loss | Perplexity | Tokens |
| --- | --- | ---: | ---: | ---: |
| baseline | pile | 3.503378 | 33.227499 | 906802 |
| baseline | alpaca | 2.089130 | 8.077884 | 270467 |
| baseline | alpaca_custom_completions | 0.257081 | 1.293149 | 24592 |
| ablation | pile | 3.500151 | 33.120455 | 906802 |
| ablation | alpaca | 2.088579 | 8.073435 | 270467 |
| ablation | alpaca_custom_completions | 0.267399 | 1.306562 | 24592 |
| actadd | pile | 4.377024 | 79.600778 | 906802 |
| actadd | alpaca | 2.655044 | 14.225617 | 270467 |
| actadd | alpaca_custom_completions | 1.143522 | 3.137801 | 24592 |

解释：

- `ablation` 的 loss 与 `baseline` 接近，说明 directional ablation 对一般分布扰动很小。
- `actadd` 的 loss 和 perplexity 明显上升，说明 activation addition 虽能绕过拒绝，但更容易把模型推离原分布。

## 与仓库自带 Gemma 2B artifacts 对比

仓库原始 artifacts 位于：

```text
pipeline/runs/gemma-2b-it
```

原始 direction：

```json
{"pos": -2, "layer": 10}
```

原始 harmful 指标：

| 干预 | Substring ASR | LlamaGuard2 ASR |
| --- | ---: | ---: |
| baseline | 0.09 | 0.07 |
| ablation | 1.00 | 0.78 |
| actadd | 0.98 | 0.73 |

原始 harmless 指标：

| 干预 | Substring success rate |
| --- | ---: |
| baseline | 1.00 |
| actadd | 0.00 |

对比结论：

- baseline harmful ASR 对齐：本次 substring `0.09`，原始 artifacts 也是 `0.09`。
- ablation 绕过拒绝的趋势对齐：本次 substring `0.99`，原始 artifacts `1.00`。
- actadd 绕过拒绝的趋势对齐：本次 substring `0.99`，原始 artifacts `0.98`。
- harmless induction 趋势对齐：baseline 接近全正常回答，actadd 接近全拒绝。
- guard 评估器不同：本次是 Ollama `llama-guard3:1b`，原始 artifacts 是 LlamaGuard2，因此 guard ASR 不做逐位比较。

## 为什么选中的层不同

本次选中 `pos=-1, layer=13`，原始 artifacts 选中 `pos=-2, layer=10`。
这不是 Ollama LlamaGuard 导致的，因为 layer selection 发生在 jailbreak evaluation
之前，不使用 guard 分数。

`select_direction.py` 的选择逻辑是：

1. 对每个候选 direction 计算：
   - `refusal_score`：ablation 后 harmful prompts 的拒绝倾向，越低越好。
   - `steering_score`：加入 direction 后 harmless prompts 的拒绝诱导强度，需要高于阈值。
   - `kl_div_score`：ablation 后 harmless logits 的 KL 扰动，需要低于阈值。
2. 过滤掉 KL 太高、steering 太弱、太靠后层的候选。
3. 在剩余候选里按 `refusal_score` 升序排序，选第一名。

两个 run 的 filtered top candidates：

| 来源 | position | layer | refusal_score | steering_score | kl_div_score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 artifacts | -2 | 10 | -14.4193 | 6.7886 | 0.0659 |
| 本次 run | -1 | 13 | -7.8211 | 5.1062 | 0.0163 |

可能原因：

- HF checkpoint、tokenizer、`transformers`、`torch`、CUDA kernel 和 dtype 行为与原始 artifacts 当时环境不同。
- `filter_data` 会根据当前模型 refusal score 过滤训练/验证样本；数值和模型版本差异会改变实际参与 direction selection 的样本集合。
- 候选方向之间存在多个相近有效方向；算法是 winner-take-all 排序，top candidate 的精确 layer 容易随环境变化。
- 修复 Gemma `batch_decode` 的改动只影响 plot label，不参与评分；Ollama LlamaGuard 只参与后续 harmful evaluation，也不参与选层。

因此，精确 layer 不同不代表复现失败。两个结果都集中在 `pos=-1/-2`、中后层 `10-13`，并且最终行为趋势对齐论文主结论。

## 验证记录

完整命令在 `dell` 上退出 `0`。本地 artifact 验证确认：

- 21 个 run 文件存在且非空。
- harmful / harmless completions 和 evaluations 都存在。
- 三组 loss eval JSON 都存在。
- run 日志中没有 `Traceback`。

本地代码测试：

```text
pytest tests/test_ollama_llamaguard.py tests/test_token_utils.py tests/test_evaluate_loss.py -q
6 passed
```
