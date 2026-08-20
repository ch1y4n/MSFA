# Layer 5 · Agent-Level Static Scan

STI 论文 **Layer 5** 的静态部分：在 **875 个开源 Agent 项目**上做 special-token 静态扫描，
量化“Agent 应用是否在不可信内容进入模型前，清理/隔离目标模型的 special token / chat-template marker”。

本目录只保留复现所需的三样东西：**875 个项目清单、批量克隆脚本、静态扫描脚本（及规则）**。
仓库克隆和扫描结果不随 artifact 提供，按下面步骤即可复现。

## 内容

| 路径 | 说明 |
|---|---|
| `agent_projects_875.csv` | 被扫描的 875 个 Agent 项目清单（`name` / `url` / `source`） |
| `scripts/00_clone_agent_repos.py` | 批量克隆脚本：读清单 → 并发 `git clone` 到 `repo/<owner>__<repo>` |
| `scripts/01_load_rules_scan_files.py` | 静态扫描器：加载 YAML 规则 → ripgrep 并发扫描 `repo/` → 汇总命中 |
| `rules/*.yaml` | 论文静态扫描使用的 special-token 规则：`qwen / llama / mistral / gemma / deepseek / universal` |
| `rules_optional/glm.yaml` | 补充的 GLM 规则，不计入论文报告的五模型族扫描口径 |

## 使用方式

```bash
# 1) 批量克隆 875 个 Agent 项目到 static/repo/（浅克隆、并发、可断点续跑）
py scripts/00_clone_agent_repos.py            # 默认读取 agent_projects_875.csv，8 并发
#   可选：--workers 16 --timeout 600 --full

# 2) 用规则做 special-token 静态扫描
py scripts/01_load_rules_scan_files.py        # 扫描 repo/，结果写 scan_results/
```

> 需要 `git` 与 Python 3（Windows 下用 `py`）。

## 875 清单的构成

```
655  manual_checklist_x            (README 分类后人工勾选)
167  readme_recovered_agent        (人工复核恢复的 Agent)
 49  uncertain_audit               (30 yes + 19 boundary_yes)
  4  disclosure_supplement         (void / semantic-kernel / aider / AutoGPT，漏洞披露实测补充)
= 875
```

论文口径：star ≥ 1,000 检索得到约 1,917 个仓库，排除非 Agent 后得到 **875** 个 Agent 项目；
人工审计后其中 **30** 个（3.4%）有 special-token 输入路径的防御处理。

论文所述五个模型族共提供 **205** 个 detection items，精确去重后为 **202** 个签名。
该统计包含 `rules/` 中的五个模型族规则及共享的 `universal.yaml`，不包含补充的 GLM 规则。
