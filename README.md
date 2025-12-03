# DailyArxiv

一个自动抓取arXiv每日论文并生成中文解读的工具。

## 功能特性

- 📚 自动抓取多个arXiv类别的每日论文
- 🌐 支持HTML解析和API两种获取方式
- 🤖 集成DeepSeek LLM解读功能（可选）
- 📝 生成包含中文翻译和术语解释的Markdown文件
- 🗂️ 按类别分类保存论文
- 🧠 先进的聚类模块：强预处理、OpenAI/HF嵌入、UMAP降低维度、HDBSCAN+噪声再聚类、LLM自动总结
- 🧾 输出 JSON 聚类报告包含主题关键词、代表论文与剩余噪声信息

## 安装

1. 克隆项目：
```bash
git clone <repository-url>
cd DailyArxiv
```

2. 安装依赖：
```bash
uv sync
```

3. 配置DeepSeek API密钥（如需使用LLM功能）：
```bash
export DEEPSEEK_API_KEY=你的DeepSeek_API密钥
```

## 使用方法

### 主流程（`main.py`）
```bash
uv run main.py --date 2025-12-02
```
此脚本串联 `fetch_arxiv.py` → `cluster_topics.py` → `summarize_clusters.py` → `analyze_trends.py` → `generate_report.py`。关键参数示例：

| 参数 | 说明 |
|------|------|
| `--date` | 目标日期（默认当天）。 |
| `--categories` | 传递给抓取阶段的类别列表。 |
| `--run-suffix` | 追加到自动生成文件名的后缀（如 `hep-ph`）。 |
| `--skip-fetch/--skip-cluster/--skip-summary/--skip-trend/--skip-report` | 跳过某个阶段。 |
| `--embed-*` | 传递给聚类阶段的参数（`--embed-backend`、`--embed-model`、`--embed-batch-size`、`--embed-device` 等）。 |
| `--summary-*` | 控制聚类摘要的 LLM 设置（`--summary-model`、`--summary-max-papers` 等）。 |

### 单独运行抓取（`fetch_arxiv.py`）
```bash
uv run fetch_arxiv.py --date 2025-12-02 --categories cs.CL stat.ML --max-results 1500
```
输出默认写入 `data/arxiv_<date>.json`；可通过 `--save-path`、`--suffix` 自定义。常见选项：

| 参数 | 说明 |
|------|------|
| `--date` | 支持 `today`/`yesterday` 或 `YYYY-MM-DD`。 |
| `--categories` | 一组 arXiv 类别；默认覆盖全部列表。 |
| `--max-results` | 限制总记录数。 |
| `--save-path` | 指定输出文件或目录。 |
| `--retries` / `--backoff` | 网络重试次数与退避策略。 |
| `--suffix` | 自动在文件名中插入后缀（如 `cs-cluster`）。 |

### 单独运行聚类（`cluster_topics.py`）
```bash
uv run cluster_topics.py --date 2025-12-02 --backend openai --stopword-filter --summarizer-model gpt-4o-mini
```
该命令读取 `data/arxiv_<date>.json`，经过预处理、嵌入缓存、UMAP + HDBSCAN（含噪声救援）、LLM摘要，最终生成 `data/clusters_<date>.json`，JSON 中包含 `clusters`/`metadata`/`noise` 的完整结构。

| 参数 | 说明 |
|------|------|
| `--date` | 指定聚类数据的日期。 |
| `--backend` / `--model` | 选择 OpenAI (`text-embedding-3-large`) 或 HuggingFace fallback。 |
| `--batch-size` | OpenAI 批量请求大小。 |
| `--device` | HuggingFace 设备（`cpu`/`cuda`/`auto`）。 |
| `--cache-dir` | SQLite 缓存目录（默认为 `.cache/embeddings.db`）。 |
| `--stopword-filter` | 去除简单停用词。 |
| `--summarizer-model` | LLM 模型，默认 `gpt-4o-mini`。 |
| `--suffix` | 用于区分输出文件名（`clusters_<suffix>_<date>.json`）。 |

## 项目结构

```
DailyArxiv/
├── main.py               # 全流程编排：fetch → cluster → summarize → trends → report
├── fetch_arxiv.py        # Phase 1：抓取 arXiv 数据
├── cluster_topics.py     # Phase 2：预处理、嵌入、UMAP+HDBSCAN、噪声救援与 LLM 聚类总结
├── preprocess.py         # 预处理文本（去 LaTeX、停用词等）
├── embed.py              # 嵌入实现（OpenAI + HF + 缓存）
├── cluster.py            # 聚类/UMAP/HDBSCAN 核心逻辑
├── summarize.py          # LLM 聚类摘要
├── summarize_clusters.py # Phase 3：汇总聚类成 Markdown/JSON 报告
├── analyze_trends.py     # Phase 4：趋势分析
├── generate_report.py    # Phase 5：生成日报/邮件内容
├── data/                 # 抓取与聚类数据（arxiv_xxx.json, clusters_xxx.json）
├── reports/              # 输出的 Markdown 报告
├── templates/            # 报告/摘要模版
├── pyproject.toml        # 依赖与配置
├── README.md             # 项目文档
└── uv.lock               # 依赖锁
```

## 开发路线图

1. **已完成**
   - 完成 `fetch_arxiv.py` 模块：按类别抓取 arXiv 新发论文、清洗可选字段并保存 JSON。
   - 完成 `cluster_topics.py` + 相关工具（`preprocess.py`、`embed.py`、`cluster.py`、`summarize.py`）：实现强预处理、UMAP 降维、HDBSCAN 聚类、噪声救援及 LLM 主题标签/关键词/描述输出。

2. **后续计划**
   - 为聚类结果生成更详细的“Cluster Summary Report”，聚合关键词与代表论文。
   - 添加 UMAP 可视化输出（静态图或交互式）以辅助人工审核聚类质量。
   - 在 summary 阶段引入 subcluster analysis，进一步细化每个主题内部的子主题。
   - 完善 `generate_report.py`，输出日报系列（Markdown/HTML）并支持自定义模版。
   - 加入邮件发送模块，将日报推送给订阅用户（可能通过 SMTP/API）。


## 许可证

MIT License

---

**注意**: 使用LLM功能会产生API调用费用，请合理控制使用量。
