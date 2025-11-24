# DailyArxiv

一个自动抓取arXiv每日论文并生成中文解读的工具。

## 功能特性

- 📚 自动抓取多个arXiv类别的每日论文
- 🌐 支持HTML解析和API两种获取方式
- 🤖 集成DeepSeek LLM解读功能（可选）
- 📝 生成包含中文翻译和术语解释的Markdown文件
- 🗂️ 按类别分类保存论文

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

### 基本使用（无LLM解读）
```bash
python main.py
```

### 启用LLM解读
```bash
python main.py --llm
```

### 自定义LLM处理论文数量
```bash
python main.py --llm --max-papers 5
```

## 输出文件

论文将按类别保存在 `feed/` 目录下，例如：
- `feed/hep-ph/2025-11-24-arxiv.md`
- `feed/astro-ph/2025-11-24-arxiv.md`

### 输出格式示例

```markdown
# arXiv hep-ph 今日论文 (2025-11-24)

共找到 39 篇论文

## 1. A new suite of Lund-tree observables to resolve jets

**作者**: Melissa van Beekveld, Luca Buonocore, Silvia Ferrario Ravasio, Pier Francesco Monni, Alba Soto-Ontoso, Gregory Soyez

**PDF链接**: [https://arxiv.org/pdf/2511.16723.pdf](https://arxiv.org/pdf/2511.16723.pdf)

**摘要**: We introduce a class of collider observables, named Lund-Tree Shapes (LTS)...

### 中文翻译

**标题**: 一套新的Lund树可观测量用于解析喷注

**摘要**: 我们引入了一类对撞机可观测量，称为Lund树形状（LTS）...

### 关键术语解释

**Lund jet plane** (Lund喷注平面)

Lund喷注平面是一种用于描述量子色动力学（QCD）辐射模式的相空间表示方法...

---
```

## 支持的arXiv类别

- astro-ph (天体物理学)
- cond-mat (凝聚态物理)
- gr-qc (广义相对论与量子宇宙学)
- hep-ph (高能物理-现象学)
- hep-th (高能物理-理论)
- math-ph (数学物理)
- quant-ph (量子物理)
- 等等...

## LLM解读功能设置指南

### 概述

DailyArxiv集成了DeepSeek LLM解读功能，可以为每篇论文自动生成：
- 📖 标题和摘要的中文翻译
- 🔬 关键专业术语的解释（适合博士水平）

### 快速开始

#### 1. 获取DeepSeek API密钥

1. 访问 [DeepSeek Platform](https://platform.deepseek.com/)
2. 注册或登录账户
3. 进入API密钥管理页面
4. 创建新的API密钥
5. 复制生成的密钥

#### 2. 配置环境变量

```bash
export DEEPSEEK_API_KEY=你的DeepSeek_API密钥
```

#### 3. 测试LLM功能

```bash
python test_llm.py
```

如果看到成功消息，说明配置正确。

#### 4. 使用LLM解读功能

```bash
# 启用LLM解读，每个类别处理3篇论文（默认）
python main.py --llm

# 启用LLM解读，每个类别处理5篇论文
python main.py --llm --max-papers 5
```

### 功能详解

#### 中文翻译

LLM会为每篇论文生成：
- **标题翻译**：准确的专业翻译
- **摘要翻译**：流畅的中文摘要，保持专业术语的准确性

#### 关键术语解释

系统会自动识别摘要中最重要、最专业的3-5个术语，并为每个术语提供：
- **英文术语**：原始术语名称
- **中文翻译**：术语的中文对应
- **详细解释**：适合博士水平的深入解释

### 配置选项

#### 环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API密钥 | 必需 |
| `OPENAI_MODEL` | 使用的模型 | `deepseek-chat` |
| `OPENAI_BASE_URL` | API基础URL | `https://api.deepseek.com/v1` |

#### 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--llm` | 启用LLM解读 | `False` |
| `--max-papers` | 每个类别最大处理数量 | `3` |

### 故障排除

#### 常见问题

1. **API密钥错误**
   ```
   Error: Incorrect API key provided
   ```
   解决方案：检查API密钥是否正确，是否有足够的余额

2. **网络连接问题**
   ```
   Error: Connection error
   ```
   解决方案：检查网络连接，或尝试使用代理

3. **速率限制**
   ```
   Error: Rate limit exceeded
   ```
   解决方案：减少处理数量，或添加延迟

4. **Token限制**
   ```
   Error: Context length exceeded
   ```
   解决方案：论文摘要过长，系统会自动处理

### 高级用法

#### 自定义提示词

修改 `llm_processor.py` 中的 `_build_prompt` 方法来自定义提示词：

```python
def _build_prompt(self, title: str, abstract: str) -> str:
    # 自定义你的提示词
    return f"""
    请为以下论文生成专业的中文解读：
    标题：{title}
    摘要：{abstract}
    
    ...
    """
```

#### 使用其他模型

支持所有OpenAI兼容的模型：
```python
processor = LLMProcessor(model="deepseek-reasoner")
```

## 项目结构

```
DailyArxiv/
├── main.py              # 主程序
├── llm_processor.py     # LLM处理模块
├── test_llm.py         # LLM功能测试脚本
├── pyproject.toml       # 项目配置和依赖
├── README.md           # 项目说明
└── feed/               # 论文输出目录
    ├── hep-ph/
    ├── astro-ph/
    └── ...
```

## 开发

### 添加新的arXiv类别

在 `main.py` 的 `categories` 列表中添加新的类别代码。

### 自定义LLM提示词

修改 `llm_processor.py` 中的 `_build_prompt` 方法来自定义提示词。

## Phase 2：LLM解读模块整合完成

### 任务完成情况

已成功为DailyArxiv项目整合LLM解读模块，为每篇论文增加了两个新字段：

1. **中文翻译**（标题 + 摘要）
2. **专业术语解释**（适合博士水平）

### 主要功能特性

#### 1. LLM解读功能
- 使用DeepSeek模型生成专业解读
- 自动识别关键术语并提供深入解释
- 生成准确流畅的中文翻译

#### 2. 批量处理
- 支持多篇论文批量处理
- 可配置每个类别的处理数量
- 自动延迟避免API限制

#### 3. 错误处理
- 完善的异常处理机制
- API失败时的降级处理
- 详细的错误日志

#### 4. 配置灵活
- 支持命令行参数
- 环境变量配置
- 可自定义模型和参数

### 技术实现

#### 架构设计
- 模块化设计，LLM功能独立封装
- 与现有代码无缝集成
- 支持扩展其他LLM提供商

#### 提示词工程
使用优化的提示词：
```
请阅读以下论文内容：
标题：{title}
摘要：{abstract}

请生成如下内容：
1. 标题与摘要的中文翻译
2. 摘要中的关键术语列表，并解释每个术语（适合博士水平）
```

### 完成标准

- [x] 实现LLM解读核心功能
- [x] 集成到主程序
- [x] 支持命令行参数
- [x] 完善错误处理
- [x] 创建使用文档
- [x] 编写测试脚本
- [x] 更新项目文档

## 许可证

MIT License

---

**注意**: 使用LLM功能会产生API调用费用，请合理控制使用量。