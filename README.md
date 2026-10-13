# PaperAgent

**简体中文** | [English](README_EN.md)

面向深空天体测量与行星历表领域的学术文献智能体，支持混合路由检索、长思维链学术推演、文件级精准溯源与双端自适应交互。

[在线体验](https://paper.sues.fun) · Streamlit · FAISS · BGE-M3 · MinerU · 前沿学术推理内核

**作者**：[杨哲](https://zheoi.com)

---

## 界面预览

![PaperAgent 桌面端界面](assets/screenshots/desktop-chat.png)

<p align="center">
  <img src="assets/screenshots/mobile-chat.png" alt="PaperAgent 移动端界面" width="390">
</p>

---

## 项目简介

**PaperAgent** 是专为深空天体测量学（Astrometry）、行星与不规则卫星精密历表、Gaia 星表观测改正等科研文献打造的学术智能体。

系统采用**混合查询路由**架构：
1. **确定性精准匹配**：作者论文清单、历表数据来源、指定对象对比等高频结构化查询直接命中确定性索引，毫秒级响应，杜绝无效生成；
2. **深度学术 RAG**：复杂机理推演、参数残差与观测方法总结自动进入增强检索生成链，结合前沿长上下文模型输出具备严谨逻辑与文件级溯源依据的学术答案。

当前知识库收录了 61 篇深空天体测量核心论文，支持极速推理、深度推演等多档位切换，具备透明可折叠的思维链（Reasoning）与运行诊断监控。

---

## 核心特性

- **双轨混合路由**：结构化确定性查询毫秒级精准直出，开放学术问题进入增强 RAG。
- **高精文献知识库**：采用 MinerU 提取论文图表与公式、1600 字符层次切片、BGE-M3 语义向量化及 FAISS 高速索引。
- **多档位前沿推理内核**：
  - ⚡ **极速推理（Flash）**：超长上下文窗口与敏捷响应，适合日常快速文献查阅；
  - 🧠 **深度推演（Thinking）**：开启深层思维链，专攻观测处理方法对比、残差精度与机理推演；
  - 🔬 **综合研报（Pro）**：长文本跨篇综合提炼，快速构建领域综述。
- **透明溯源与严谨考证**：每条学术结论均关联论文名称、页码定位与原文摘录卡片。
- **现代自适应交互界面**：基于 Streamlit 深度定制，自适应适配桌面宽屏与移动端手机视口，布局零溢出，交互优雅流畅。

---

## 架构设计

```
[ 学术论文库 (PDF) ]
        │
        ▼ (MinerU 深度解析)
[ 结构化 Markdown / 文本清洗 ]
        │
        ├────────────────────────────────┐
        ▼ (层次切块 & BGE-M3 向量化)      ▼ (元数据与关键词索引)
  [ FAISS 向量知识库 ]              [ 确定性学术索引 (JSON) ]
        │                                │
        └────────────────┬───────────────┘
                         ▼
                 [ 混合查询路由器 ]
                 /              \
     (结构化命中)                (开放语义提问)
         /                          \
[ 确定性极速响应 ]            [ 增强检索 RAG 链 ]
                                     │
                                     ▼
                        [ 长上下文推理内核 ]
                         (Thinking / Flash)
                                     │
                                     ▼
                        [ 双端自适应 Web 交互 ]
```

---

## 快速开始

### 运行环境
- Python 3.10+
- 本地 BGE-M3 嵌入模型 / FAISS

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，配置推理接口地址、认证密钥及默认模型档位
```

### 3. 启动 Web 界面
```bash
streamlit run app.py --server.port 8000
```

---

## 构建文献知识库

如需导入自有学术论文语料：
1. 将论文 PDF 放入 `data/papers/` 目录；
2. 执行 MinerU 解析：
   ```bash
   bash scripts/parse_with_mineru.sh
   ```
3. 构建向量索引与结构化检索库：
   ```bash
   bash scripts/rebuild_index.sh
   ```

---

## 测试验证

项目内置完善的自动化单元测试（覆盖查询路由、推理客户端、性能配置、RAG 提示词及检索逻辑）：

```bash
python -m unittest discover -s tests
```

---

## 作者与支持

- **作者**：杨哲
- **个人主页**：[zheoi.com](https://zheoi.com)
- **学术领域**：深空天体测量与行星历表动力学

---

## License

[Apache License 2.0](LICENSE)
