# 工业 AI 视觉质检系统 (InspectAI) 🚀

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![AI-PM](https://img.shields.io/badge/定位-AI--产品经理-orange.svg)

这是一个基于 **PatchCore** 算法的工业级端到端表面缺陷检测系统。本项目专为 **AI 产品经理 (AI-PM)** 的作品集设计，旨在展示在制造业（如 AOI/质量控制）场景下的技术深度、业务理解力及落地能力。

## 🌟 核心特性
- **无监督异常检测**：仅需“合格品”样本即可完成模型训练——完美解决工业场景下坏品样本稀缺的冷启动痛点。
- **SOTA 级性能**：在 MVTec AD 瓶子数据集上达到 100% AUROC，实现 **0.0% 漏杀率 (FRR)**。
- **像素级缺陷定位**：自动生成异常热力图，精准标注划痕、破损、污染等缺陷位置。
- **高级质检看板**：基于 FastAPI 实现的深色系、高对比度 Web 交互界面，支持实时推理与良率监控。

## 🛠 技术栈
- **AI 核心**: Python, TensorFlow (ResNet-50 特征提取), Numpy (Coreset 记忆库), Scikit-Learn。
- **后端服务**: FastAPI, Uvicorn (高性能异步架构)。
- **前端交互**: HTML5, Vanilla CSS (高级玻璃拟态设计), JavaScript (ES6)。
- **数据集**: MVTec AD (工业视觉标准数据集)。

## 📊 核心业务指标 (PM 关注点)
| 指标项 | 结果 | 业务影响 |
| :--- | :--- | :--- |
| **FAR (过杀率)** | 10.0% | 决定了产线二次人工复核的工作量。 |
| **FRR (漏杀率)** | **0.0%** | 确保零缺陷流向客户，保护品牌声誉。 |
| **检测速度** | ~1.2s/pcs | 匹配中高速流水线的节拍需求。 |

## 🚀 快速开始

### 1. 环境准备
```bash
git clone https://github.com/Cordi100/industrial-testing.git
cd industrial-testing
pip install -r requirements.txt
```

### 2. 模型训练
将 MVTec 瓶子数据集放入 `data/mvtec_ad/bottle` 目录。
```bash
python -m src.patchcore
```

### 3. 启动质检看板
```bash
python -m uvicorn web.app:app --reload
```
访问 `http://localhost:8000` 即可开始体验 AI 质检！

## 📂 项目结构
```text
industrial_ML/
├── data/           # (已忽略) MVTec AD 原始数据集
├── models/         # (已忽略) 训练好的模型权重 (.pkl)
├── src/            # AI 核心逻辑 (算法、评估引擎)
├── web/            # 质检应用层 (FastAPI 路由、前端资产)
├── README.md       # 项目中文说明文档
└── requirements.txt
```
