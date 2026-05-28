# GLM-OCR 一键部署

基于智谱 AI [GLM-OCR](https://github.com/zai-org/GLM-OCR) (0.9B) 的本地一键部署方案，集成 Gradio Web UI，支持单图/PDF 批量 OCR 识别。

## 环境要求

- Windows 10/11
- NVIDIA GPU (8GB+ 显存)，推荐 RTX 3060 及以上
- 至少 10GB 可用磁盘空间

## 快速开始

### 1. 安装（仅首次）

双击 **`install_glmocr.bat`**，自动完成：
- 创建 Python 3.12 环境
- 安装 PyTorch + CUDA、Transformers、Gradio 等依赖
- 从 ModelScope 下载 GLM-OCR 模型 (~2GB)

### 2. 启动

双击 **`一键启动GLMOCR.bat`**，浏览器自动打开 Web UI。

## 功能

- **单图识别**：上传图片，一键 OCR 输出 Markdown
- **PDF 批量处理**：上传 PDF，自动逐页识别
- **多种模式**：通用 Markdown / 纯文本 / 表格 / 公式识别
- **结果导出**：Markdown 全文 + JSONL 知识库分块

## 文件说明

```
├── glmocr_webui.py          # Gradio Web UI 主程序
├── install_glmocr.bat       # 双击安装
├── install_glmocr.ps1       # 安装脚本
├── start_glmocr.ps1         # 启动脚本
├── 一键启动GLMOCR.bat       # 双击启动
└── .gitignore
```

## 许可证

本项目代码遵循 MIT License。GLM-OCR 模型由智谱 AI 开源，遵循 MIT License。
