# VibeVoice Desktop 语音处理系统

## 1. 项目介绍
**VibeVoice Desktop** 是基于 **Electron** 和 **Vue 3** 框架对原始 [VibeVoice](https://github.com/vibe-voice/VibeVoice) 语音系统进行的深度桌面化封装。本项目旨在将强大的 AI 语音识别（ASR）与语音合成（TTS）能力带给普通用户，提供一个稳定、直观、且无需复杂配置的生产力工具。

系统集成了 **ASR 任务队列**（支持长音频）、**流式 TTS 合成**、**历史记录管理**以及**一键批量导出**功能，并支持 **GPU(CUDA)** 与 **CPU** 模式的动态切换。

## 2. 核心架构
系统采用“前后端分离 + 桌面容器”的架构：
- **前端 (Renderer Process)**: 使用 Vue 3 + TypeScript + Element Plus 构建。负责 UI 交互、实时状态显示、历史记录预览及播放。
- **主进程 (Main Process)**: Electron 核心。负责后端 FastAPI 进程的生命周期监控、文件系统操作及嵌入式运行时管理。
- **后端 (Python Backend)**: 基于 FastAPI 封装的 VibeVoice 核心模型服务。

## 3. 目录结构
```text
VibeVoice/
├── vibevoice/              # 核心引擎 (Python 逻辑)
│   ├── modular/            # 模型定义与推理模块
│   ├── processor/          # 音频与文本处理器
│   ├── schedule/           # 扩散模型调度器
│   └── service.py          # 封装的 TTS 服务类
├── vibevoice-app/          # 桌面端应用 (Electron + Vue)
│   ├── electron/           # Electron 主进程代码
│   ├── src/                # Vue 前端渲染进程代码
│   └── backend/            # 应用后端
│       ├── server.py       # FastAPI 服务入口
│       └── resources/      # 静态资源 (音色模型等)
├── pyproject.toml          # Python 项目配置
└── README.md               # 项目说明
```

## 4. 本地运行指南

### 前置要求
- **Node.js**: v16+
- **Python**: v3.10+ (推荐使用虚拟环境)
- **CUDA**: 如需使用 GPU 加速，需安装相应的 NVIDIA 驱动和工具包。

### 步骤 1：安装 Python 依赖
在项目根目录下，安装 VibeVoice 核心依赖：
```bash
pip install -e .
pip install fastapi uvicorn librosa
```

### 步骤 2：安装前端依赖
进入 `vibevoice-app` 目录：
```bash
cd vibevoice-app
npm install
```

### 步骤 3：启动开发环境
在 `vibevoice-app` 目录下运行：
```bash
npm run dev
```
此命令将同时启动前端开发服务器和后端的 FastAPI 服务。

## 5. 目前开发进度

### 已完成功能 (Phase 1-5)
- [x] **后端封装**: 基于 FastAPI 实现 ASR 和 TTS 的接口封装。
- [x] **进程管理**: Electron 自动启动/关闭/重启 Python 后端。
- [x] **ASR 增强**: 实现异步 Job 队列，支持长音频转写进度显示及取消。
- [x] **TTS 优化**: 支持角色音色自动扫描、流式合成预览。
- [x] **历史记录**: 实现 ASR/TTS 结果的本地持久化存储。
- [x] **设置中心**: 支持动态切换运行设备 (CPU/GPU) 及端口配置。
- [x] **导出系统**: 支持自定义导出路径及批量导出。
- [x] **Windows 嵌入式运行时集成**: Electron 主进程在生产模式下改为加载 `python-win/python.exe`，并补齐 `PYTHONHOME/PYTHONPATH` 及 `electron-builder` 资源打包配置，实现“随包可运行”的基础能力。

### 进行中功能
- [ ] **打包回归测试**: Windows 安装包全链路验证（安装/启动/ASR/TTS/导出/重启后端）与异常场景修复。
- [ ] **UI 润色**: 进一步优化界面动画与交互细节。

### 最新打包产物（Windows）
- [x] **首次打包验证通过**: 已成功生成 NSIS 安装包（x64）。
- 安装包下载（GitHub Release）：[VibeVoice Desktop Setup 0.1.0.exe](https://github.com/whr810012/VibeVoice-APP/releases/download/v0.1.0/VibeVoice-Desktop-Setup-0.1.0.exe)
- 解包目录：`vibevoice-app/release/win-unpacked`
- 备注：当前使用默认 Electron 图标（尚未配置 `build/icon.ico`）；若版本号变化，请同步更新 Release 链接中的 `v0.1.0` 与文件名。

## 6. 后续开发计划
- [ ] **多语言支持**: 引入 i18n，支持中英文界面切换。
- [ ] **模型热更新**: 支持在 UI 上直接下载/更新模型权重文件。
- [ ] **实时监听**: 实现类似语音助手的实时语音唤醒与识别模式。
- [ ] **多端适配**: 完成 macOS 与 Linux 平台的打包适配。

## 7. 核心优势（对比原始 VibeVoice）

### 🚀 一键部署，零环境依赖
- **现状**: 原始版本需要用户手动安装 Python、Git，配置虚拟环境，并处理复杂的 pip 依赖冲突。
- **优势**: 本系统集成了 **嵌入式 Python 环境**。用户只需解压或安装 `.exe` 包即可运行，真正实现“双击即用”，无需预装任何编程环境。

### 📊 全流程可视化交互
- **现状**: 原始版本主要通过命令行（CLI）或基础 Web 界面操作，缺乏直观的任务进度反馈。
- **优势**: 
    - **ASR 进度条**: 实时显示长音频识别百分比，支持随时手动取消任务。
    - **TTS 预览播放**: 合成后立即通过内置播放器试听，无需在文件夹中寻找文件。
    - **可视化配置**: 在 UI 上直接切换 GPU/CPU 模式、修改 API 端口，实时生效。

### 🛡️ 智能后端进程管理
- **现状**: 原始版本需要手动启动后端脚本，且修改配置后需手动重启服务。
- **优势**: Electron 主进程自动托管 Python 后端。当用户在设置页切换“计算设备”时，系统会**自动重启后端进程**并应用新环境，确保模型加载无缝切换。

### 📂 增强的数据资产管理
- **现状**: 原始生成的文件分散在临时文件夹中，查找困难且容易丢失。
- **优势**: 
    - **历史记录中心**: 自动持久化存储所有 ASR 文本和 TTS 音频，支持重命名和批量管理。
    - **一键批量导出**: 支持用户选择自定义目录，一键将历史任务分类导出到本地指定位置。

### ⚙️ 硬件自适应与智能检测
- **现状**: 原始版本依赖用户自行判断 CUDA 是否可用，配置错误会导致程序崩溃。
- **优势**: 启动时自动检测系统 CUDA 环境，并在设置页面提供清晰的设备状态提示。支持一键切换 **CUDA 加速** 与 **CPU 兼容模式**，确保低配硬件也能稳定运行。

---
© 2026 VibeVoice Desktop Team. Powered by VibeVoice Core.
