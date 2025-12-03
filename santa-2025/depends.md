# Dependencies (Santa-2025) / 依赖清单

English first, then Chinese below each item. 先英文后中文。

## Python Packages / Python 包
- PyTorch (`torch`): PPO training.
  - 用途：PPO训练。
- pybind11: C++/Python bindings for `satgeom`.
  - 用途：C++/Python绑定。
- setuptools: Build the extension.
  - 用途：构建扩展。
- Optional: numpy, matplotlib, tensorboard.
  - 可选：数值与可视化。

## C++ Toolchain / C++ 工具链
- C++17-capable compiler (gcc/clang).
  - 需要支持 C++17 的编译器。
- Build essentials (Linux): `build-essential`.
  - Linux 需安装基本编译工具。

## Conda Installation / Conda 安装

```bash
# Create and activate env / 创建并激活环境
conda create -n santa_env python=3.10 -y
conda activate santa_env

# Core Python deps via conda-forge / 通过 conda-forge 安装核心依赖
conda install -y -c conda-forge pybind11 setuptools

# PyTorch (CUDA optional) / 安装 PyTorch（可选CUDA）
# CPU only:
conda install -y -c pytorch pytorch
# GPU 安装（RTX 5060, SM_120，CUDA 12.8 提示）
# 说明：PyTorch 官方轮子通常按 cu121/cu124/cu126 等标签提供，目前未提供 "cu128"。
# 请选择与你的驱动兼容的最近版本（如 CUDA 12.1 或 12.6）。
# Conda 示例（优先）：
conda install -y -c pytorch -c nvidia pytorch cuda=12.1
# 或者（如环境提供）：
# conda install -y -c pytorch -c nvidia pytorch cuda=12.6

# Pip 示例（当 conda 无法满足时）：
# 根据 PyTorch 官方下载页选择索引 URL，例如 CUDA 12.1：
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# 如未来提供 cu128（CUDA 12.8），请改用对应索引 URL。
```
> github上的方案<br>
*RTX 5060 成功运行 PyTorch Nightly + CUDA 12.9 的完整配置教程*
https://github.com/Scarfy-sysu/rtx5060-pytorch-cuda129

# Optional extras / 可选依赖
conda install -y -c conda-forge numpy matplotlib tensorboard
```

## When Conda Is Not Preferred / Conda 非首选时
- You can use pip for Python deps. 可用 pip 安装：

```bash
pip install pybind11 setuptools torch
pip install numpy matplotlib tensorboard  # optional
```

## Build Extension / 构建扩展
```bash
# From repo root / 在仓库根目录
pip install -e santa-2025
```

## Linux Compiler Setup / Linux 编译器安装
```bash
sudo apt-get update
sudo apt-get install -y build-essential
# gcc --version should show >= 7 (C++17) / 至少支持 C++17
```

## Verify / 验证
```bash
python -c "import satgeom; print('satgeom OK')"
python -c "import torch; print('torch', __import__('torch').__version__)"
# Verify GPU / 验证 GPU
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
  print('GPU name:', torch.cuda.get_device_name(0))
  print('CUDA runtime version:', torch.version.cuda)
PY
```Searched for files matching `**/requirements.txt`, no matches

Searched for files matching `**/CMakeLists.txt`, no matches

Based on my examination of the project files, here are all the dependencies:

## 📦 项目依赖项 / Project Dependencies

### **Python 依赖 / Python Dependencies**

#### 核心库 / Core Libraries:
1. **pybind11** - C++/Python 绑定库 (C++/Python binding library)
   - 用途: 将 C++ SAT 几何模块暴露给 Python
   - 安装: `pip install pybind11`

2. **PyTorch (torch)** - 深度学习框架 (Deep learning framework)
   - 用途: PPO 强化学习实现 (Actor-Critic 网络, 优化器)
   - 相关模块:
     - `torch.nn` - 神经网络组件
     - `torch.optim` - 优化器 (Adam)
     - `torch.distributions` - 概率分布 (Normal)
   - 安装: `pip install torch`

#### 标准库 / Standard Library:
3. **setuptools** - 构建工具 (Build tools)
   - 用途: 编译 C++ 扩展模块
   - 通常随 Python 预装

4. **dataclasses** - 数据类 (Python 3.7+)
   - 用途: PPOConfig 配置类

5. **typing** - 类型提示 (Type hints)
   - 用途: List, Tuple 等类型标注

6. **math, time, random** - 基础工具库

---

### **C++ 依赖 / C++ Dependencies**

#### 编译器要求 / Compiler Requirements:
- **C++17 标准** (C++17 standard)
  - 编译标志: `-std=c++17`
  - 支持的编译器: GCC 7+, Clang 5+, MSVC 2017+

#### 头文件库 / Header Libraries:
1. **pybind11 headers** - Python/C++ 互操作
   - 从 `pybind11` Python 包获取
   - 通过 `pybind11.get_include()` 定位

2. **C++ STL (标准模板库)**:
   - `<vector>` - 动态数组
   - `<cmath>` - 数学函数 (cos, sin, sqrt, M_PI)
   - `<numeric>` - 数值算法
   - `<algorithm>` - 通用算法 (min, max, move)
   - `<iostream>` - I/O 流 (collision_detection.cpp)
   - `<memory>` - 智能指针 (collision_detection.cpp)

---

### **构建系统依赖 / Build System Dependencies**

#### 编译优化 / Compilation Flags:
- `-O2` 或 `-O3` - 优化级别
- 可选: `-march=native` - 本地架构优化

#### Python 环境 / Python Environment:
- **Python 3.9** (推荐版本 / Recommended)
- **Conda** (可选环境管理 / Optional environment manager)
  - 创建命令: `conda create -n santa_env python=3.9`

---

### **快速安装清单 / Quick Installation Checklist**

```bash
# 1. 创建 Python 环境 / Create Python environment
conda create -n santa_env python=3.9 -y
conda activate santa_env

# 2. 安装 Python 依赖 / Install Python dependencies
pip install pybind11 torch

# 3. 构建 C++ 扩展 / Build C++ extension
cd /mnt/d/CS/Kaggle
pip install -e santa-2025

# 4. 验证安装 / Verify installation
python -c "import satgeom; print('✓ satgeom loaded')"
python -c "import torch; print('✓ torch version:', torch.__version__)"
```

---

### **可选依赖 / Optional Dependencies**

对于生产环境或性能优化 / For production or performance:
- **NumPy** - 数值计算 (可能需要用于数据处理)
- **Matplotlib** - 可视化训练曲线
- **TensorBoard** - 训练监控
- **gym/gymnasium** - 标准 RL 环境接口 (如需重构)

---

### **依赖关系图 / Dependency Graph**

```
ppo_agent.py ──┬──> torch (PyTorch)
               └──> ppo_env.py ──> satgeom (C++ extension)
                                      │
                                      └──> pybind11 + geom.hpp
                                             │
                                             └──> C++17 STL
```

所有依赖已列出！需要安装其他包或有疑问请告诉我。