"""
English: setup.py builds the pybind11 C++ extension exposing SAT geometry.
中文：该 setup.py 用于构建 pybind11 C++ 扩展，暴露 SAT 几何功能。
Step-by-step explanation below each logical block.
"""

from setuptools import setup, Extension  # EN: Core build utilities. CN：核心构建工具。
import sys                               # EN: Not strictly needed; left for potential future use. CN：当前未使用，保留以便扩展。
from pathlib import Path                 # EN: Path handling for include dirs. CN：路径处理。

try:
    import pybind11                     # EN: Attempt to import pybind11 to obtain include path. CN：导入 pybind11 获取头文件路径。
except ImportError:
    print("pybind11 not found. Please install: pip install pybind11")  # EN: User guidance. CN：提示用户安装。
    raise                                # EN: Stop build if missing. CN：缺失则终止。

# EN: Collect include directories: pybind11 headers + current folder for geom.hpp.
# CN：收集头文件路径：pybind11 的头文件 + 当前目录以找到 geom.hpp。
include_dirs = [pybind11.get_include(), str(Path(__file__).parent)]

# EN: Define extension module metadata.
# CN：定义扩展模块元数据。
ext_modules = [
    Extension(
        'satgeom',                      # EN: Module name imported in Python. CN：Python 中导入的模块名。
        sources=['sat_bindings.cpp'],   # EN: C++ source implementing bindings. CN：实现绑定的 C++ 源文件。
        include_dirs=include_dirs,      # EN: Header search paths. CN：头文件搜索路径。
        language='c++',                 # EN: Explicitly set language. CN：显式设定语言。
        extra_compile_args=['-std=c++17','-O2']  # EN: Use C++17 + optimization O2. CN：使用 C++17 与 O2 优化。
    )
]

# EN: Setup call builds wheel/extension when invoked (e.g. build_ext --inplace).
# CN：setup 调用在执行 build_ext --inplace 时构建扩展。
setup(
    name='satgeom',                     # EN: Package name. CN：包名称。
    version='0.1.0',                    # EN: Initial version. CN：初始版本号。
    description='Simple SAT convex polygon geometry (pybind11)',  # EN: Short description. CN：简要描述。
    ext_modules=ext_modules,            # EN: Pass extension list. CN：传入扩展列表。
    zip_safe=False,                     # EN: Not safe to zip (compiled ext). CN：编译扩展不适合 zip 压缩。
)
