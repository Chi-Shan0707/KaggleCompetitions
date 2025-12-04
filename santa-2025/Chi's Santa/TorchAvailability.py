try:
    import torch
except ModuleNotFoundError:
    print("未安装 PyTorch：请先安装后再检测。")
    print("示例安装命令（根据需要选择CPU或CUDA版本）：")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("或（CUDA 12.1）：")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
else:
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 版本:", torch.version.cuda)

    cuda_available = torch.cuda.is_available()
    print("CUDA 是否可用:", cuda_available)

    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            print("可用GPU数量:", device_count)
            for i in range(device_count):
                print(f"GPU {i} 名称:", torch.cuda.get_device_name(i))

            # 简单张量分配测试
            x = torch.rand(1).to("cuda")
            print("CUDA 张量测试成功:", x.device)

        except Exception as e:
            print("CUDA 测试失败:", e)
    else:
        print("GPU 不可用（可能未安装CUDA或驱动/WSL配置问题）")

import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 版本:", torch.version.cuda)
print("CUDA 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
else:
    print("GPU 不可用")