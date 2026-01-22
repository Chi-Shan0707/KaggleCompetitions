# ERROR PATTERNS & PREVENTION STRATEGIES

本文档记录了整个 Disaster Tweets 分类项目中识别到的错误模式及其预防策略。

---

## 错误模式目录 (Pattern Index)

| Pattern ID | 错误模式 | 组件 | 根本原因 | 预防策略 | 状态 |
|-----------|---------|------|---------|---------|------|
| **P001** | 索引越界错误 (Index Out of Bounds) | `inference.py` | 采样大小 > 数据大小 | `safe_sample()` 自动调整 | ✅ 已修复 |
| **P002** | 类型不匹配 (TypeError/AttributeError) | `model_trainer.py` | 返回值格式不统一 (tuple vs ModelOutput) | 强制 `return_dict=True` | ✅ 已修复 |
| **P003** | 设备不一致 (Device Mismatch) | `model_trainer.py` | 张量在不同设备 (GPU/CPU) | 动态设备同步检查 | ✅ 已修复 |
| **P004** | 缺失数据 (Missing Values) | `data_preprocessor.py` | 未验证 CSV 数据完整性 | 数据验证通道 (audit trail) | ✅ 已修复 |
| **P005** | 空序列错误 (Empty Sequence) | `data_preprocessor.py` | URL/mention 清理后序列为空 | 最小长度验证 + 回退 `[EMPTY]` | ✅ 已修复 |
| **P006** | 配置参数错误 (Invalid Config) | `config.py` | 无范围检查 | 参数范围验证 + 文档化 | ✅ 已修复 |
| **P007** | 模型序列化失败 (Pickle Error) | `model_trainer.py` | 局部类无法序列化 | 全局类定义 (module-level) | ✅ 已修复 |
| **P008** | 版本不兼容 (Version Mismatch) | 所有模块 | 旧代码 vs 新 transformers 4.57.6 | API 检查 + 向前兼容 | ✅ 已修复 |

---

## 详细分析与修复方案

### P001: 索引越界错误 (Index Out of Bounds)

**症状**: `IndexError: list index out of range` 或 `ValueError: cannot take a larger sample than population`

**根本原因**: 
```python
# 错误代码
sample_indices = np.random.choice(len(dataset), sample_size=100, replace=False)
# 当 len(dataset) < 100 时崩溃
```

**预防策略**:
```python
# 修复代码 (inference.py)
def safe_sample(self, dataset, sample_size):
    actual_size = min(sample_size, len(dataset))  # 自动调整
    indices = np.random.choice(len(dataset), size=actual_size, replace=False)
    return dataset.select(indices), actual_size
```

**应用位置**: `inference.py:validate_sample_predictions()`

---

### P002: 返回值类型不统一 (Type Mismatch)

**症状**: `AttributeError: 'numpy.ndarray' object has no attribute 'predictions'`

**根本原因**:
- transformers 不同版本返回不同类型 (tuple vs ModelOutput)
- 代码尝试访问不存在的属性

**预防策略** - 强制标准化输出:
```python
# WeightedBertModel.forward() 中强制 return_dict=True
outputs = super().forward(..., return_dict=True, **kwargs)
# 保证输出永远是 ModelOutput，不再处理 tuple
```

**应用位置**: `model_trainer.py:WeightedBertModel.forward()`

---

### P003: 设备不一致 (Device Mismatch)

**症状**: `RuntimeError: expected device cuda:0 but got device cpu`

**根本原因**:
```python
# 错误代码
loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)  # weights 在 CPU
loss = loss_fct(logits.to('cuda'), labels)  # logits 在 GPU
# 设备不匹配
```

**预防策略**:
```python
# 修复代码
if self.class_weights.device != outputs.logits.device:
    self.class_weights = self.class_weights.to(outputs.logits.device)
```

**应用位置**: `model_trainer.py:WeightedBertModel.forward()`

---

### P004: 缺失数据验证 (Missing Values)

**症状**: `ValueError: cannot convert float NaN to integer` 或数据处理崩溃

**根本原因**: 未在数据加载阶段验证 CSV 完整性

**预防策略** - 数据审计通道:
```python
# data_preprocessor.py 中的审计输出
print(f"[DATA_AUDIT] Original shape: {df.shape}")
print(f"  Text column null count: {df[text_col].isna().sum()}")
print(f"  Label column null count: {df[label_col].isna().sum()}")

df = df.dropna(subset=[text_col])  # 显式删除
```

**应用位置**: `data_preprocessor.py:prepare_dataset()`

---

### P005: 空序列处理 (Empty Sequence)

**症状**: 清理后的文本变为空字符串，BERT tokenizer 返回不预期的结果

**根本原因**:
```python
text = "http://example.com @user #"  # 全是 URL、mention、hashtag
cleaned = clean_text(text)  # → "" (空)
# 后续处理失败
```

**预防策略** - 回退机制:
```python
def clean_text(self, text):
    if pd.isna(text):
        return "[EMPTY]"  # 回退标记
    
    # ... 清理逻辑 ...
    
    if not text:
        return "[EMPTY]"  # 清理后若为空，返回回退标记
```

**应用位置**: `data_preprocessor.py:clean_text()`

---

### P006: 配置参数验证 (Invalid Config)

**症状**: `AssertionError: LEARNING_RATE should be between 0 and 1` 或训练失败

**根本原因**: 参数范围无限制，容易被设置为不合理值

**预防策略** - 范围验证与文档化:
```python
# config.py
_BOUNDS = {
    'BATCH_SIZE': (1, 256),
    'LEARNING_RATE': (1e-8, 1e-2),
    'NUM_EPOCHS': (1, 50),
    ...
}

@classmethod
def validate(cls):
    for param, (min_val, max_val) in cls._BOUNDS.items():
        actual = getattr(cls, param)
        if not (min_val <= actual <= max_val):
            raise ValueError(f"{param} outside bounds")
```

**参数效果文档** - 见 `config.py` 中的 PARAMETER EFFECTS GUIDE

---

### P007: 模型序列化失败 (Pickle Error)

**症状**: `AttributeError: Can't pickle local object 'ModelTrainer.train.<locals>.WeightedBertModel'`

**根本原因**: 类定义在函数内部 (local scope)，pickle 无法序列化

**预防策略** - 全局类定义:
```python
# model_trainer.py 顶部（模块级别）
class WeightedBertModel(BertForSequenceClassification):
    def __init__(self, model, weights):
        super().__init__(model.config)
        self.bert = model.bert
        self.classifier = model.classifier
        self.class_weights = weights
    
    def forward(self, ...):
        ...

# 这样定义的类可以被序列化
```

**应用位置**: `model_trainer.py:class WeightedBertModel` (顶部)

---

### P008: 版本不兼容 (Version Mismatch)

**症状**: `TypeError: unexpected keyword argument` 或模块找不到

**根本原因**:
- `requirements.txt`: transformers 4.30.0
- 实际环境: transformers 4.57.6
- API 已变化，导致参数不被支持

**预防策略** - API 兼容性检查:
```python
# model_trainer.py
supported_params = inspect.signature(TrainingArguments.__init__).parameters
filtered_kwargs = {k: v for k, v in ta_kwargs.items() if k in supported_params}

# 移除不支持的参数
ignored = set(ta_kwargs) - set(filtered_kwargs)
if ignored:
    print(f"[DEBUG] Ignored unsupported (v{transformers.__version__}): {ignored}")
```

**应用位置**: 
- `requirements.txt`: 更新到兼容版本
- `model_trainer.py:_get_training_args()`: 参数过滤
- 所有模块: 版本检查

---

## 快速修复检查清单

当遇到新错误时，按这个顺序排查:

- [ ] **第1步**: 查看错误堆栈中出现的模块 → 对应 Pattern ID
- [ ] **第2步**: 找到本表格中的 Pattern，查看"预防策略"
- [ ] **第3步**: 检查"应用位置"代码是否已按预防策略修改
- [ ] **第4步**: 若问题持续，启用 `DEBUG_ENABLED=True` 获取详细日志

---

## 测试覆盖矩阵

| Pattern | 测试场景 | 预期行为 | 当前状态 |
|---------|---------|---------|---------|
| P001 | `validate_sample_predictions(dataset, sample_size=1000)` 而 dataset 只有 100 样本 | 自动调整为 100，不崩溃 | ✅ |
| P002 | 调用 `trainer.predict()` | 返回 numpy 数组，不报 AttributeError | ✅ |
| P003 | 在 GPU 上训练，权重在 CPU | 权重自动移到 GPU | ✅ |
| P004 | 输入 CSV 有 NaN 值 | 显示审计日志，自动删除 | ✅ |
| P005 | 推文仅包含 URL | 返回 `[EMPTY]`，不导致空序列错误 | ✅ |
| P006 | `Config.LEARNING_RATE = 10` (超出范围) | 在 `Config.validate()` 时报错 | ✅ |
| P007 | 保存并加载 WeightedBertModel | 成功加载，无 Pickle 错误 | ✅ |
| P008 | 在 transformers 4.57.6 环境运行 | 自动适配，使用支持的参数 | ✅ |

---

## 版本兼容性说明

项目已更新为与以下版本兼容:

```
transformers==4.57.6
torch==2.9.1+cu130
datasets==4.48.0
scikit-learn==1.5.1
pandas==2.2.0
numpy==1.26.4
```

见 `requirements.txt` 了解最新版本。

---

## 维护指南

当添加新功能时，记得:

1. **确认输出类型**: 所有返回值明确标注类型 (Type hints)
2. **验证设备**: 涉及张量操作时检查设备一致性
3. **边界检查**: 采样、索引、数组操作前验证大小
4. **错误消息**: 使用 `[ERROR]` 前缀和详细上下文
5. **日志记录**: 在 `[DEBUG]` 输出中记录关键决策

---

**最后更新**: 2026-01-22 | **负责人**: GitHub Copilot
