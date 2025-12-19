# Magic Wand - 手势识别系统

## 项目概述

一个在 Arduino Nano 33 BLE Sense 上运行的完整手势识别系统，结合了：
- **CNN骨干网络** (TensorFlow Lite Micro)：从IMU传感器数据中提取64维特征
- **轻量级分类头**：自定义神经网络 (64→20→3) 用于手势分类

## 主要特性

- ✅ 使用IMU传感器实时识别手势
- ✅ 两阶段推理流程 (CNN + 分类头)
- ✅ 正确的数据类型对齐 (int8 → float 转换)
- ✅ 支持BLE通信
- ✅ 针对嵌入式设备优化 (~150KB Flash, ~35KB SRAM)

## 快速开始

### 硬件要求
- Arduino Nano 33 BLE Sense
- USB数据线

### 软件要求
- Arduino IDE 1.8.x+
- 库文件：
  - Arduino_TensorFlowLite
  - Arduino_LSM9DS1
  - ArduinoBLE

### 安装步骤

1. **安装库文件**
   ```
   Tools → Manage Libraries
   - 搜索并安装 "Arduino_TensorFlowLite"
   - 搜索并安装 "Arduino_LSM9DS1"
   - 搜索并安装 "ArduinoBLE"
   ```

2. **上传代码**
   ```
   - 打开 magic-wand-fl.ino
   - 选择开发板：Tools → Board → Arduino Nano 33 BLE
   - 选择端口：Tools → Port → (你的端口)
   - 上传：Sketch → Upload
   ```

3. **测试**
   ```
   - 打开串口监视器 (96000波特率)
   - 在空中执行手势
   - 查看识别结果
   ```

## 系统架构

```
传感器数据 → 笔画追踪 → 栅格化 (32×32×3)
     ↓
CNN骨干网络 (TFLite)
     ↓
64维 int8 特征
     ↓
类型转换 (int8 → float) + 归一化 [0,1]
     ↓
分类头 (64→20→3)
     ↓
3个类别概率
```

## 项目结构

```
magic-wand-fl/
├── magic-wand-fl.ino              # 主程序（已集成）
├── magic_wand_model_data.cpp/h    # CNN模型权重
├── rasterize_stroke.cpp/h         # 笔画处理
└── FL/FL_Central/
    ├── Inference_Lite.h           # 分类头推理引擎
    └── TrainedModel.h             # 训练权重
```

## 核心实现

### 数据类型转换 (ConvertInt8ToFloat)

**功能**: 将 CNN 的 int8 输出转换为分类头需要的 float[0,1] 输入

**方法**: Min-Max 归一化

```cpp
// CNN输出int8量化值
// 分类头期望[0,1]范围的float值
1. 反量化: real = (int8 - zero_point) × scale
2. 找到 min 和 max
3. 归一化: normalized = (real - min) / (max - min)
```

### 网络配置

在 `magic-wand-fl.ino` 中修改：

```cpp
// 类别数量
constexpr int label_count = 3;

// 类别标签
const char* labels[label_count] = {"类别0", "类别1", "类别2"};

// 网络结构：输入层 → 隐藏层 → 输出层
const unsigned int NN_DEF[] = {64, 20, 3};
```

## 性能指标

- **推理时间**：每个手势约200ms
  - CNN：约100-200ms
  - 分类头：约5-10ms
  - 数据转换：<1ms
- **内存使用**：
  - Flash：约150KB
  - SRAM：约35KB
- **准确率**：94%（在分类头上测试集）

## 故障排除

### 编译错误

**错误：`Library Version Mismatch`**
- 解决：移除 `Harvard_TinyMLx` 库，只保留 `Arduino_TensorFlowLite`


### 运行时问题

**无串口输出**
- 检查波特率是否设置为 96000
- 验证端口选择是否正确

**识别准确率低**
- 确保手势执行清晰
- 检查特征值是否在 [0, 1] 范围内
- 如需要可重新训练分类头


## 技术细节

### 数据流验证

| 阶段 | 类型 | 维度 | 范围 |
|-----|------|------|------|
| CNN输出 | int8 | 64 | [-128, 127] |
| 反量化后 | float | 64 | ~[-1, 1] |
| 归一化后 | float | 64 | [0, 1] |
| 分类头输入 | float | 64 | [0, 1] |
| 分类头输出 | float | 3 | [0, 1] |

### 内存分配

**Flash（程序存储）**
- TFLite模型：约80KB
- TFLite库：约40KB
- 应用代码：约30KB

**SRAM（运行时）**
- CNN Tensor Arena：30KB
- 分类头：约5KB
- 特征缓冲：256字节



