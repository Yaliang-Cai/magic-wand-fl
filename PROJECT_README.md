# Magic Wand - Gesture Recognition System

## Overview

A complete gesture recognition system for Arduino Nano 33 BLE Sense that combines:
- **CNN Backbone** (TensorFlow Lite Micro): Extracts 64-dimensional features from IMU sensor data
- **Lightweight Classification Head**: Custom neural network (64→20→3) for gesture classification

## Features

- ✅ Real-time gesture recognition using IMU sensors
- ✅ Two-stage inference pipeline (CNN + Classification Head)
- ✅ Proper data type alignment (int8 → float conversion)
- ✅ BLE communication support
- ✅ Optimized for embedded devices (~150KB Flash, ~35KB SRAM)

## Quick Start

### Hardware Requirements
- Arduino Nano 33 BLE Sense
- USB cable

### Software Requirements
- Arduino IDE 1.8.x+
- Libraries:
  - Arduino_TensorFlowLite
  - Arduino_LSM9DS1
  - ArduinoBLE

### Installation

1. **Install Libraries**
   ```
   Tools → Manage Libraries
   - Search and install "Arduino_TensorFlowLite"
   - Search and install "Arduino_LSM9DS1"
   - Search and install "ArduinoBLE"
   ```

2. **Upload Code**
   ```
   - Open magic-wand-fl.ino
   - Select Board: Tools → Board → Arduino Nano 33 BLE
   - Select Port: Tools → Port → (your port)
   - Upload: Sketch → Upload
   ```

3. **Test**
   ```
   - Open Serial Monitor (96000 baud)
   - Perform gestures in the air
   - View recognition results
   ```

## System Architecture

```
Sensor Data → Stroke Tracking → Rasterization (32×32×3)
     ↓
CNN Backbone (TFLite)
     ↓
64-dim int8 features
     ↓
Type Conversion (int8 → float) + Normalization [0,1]
     ↓
Classification Head (64→20→3)
     ↓
3 Class Probabilities
```

## Project Structure

```
magic-wand-fl/
├── magic-wand-fl.ino              # Main program (integrated)
├── magic_wand_model_data.cpp/h    # CNN model weights
├── rasterize_stroke.cpp/h         # Stroke processing
└── FL/FL_Central/
    ├── Inference_Lite.h           # Classification head engine
    └── TrainedModel.h             # Trained weights
```

## Key Implementation

### Data Type Conversion (ConvertInt8ToFloat)

**Function:** Converts the int8 output of a CNN to the float[0,1] input required by the classification head.

**Method:** Min-Max Normalization

```cpp
1. Dequantize: real = (int8 - zero_point) × scale

2. Find min and max

3. Normalize: normalized = (real - min) / (max - min)
  }
}
```

### Network Configuration

Modify in `magic-wand-fl.ino`:

```cpp
// Number of classes
constexpr int label_count = 3;

// Class labels
const char* labels[label_count] = {"Class_0", "Class_1", "Class_2"};

// Network structure: Input → Hidden → Output
const unsigned int NN_DEF[] = {64, 20, 3};
```

## Performance

- **Inference Time**: ~200ms per gesture
  - CNN: ~100-200ms
  - Classification Head: ~5-10ms
  - Data Conversion: <1ms
- **Memory Usage**:
  - Flash: ~150KB
  - SRAM: ~35KB
- **Accuracy**: 94% (testset on classification head)

## Troubleshooting

### Compilation Errors

**Error: `Library Version Mismatch`**
- Solution: Remove `Harvard_TinyMLx` library, keep only `Arduino_TensorFlowLite`


### Runtime Issues

**No Serial Output**
- Check baud rate is set to 96000
- Verify correct port selection

**Low Accuracy**
- Ensure gestures are performed clearly
- Check feature values are in [0, 1] range
- Retrain classification head if needed


## Technical Details

### Data Flow Validation

| Stage | Type | Dimension | Range |
|-------|------|-----------|-------|
| CNN Output | int8 | 64 | [-128, 127] |
| Dequantized | float | 64 | ~[-1, 1] |
| Normalized | float | 64 | [0, 1] |
| Head Input | float | 64 | [0, 1] |
| Head Output | float | 3 | [0, 1] |

### Memory Breakdown

**Flash (Program Storage)**
- TFLite Model: ~80KB
- TFLite Library: ~40KB
- Application Code: ~30KB

**SRAM (Runtime)**
- CNN Tensor Arena: 30KB
- Classification Head: ~5KB
- Feature Buffer: 256 bytes




