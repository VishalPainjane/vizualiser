# NN3D Exporter

Python library to export neural network models to `.nn3d` format for 3D visualization.

## Installation

```bash
# Basic installation
pip install nn3d-exporter

# With PyTorch support
pip install nn3d-exporter[pytorch]

# With ONNX support
pip install nn3d-exporter[onnx]

# With all frameworks
pip install nn3d-exporter[all]
```

## Quick Start

### Export PyTorch Model

```python
import torch.nn as nn
from nn3d_exporter import export_pytorch_model

# Define your model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 56 * 56, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()

# Export to .nn3d format
export_pytorch_model(
    model,
    output_path="simple_cnn.nn3d",
    input_shape=(1, 3, 224, 224),
    model_name="Simple CNN"
)
```

### Export ONNX Model

```python
from nn3d_exporter import export_onnx_model

# Export an existing ONNX model
export_onnx_model(
    model_path="resnet50.onnx",
    output_path="resnet50.nn3d",
    model_name="ResNet-50"
)
```

### Using the Exporter Classes

For more control, use the exporter classes directly:

```python
from nn3d_exporter import PyTorchExporter

exporter = PyTorchExporter(
    model=model,
    input_shape=(1, 3, 224, 224),
    model_name="My Model"
)

# Export to NN3DModel object
nn3d_model = exporter.export()

# Customize before saving
nn3d_model.visualization.theme = "blueprint"
nn3d_model.visualization.layout = "force"

# Save to file
nn3d_model.save("my_model.nn3d")

# Or get JSON string
json_str = nn3d_model.to_json()
```

## Supported Frameworks

### PyTorch

- All standard `torch.nn` layers
- Custom modules (exported as "custom" type)
- Automatic shape inference via forward pass
- Parameter counting

### ONNX

- Standard ONNX operators
- Shape extraction from model metadata
- Operator attributes preserved

## API Reference

### `export_pytorch_model(model, output_path, input_shape=None, model_name=None)`

Export a PyTorch model to .nn3d file.

| Parameter     | Type        | Description                   |
| ------------- | ----------- | ----------------------------- |
| `model`       | `nn.Module` | PyTorch model to export       |
| `output_path` | `str`       | Path for output .nn3d file    |
| `input_shape` | `tuple`     | Input tensor shape (optional) |
| `model_name`  | `str`       | Model name (optional)         |

### `export_onnx_model(model_path, output_path, model_name=None)`

Export an ONNX model to .nn3d file.

| Parameter     | Type  | Description                |
| ------------- | ----- | -------------------------- |
| `model_path`  | `str` | Path to ONNX model file    |
| `output_path` | `str` | Path for output .nn3d file |
| `model_name`  | `str` | Model name (optional)      |

## License

MIT License
