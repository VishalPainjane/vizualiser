<div align="center">

<img src="docs/preview.png" width="100%" alt="Vizualiser - 3D Neural Network Architecture Visualization">

<br />
<br />
<!--
[![Featured on Product Hunt](https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=346331&theme=light)](https://www.producthunt.com/posts/vizualiser)
[![Top Post Badge](https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=346331&theme=light&period=daily)](https://www.producthunt.com/posts/vizualiser)
-->
<br />

[![GitHub stars](https://img.shields.io/github/stars/VishalPainjane/vizualiser?style=flat-square&logo=github&color=yellow)](https://github.com/VishalPainjane/vizualiser/stargazers)
<!-- [![GitHub forks](https://img.shields.io/github/forks/VishalPainjane/vizualiser?style=flat-square&logo=github&color=blue)](https://github.com/VishalPainjane/vizualiser/network/members) -->
[![GitHub watchers](https://img.shields.io/github/watchers/VishalPainjane/vizualiser?style=flat-square&logo=github)](https://github.com/VishalPainjane/vizualiser/watchers)
[![GitHub contributors](https://img.shields.io/github/contributors/VishalPainjane/vizualiser?color=blue&style=flat-square&logo=github)](https://github.com/VishalPainjane/vizualiser/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/VishalPainjane/vizualiser?color=blue&style=flat-square&logo=github)](https://github.com/VishalPainjane/vizualiser/commits/main)
[![GitHub license](https://img.shields.io/github/license/VishalPainjane/vizualiser?color=blue&style=flat-square)](LICENSE)
[![GitHub closed issues](https://img.shields.io/github/issues-closed/VishalPainjane/vizualiser?color=blue&style=flat-square&logo=github)](https://github.com/VishalPainjane/vizualiser/issues?q=is%3Aissue+is%3Aclosed)
[![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/VishalPainjane/vizualiser?color=blue&style=flat-square&logo=github)](https://github.com/VishalPainjane/vizualiser/pulls?q=is%3Apr+is%3Aclosed)
<!-- 
[![npm version](https://img.shields.io/npm/v/vizualiser?style=flat-square&logo=npm&color=red)](https://www.npmjs.com/package/vizualiser)
[![npm downloads](https://img.shields.io/npm/dm/vizualiser?style=flat-square&logo=npm&color=red)](https://www.npmjs.com/package/vizualiser)
[![Docker Pulls](https://img.shields.io/docker/pulls/vizualiser/vizualiser?style=flat-square&logo=docker&color=blue)](https://hub.docker.com/r/vizualiser/vizualiser)
[![Build Status](https://img.shields.io/github/actions/workflow/status/VishalPainjane/vizualiser/ci.yml?style=flat-square&logo=github-actions)](https://github.com/VishalPainjane/vizualiser/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/VishalPainjane/vizualiser?style=flat-square&logo=codecov)](https://codecov.io/gh/VishalPainjane/vizualiser)
[![Code Quality](https://img.shields.io/codacy/grade/abc123?style=flat-square&logo=codacy)](https://www.codacy.com/app/vizualiser/vizualiser)

[![Discord](https://img.shields.io/discord/123456789?style=flat-square&logo=discord&label=Discord&color=7289da)](https://discord.gg/vizualiser)
[![Twitter Follow](https://img.shields.io/twitter/follow/vizualiser?style=flat-square&logo=twitter&color=1DA1F2)](https://twitter.com/vizualiser)
[![YouTube Channel Subscribers](https://img.shields.io/youtube/channel/subscribers/UC123?style=flat-square&logo=youtube&color=red)](https://youtube.com/@vizualiser)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/vizualiser?style=flat-square&logo=reddit&color=FF4500)](https://reddit.com/r/vizualiser) -->

</div>

<br />

## 3D Neural Network Architecture Visualizer

Vizualiser is an open-source platform for visualizing, analyzing, and understanding deep learning architectures in stunning 3D. This is a professional tool for inspecting model structure, layer parametrization, and inter-layer connectivity. It focuses on clarity, accuracy, and performance rendering large architectures while keeping interactive frame rates on modern hardware.

<div align="center">

**[Live Demo](https://vizualiser.ai/demo)** |
**[Documentation](https://docs.vizualiser.ai)** |
**[Gallery](https://vizualiser.ai/gallery)** |
**[API Reference](https://api.vizualiser.ai)** |
**[Discord Community](https://discord.gg/vizualiser)**

</div>

---

## 🌟 Why Vizualiser?

- **Blazing fast** - GPU-accelerated, real-time rendering for massive models
- **Plug & Play** - Drag and drop, or integrate with your ML pipeline
- **Open ecosystem** - Exporters, plugins, and a thriving community
- **Beautiful** - Publication-ready visuals and dark/light themes

---

## 📦 Supported Model Formats

| Format          | Extension                    | Support       | Notes                                |
| --------------- | ---------------------------- | ------------- | ------------------------------------ |
| **NN3D**        | `.nn3d`, `.json`             | ✅ Full       | Native JSON format                   |
| **ONNX**        | `.onnx`                      | ✅ Full       | Parsed directly in browser           |
| **SafeTensors** | `.safetensors`               | ✅ Full       | LLM weights with structure inference |
| **PyTorch**     | `.pt`, `.pth`, `.ckpt`       | ⚠️ Conversion | Use Python exporter                  |
| **Keras/TF**    | `.h5`, `.hdf5`               | ⚠️ Conversion | Convert to ONNX first                |
| **Binary**      | `.bin`, `.weights`           | ⚠️ Conversion | Weights only, needs model code       |
| **Pickle**      | `.pkl`, `.pickle`, `.joblib` | ⚠️ Conversion | Use Python exporter                  |

## ✨ Features

### 🎨 3D Visualization

- **Layer Geometry**: Each layer type renders as a distinct 3D shape (boxes, spheres, custom geometry)
- **Connection Rendering**: Multiple edge styles (lines, bezier curves, 3D tubes, arrows)
- **Color Coding**: Automatic color assignment based on layer categories
- **Level of Detail (LOD)**: Optimized rendering for large networks

### 🔄 Interactive Navigation

- **Orbit Controls**: Click and drag to rotate the view
- **Zoom**: Scroll wheel to zoom in/out
- **Pan**: Right-click and drag to pan
- **Selection**: Click layers to view detailed information
- **Hover**: Hover over layers for quick tooltips

### 📐 Layout Algorithms

- **Layered**: Traditional left-to-right topological layout
- **Force-Directed**: Physics-based spring simulation
- **Circular**: Nodes arranged in a circle
- **Hierarchical**: Tree-like arrangement based on network depth

### 🛠️ Tools & Controls

- **File Upload**: Drag & drop or click to load model files
- **Layout Selector**: Switch between layout algorithms
- **Edge Styles**: Toggle between line, bezier, tube, and arrow styles
- **Label Toggle**: Show/hide layer labels
- **Edge Toggle**: Show/hide connections
- **Keyboard Shortcuts**: Quick access to common actions

---

## ⚡ Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn
- Python 3.11+ (for backend)

### Quick Start (Docker Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/vizualiser.git
cd vizualiser

# Start with Docker Compose
docker-compose up --build

# Or use the convenience scripts:
# Windows: docker-start.bat
# Linux/Mac: ./docker-start.sh
```

Access the application:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

```bash
# Stop containers
docker-compose down

# View logs
docker-compose logs -f
```

### Manual Installation (Development)

```bash
# Clone the repository
git clone https://github.com/your-username/vizualiser.git
cd vizualiser

# Install frontend dependencies
npm install

# Start frontend development server
npm run dev

# In another terminal, start the backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open [http://localhost:5173](http://localhost:5173) to view the visualizer.

### Production Build

```bash
npm run build
npm run preview
```

---

## 🕹️ Usage

### Loading Models

1. **Drag & Drop**: Drag any supported model file onto the drop zone
2. **Click to Upload**: Click the drop zone and select a file
3. **Sample Models**: Try the included samples in `samples/`

**Directly Supported Formats:**

- `.nn3d` / `.json` - Native format (full structure)
- `.onnx` - ONNX models (parsed in browser)
- `.safetensors` - SafeTensors (structure inferred from weights)

**Formats Requiring Conversion:**

- `.pt` / `.pth` / `.ckpt` - Use Python exporter
- `.h5` / `.hdf5` - Convert to ONNX first

### Keyboard Shortcuts

| Key   | Action                                                         |
| ----- | -------------------------------------------------------------- |
| `1-4` | Switch layout (1=Layered, 2=Force, 3=Circular, 4=Hierarchical) |
| `L`   | Toggle labels                                                  |
| `E`   | Toggle edges                                                   |
| `Esc` | Deselect layer                                                 |
| `R`   | Reset camera view                                              |

### Navigation

- **Left Mouse + Drag**: Rotate view
- **Right Mouse + Drag**: Pan view
- **Scroll Wheel**: Zoom in/out
- **Left Click on Layer**: Select and view details
- **Hover on Layer**: Show tooltip

---

## 📄 .nn3d File Format

The `.nn3d` format is a JSON-based schema for describing neural network architectures:

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "My Model",
    "framework": "pytorch",
    "totalParams": 1000000
  },
  "graph": {
    "nodes": [
      {
        "id": "layer1",
        "type": "conv2d",
        "name": "Conv Layer",
        "params": { "inChannels": 3, "outChannels": 64 },
        "outputShape": [1, 64, 224, 224]
      }
    ],
    "edges": [{ "source": "input", "target": "layer1" }]
  },
  "visualization": {
    "layout": "layered",
    "theme": "dark"
  }
}
```

### Supported Layer Types

| Category           | Types                                                    |
| ------------------ | -------------------------------------------------------- |
| **Input/Output**   | input, output                                            |
| **Convolution**    | conv1d, conv2d, conv3d, convTranspose2d, depthwiseConv2d |
| **Linear**         | linear, dense                                            |
| **Activation**     | relu, gelu, sigmoid, tanh, softmax, leakyRelu, swish     |
| **Normalization**  | batchNorm, layerNorm, groupNorm, instanceNorm            |
| **Pooling**        | maxPool2d, avgPool2d, globalAvgPool, adaptiveAvgPool2d   |
| **Attention**      | multiHeadAttention, selfAttention, crossAttention        |
| **Recurrent**      | lstm, gru, rnn                                           |
| **Regularization** | dropout, dropPath                                        |
| **Operations**     | add, concat, multiply, split, reshape, flatten, permute  |
| **Embedding**      | embedding, positionalEncoding                            |

---

## 🐍 Python Exporters

Export PyTorch and ONNX models to `.nn3d` format:

### Installation

```bash
cd exporters/python
pip install -e .
```

### PyTorch Export

```python
from nn3d_exporter import PyTorchExporter
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

exporter = PyTorchExporter(model, "My Model")
exporter.trace(torch.randn(1, 784))
exporter.save("model.nn3d")
```

### ONNX Export

```python
from nn3d_exporter import ONNXExporter

exporter = ONNXExporter.from_file("model.onnx", "My Model")
exporter.save("model.nn3d")
```

---

## 🎓 Sample Models

The `samples/` directory includes example models:

| File                       | Description                              |
| -------------------------- | ---------------------------------------- |
| `simple_mlp.nn3d`          | Basic MLP for MNIST classification       |
| `cnn_resnet.nn3d`          | ResNet-style CNN with skip connections   |
| `transformer_encoder.nn3d` | Transformer encoder block with attention |

---

## 🛠️ Tech Stack

- **React 18** - UI framework
- **Three.js** - 3D graphics via @react-three/fiber
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Zustand** - State management
- **AJV** - JSON Schema validation

---

## 📁 Project Structure

```
vizualiser/
├── src/
│   ├── schema/          # .nn3d schema & types
│   ├── core/            # State management & algorithms
│   ├── components/
│   │   ├── layers/      # 3D layer geometry
│   │   ├── edges/       # Connection rendering
│   │   ├── controls/    # Camera & interaction
│   │   └── ui/          # UI overlays
│   ├── App.tsx
│   └── main.tsx
├── exporters/
│   └── python/          # Python export package
├── samples/             # Example .nn3d files
└── public/
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🌍 Community & Support

- [Discussions](https://github.com/VishalPainjane/vizualiser/discussions)
- [Discord](https://discord.gg/xyz123)
- [Twitter](https://twitter.com/vizualiser)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/vizualiser)
- [Contributing Guide](CONTRIBUTING.md)

## ❤️ Sponsors

<p align="center">
  <a href="https://github.com/sponsors/VishalPainjane"><img src="https://img.shields.io/badge/sponsor%20us-on%20GitHub-orange?style=for-the-badge" alt="Sponsor"></a>
</p>

---

<p align="center">
  <b>Made with ❤️ by hundreds of contributors worldwide.</b>
</p>

- [Three.js](https://threejs.org/) for 3D rendering
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber/) for React integration
- [Drei](https://github.com/pmndrs/drei) for useful 3D helpers



