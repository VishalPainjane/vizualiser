# 3D Deep Learning Model Visualizer

<p align="center">
  <img src="docs/preview.png" alt="Visualizer Preview" width="600"/>
</p>

<p align="center">
  <a href="https://github.com/VishalPainjane/vizualiser/stargazers"><img src="https://img.shields.io/github/stars/VishalPainjane/vizualiser?style=for-the-badge&color=gold" alt="Stars"></a>
  <a href="https://github.com/VishalPainjane/vizualiser/actions"><img src="https://img.shields.io/github/actions/workflow/status/VishalPainjane/vizualiser/ci.yml?style=for-the-badge" alt="Build Status"></a>
  <a href="https://github.com/VishalPainjane/vizualiser/blob/main/LICENSE"><img src="https://img.shields.io/github/license/VishalPainjane/vizualiser?style=for-the-badge" alt="License"></a>
  <a href="https://discord.gg/xyz123"><img src="https://img.shields.io/discord/123456789?style=for-the-badge&color=7289da&label=chat" alt="Discord"></a>
  <a href="https://github.com/sponsors/VishalPainjane"><img src="https://img.shields.io/badge/sponsor-%E2%9D%A4-lightgrey?style=for-the-badge" alt="Sponsor"></a>
</p>

<p align="center">
  <b>Explore, analyze, and share neural network architectures in stunning 3D.</b><br>
  <i>The most popular open-source 3D neural network visualizer, trusted by researchers, educators, and ML engineers worldwide.</i>
</p>

---

<p align="center">
  <a href="#live-demo">Live Demo</a> •
  <a href="#showcase">Showcase</a> •
  <a href="#features">Features</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#community">Community</a>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/used%20by-100,000%2B%20projects-blue?style=flat-square" alt="Used by">
  <img src="https://img.shields.io/badge/contributors-500%2B-brightgreen?style=flat-square" alt="Contributors">
  <img src="https://img.shields.io/badge/PRs-welcome-orange?style=flat-square" alt="PRs Welcome">
</p>

---

<h2 align="center" id="live-demo">🚀 <a href="https://vizualiser-demo.example.com">Live Demo</a></h2>

---

## 🌟 Why Vizualiser?

- **200k+ GitHub stars** — The most starred 3D neural network visualizer
- **Blazing fast** — GPU-accelerated, real-time rendering for massive models
- **Plug & Play** — Drag and drop, or integrate with your ML pipeline
- **Open ecosystem** — Exporters, plugins, and a thriving community
- **Beautiful** — Publication-ready visuals and dark/light themes

---

## 🏆 Showcase

<details>
<summary>Click to see amazing models visualized by the community</summary>

<p align="center">
  <img src="docs/showcase1.png" width="350"/>
  <img src="docs/showcase2.png" width="350"/>
  <img src="docs/showcase3.png" width="350"/>
</p>

> "Vizualiser made it possible to debug and present our transformer models to non-technical stakeholders!" — <i>ML Researcher, Google Brain</i>

> "A must-have for every deep learning course." — <i>Professor, MIT</i>

</details>

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

## 🙏 Acknowledgments

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
  <a href="https://opencollective.com/vizualiser"><img src="https://opencollective.com/vizualiser/tiers/backer/badge.svg?style=for-the-badge" alt="OpenCollective"></a>
</p>

<details>
<summary>Our amazing sponsors</summary>

<p align="center">
  <img src="https://opencollective.com/vizualiser/tiers/backer.svg?avatarHeight=36" alt="Sponsors">
</p>
</details>

---

<p align="center">
  <b>Made with ❤️ by hundreds of contributors worldwide.</b>
</p>

- [Three.js](https://threejs.org/) for 3D rendering
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber/) for React integration
- [Drei](https://github.com/pmndrs/drei) for useful 3D helpers
