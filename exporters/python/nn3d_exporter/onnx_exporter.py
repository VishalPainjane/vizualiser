"""
ONNX Model Exporter

Export ONNX models to .nn3d format for visualization.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json

try:
    import onnx
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .schema import (
    NN3DModel, NN3DGraph, NN3DNode, NN3DEdge, NN3DMetadata,
    NN3DSubgraph, LayerParams, LayerType, VisualizationConfig
)


# Mapping from ONNX op types to NN3D layer types
ONNX_TO_NN3D_TYPE: Dict[str, str] = {
    # Convolution
    'Conv': LayerType.CONV2D.value,
    'ConvTranspose': LayerType.CONV_TRANSPOSE_2D.value,
    
    # Linear
    'Gemm': LayerType.LINEAR.value,
    'MatMul': LayerType.LINEAR.value,
    
    # Normalization
    'BatchNormalization': LayerType.BATCH_NORM_2D.value,
    'LayerNormalization': LayerType.LAYER_NORM.value,
    'InstanceNormalization': LayerType.INSTANCE_NORM.value,
    'GroupNormalization': LayerType.GROUP_NORM.value,
    'Dropout': LayerType.DROPOUT.value,
    
    # Activations
    'Relu': LayerType.RELU.value,
    'LeakyRelu': LayerType.LEAKY_RELU.value,
    'Sigmoid': LayerType.SIGMOID.value,
    'Tanh': LayerType.TANH.value,
    'Softmax': LayerType.SOFTMAX.value,
    'Gelu': LayerType.GELU.value,
    
    # Pooling
    'MaxPool': LayerType.MAX_POOL_2D.value,
    'AveragePool': LayerType.AVG_POOL_2D.value,
    'GlobalAveragePool': LayerType.GLOBAL_AVG_POOL.value,
    'GlobalMaxPool': LayerType.MAX_POOL_2D.value,
    
    # Shape operations
    'Flatten': LayerType.FLATTEN.value,
    'Reshape': LayerType.RESHAPE.value,
    'Transpose': LayerType.RESHAPE.value,
    'Squeeze': LayerType.RESHAPE.value,
    'Unsqueeze': LayerType.RESHAPE.value,
    
    # Merge operations
    'Concat': LayerType.CONCAT.value,
    'Add': LayerType.ADD.value,
    'Mul': LayerType.MULTIPLY.value,
    'Split': LayerType.SPLIT.value,
    
    # Attention
    'Attention': LayerType.ATTENTION.value,
    'MultiHeadAttention': LayerType.MULTI_HEAD_ATTENTION.value,
    
    # Recurrent
    'LSTM': LayerType.LSTM.value,
    'GRU': LayerType.GRU.value,
    'RNN': LayerType.RNN.value,
    
    # Resize/Upsample
    'Resize': LayerType.UPSAMPLE.value,
    'Upsample': LayerType.UPSAMPLE.value,
    
    # Padding
    'Pad': LayerType.PAD.value,
}


class ONNXExporter:
    """
    Export ONNX models to NN3D format.
    
    Usage:
        exporter = ONNXExporter("model.onnx")
        nn3d_model = exporter.export()
        nn3d_model.save("model.nn3d")
    """
    
    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the exporter.
        
        Args:
            model_path: Path to ONNX model file
            model_name: Name for the model (defaults to filename)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is required. Install with: pip install onnx")
        
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.model_name = model_name or model_path.split('/')[-1].replace('.onnx', '')
        
        # Initialize graph
        onnx.checker.check_model(self.model)
        self.graph = self.model.graph
        
        self.nodes: List[NN3DNode] = []
        self.edges: List[NN3DEdge] = []
        
        self._tensor_shapes: Dict[str, List[int]] = {}
        self._value_info: Dict[str, Any] = {}
        self._node_id_map: Dict[str, str] = {}
        self._output_to_node: Dict[str, str] = {}
    
    def _extract_shapes(self) -> None:
        """Extract tensor shapes from the model"""
        # Input shapes
        for input_info in self.graph.input:
            shape = []
            if input_info.type.tensor_type.HasField('shape'):
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        shape.append(dim.dim_value)
                    elif dim.HasField('dim_param'):
                        shape.append(dim.dim_param)
                    else:
                        shape.append(-1)
            self._tensor_shapes[input_info.name] = shape
            self._value_info[input_info.name] = input_info
        
        # Output shapes
        for output_info in self.graph.output:
            shape = []
            if output_info.type.tensor_type.HasField('shape'):
                for dim in output_info.type.tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        shape.append(dim.dim_value)
                    elif dim.HasField('dim_param'):
                        shape.append(dim.dim_param)
                    else:
                        shape.append(-1)
            self._tensor_shapes[output_info.name] = shape
            self._value_info[output_info.name] = output_info
        
        # Value info (intermediate tensors)
        for value_info in self.graph.value_info:
            shape = []
            if value_info.type.tensor_type.HasField('shape'):
                for dim in value_info.type.tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        shape.append(dim.dim_value)
                    elif dim.HasField('dim_param'):
                        shape.append(dim.dim_param)
                    else:
                        shape.append(-1)
            self._tensor_shapes[value_info.name] = shape
            self._value_info[value_info.name] = value_info
    
    def _get_layer_type(self, op_type: str) -> str:
        """Map ONNX op type to NN3D layer type"""
        return ONNX_TO_NN3D_TYPE.get(op_type, LayerType.CUSTOM.value)
    
    def _extract_attributes(self, node) -> Dict[str, Any]:
        """Extract node attributes"""
        attrs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode('utf-8')
            elif attr.type == onnx.AttributeProto.STRINGS:
                attrs[attr.name] = [s.decode('utf-8') for s in attr.strings]
        return attrs
    
    def _extract_params(self, node, attrs: Dict[str, Any]) -> LayerParams:
        """Extract layer parameters from ONNX node"""
        params = LayerParams()
        
        # Convolution parameters
        if 'kernel_shape' in attrs:
            params.kernel_size = attrs['kernel_shape']
        if 'strides' in attrs:
            params.stride = attrs['strides']
        if 'pads' in attrs:
            params.padding = attrs['pads']
        if 'dilations' in attrs:
            params.dilation = attrs['dilations']
        if 'group' in attrs:
            params.groups = attrs['group']
        
        # Normalization parameters
        if 'epsilon' in attrs:
            params.eps = attrs['epsilon']
        if 'momentum' in attrs:
            params.momentum = attrs['momentum']
        
        # Dropout
        if 'ratio' in attrs:
            params.dropout_rate = attrs['ratio']
        
        return params
    
    def export(self) -> NN3DModel:
        """Export the ONNX model to NN3D format"""
        
        # Extract tensor shapes
        self._extract_shapes()
        
        # Add input nodes
        for idx, input_info in enumerate(self.graph.input):
            # Skip initializers (weights)
            if input_info.name in [init.name for init in self.graph.initializer]:
                continue
            
            node_id = f"input_{idx}"
            shape = self._tensor_shapes.get(input_info.name, [])
            
            input_node = NN3DNode(
                id=node_id,
                type=LayerType.INPUT.value,
                name=input_info.name,
                output_shape=shape if shape else None,
                depth=0
            )
            self.nodes.append(input_node)
            self._output_to_node[input_info.name] = node_id
        
        # Process all operator nodes
        for idx, node in enumerate(self.graph.node):
            node_id = f"node_{idx}"
            layer_type = self._get_layer_type(node.op_type)
            attrs = self._extract_attributes(node)
            params = self._extract_params(node, attrs)
            
            # Get input/output shapes
            input_shapes = [self._tensor_shapes.get(inp, []) for inp in node.input 
                          if inp not in [init.name for init in self.graph.initializer]]
            output_shapes = [self._tensor_shapes.get(out, []) for out in node.output]
            
            input_shape = input_shapes[0] if input_shapes else None
            output_shape = output_shapes[0] if output_shapes else None
            
            # Create node
            nn3d_node = NN3DNode(
                id=node_id,
                type=layer_type,
                name=node.name or f"{node.op_type}_{idx}",
                params=params if any(v is not None for v in [
                    params.kernel_size, params.stride, params.padding,
                    params.eps, params.dropout_rate
                ]) else None,
                input_shape=input_shape if input_shape else None,
                output_shape=output_shape if output_shape else None,
                depth=idx + 1,
                attributes={'op_type': node.op_type, **attrs} if attrs else {'op_type': node.op_type}
            )
            self.nodes.append(nn3d_node)
            
            # Map outputs to this node
            for output in node.output:
                self._output_to_node[output] = node_id
            
            # Create edges from inputs
            for inp in node.input:
                # Skip initializers (weights)
                if inp in [init.name for init in self.graph.initializer]:
                    continue
                
                source_id = self._output_to_node.get(inp)
                if source_id:
                    edge = NN3DEdge(
                        source=source_id,
                        target=node_id,
                        tensor_shape=self._tensor_shapes.get(inp, None),
                        label=inp
                    )
                    self.edges.append(edge)
        
        # Add output nodes
        for idx, output_info in enumerate(self.graph.output):
            node_id = f"output_{idx}"
            shape = self._tensor_shapes.get(output_info.name, [])
            
            output_node = NN3DNode(
                id=node_id,
                type=LayerType.OUTPUT.value,
                name=output_info.name,
                input_shape=shape if shape else None,
                depth=len(self.graph.node) + 1
            )
            self.nodes.append(output_node)
            
            # Create edge from last node producing this output
            source_id = self._output_to_node.get(output_info.name)
            if source_id:
                edge = NN3DEdge(
                    source=source_id,
                    target=node_id,
                    tensor_shape=shape if shape else None
                )
                self.edges.append(edge)
        
        # Get input/output shapes for metadata
        input_shapes = [self._tensor_shapes.get(inp.name) for inp in self.graph.input 
                       if inp.name not in [init.name for init in self.graph.initializer]]
        output_shapes = [self._tensor_shapes.get(out.name) for out in self.graph.output]
        
        # Create metadata
        metadata = NN3DMetadata(
            name=self.model_name,
            framework="onnx",
            created=datetime.now().isoformat(),
            input_shape=input_shapes[0] if input_shapes else None,
            output_shape=output_shapes[0] if output_shapes else None,
            description=f"Converted from ONNX model: {self.model_path}"
        )
        
        # Create graph
        graph = NN3DGraph(
            nodes=self.nodes,
            edges=self.edges,
        )
        
        # Create visualization config
        viz_config = VisualizationConfig()
        
        return NN3DModel(
            metadata=metadata,
            graph=graph,
            visualization=viz_config
        )


def export_onnx_model(
    model_path: str,
    output_path: str,
    model_name: Optional[str] = None,
) -> NN3DModel:
    """
    Convenience function to export an ONNX model to .nn3d file.
    
    Args:
        model_path: Path to ONNX model file
        output_path: Path to save the .nn3d file
        model_name: Name for the model
        
    Returns:
        The exported NN3DModel
    """
    exporter = ONNXExporter(model_path, model_name)
    nn3d_model = exporter.export()
    nn3d_model.save(output_path)
    return nn3d_model
