import onnx
from onnx import version_converter
from onnx import shape_inference

# Load the model
model_path = "wespeaker_en_voxceleb_CAM++.onnx"
original_model = onnx.load(model_path)

# Try to convert to Opset 19
try:
    converted_model = version_converter.convert_version(original_model, 19)
    onnx.save(converted_model, "wespeaker_en_voxceleb_CAM++_opset19.onnx")
    print("Conversion successful! Saved as wespeaker_en_voxceleb_CAM++_opset19.onnx")
except Exception as e:
    print(f"Conversion failed: {e}")
    print("Note: Automatic conversion often fails for complex operators. You may need Option 1.")


# Load your ORIGINAL Opset 14 model
input_model = "wespeaker_en_voxceleb_CAM++.onnx"
output_model = "wespeaker_patched_opset14.onnx"

model = onnx.load(input_model)
print(f"Loaded {input_model}")

modified = False

# Iterate through all nodes to find AveragePool
for node in model.graph.node:
    if node.op_type == "AveragePool":
        # Look for the 'ceil_mode' attribute
        for i, attr in enumerate(node.attribute):
            if attr.name == "ceil_mode":
                print(f"Found ceil_mode in node: {node.name}")
                
                # OPTION A: Remove it entirely (Burn might assume floor, which is standard)
                # node.attribute.remove(attr)
                
                # OPTION B: Force it to 0 (False), which fits Opset 14 requirements usually
                print(f"Changing ceil_mode from {attr.i} to 0")
                attr.i = 0
                modified = True

if modified:
    onnx.save(model, output_model)
    print(f"Success! Saved patched model to {output_model}")
    print("Try importing THIS file into Burn.")
else:
    print("No AveragePool nodes with ceil_mode found. Are you sure this is the Opset 14 file?")

import onnx

# Load the PATCHED Opset 14 model (the one with ceil_mode removed)
input_path = "wespeaker_patched_opset14.onnx"
output_path = "wespeaker_burn_ready.onnx"

print(f"Loading {input_path}...")
model = onnx.load(input_path)

# 1. Inspect and Fix Input Shape
# WeSpeaker/CAM++ usually expects [Batch, Sequence, Features] or [Batch, Features, Sequence]
# Typical Fbank dim is 80. Let's assume a standard input of 1 sample, 200 frames, 80 dims.
# If your model expects [Batch, Dim, Time], use [1, 80, 200]. 
# If [Batch, Time, Dim], use [1, 200, 80].
# We will define a concrete shape to force rank calculation.
CONCRETE_SHAPE = [1, 200, 80] 

input_tensor = model.graph.input[0]
print(f"Found input tensor: {input_tensor.name}")

# Clear existing dynamic dims
input_tensor.type.tensor_type.shape.dim.clear()

# Apply concrete dims
for dim_val in CONCRETE_SHAPE:
    d = input_tensor.type.tensor_type.shape.dim.add()
    d.dim_value = dim_val

print(f"Set input '{input_tensor.name}' shape to {CONCRETE_SHAPE}")

# 2. Run Shape Inference with Data Propagation
# This propagates the [1, 200, 80] shape through the graph.
# The 'Expand' node will now see a concrete target shape and calculate its output rank.
print("Running shape inference...")
inferred_model = shape_inference.infer_shapes(model, data_prop=True, strict_mode=True)

# 3. Save
onnx.save(inferred_model, output_path)
print(f"Saved to {output_path}")
print("Try importing THIS file into Burn.")

# uv run onnxsim wespeaker_burn_ready.onnx wespeaker_final.onnx
