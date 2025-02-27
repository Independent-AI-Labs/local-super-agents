import json
import os

import onnx
import tensorflow as tf
import tf2onnx
from onnx2pytorch import ConvertModel
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, Qwen2VLForConditionalGeneration

from integration.data.config import LLAMACPP_WORKING_DIR
from integration.util.misc_util import run_subprocess_command
from quantbench.qb_config import CONVERT_HF_TO_GGUF_SCRIPT, INPUT_MODEL_FILE_EXTENSION


def convert_to_f16_gguf(input_dir_val, output_dir_val):
    convert_hf_to_gguf_path = os.path.join(LLAMACPP_WORKING_DIR, CONVERT_HF_TO_GGUF_SCRIPT)
    command_f16 = ["python", convert_hf_to_gguf_path, input_dir_val, "--outfile", output_dir_val]
    output_console_text = f"Running command: {' '.join(command_f16)}\n"
    ret_code, cmd_output = run_subprocess_command(command_f16)
    output_console_text += cmd_output
    return ret_code, output_console_text


def convert_tf_to_onnx(input_dir, output_path, fold_const=False, opset=11):
    """
    Converts a TensorFlow model to ONNX format, with options to adjust settings.

    Args:
        input_dir (str): Path to the directory containing the TensorFlow model.
        output_path (str): Path to save the ONNX model.
        fold_const (bool): Whether to fold constants during conversion.
        opset (int): The ONNX opset version to use.

    Returns:
        tuple: Exit code and output text.
    """
    try:
        model_path = None
        for filename in os.listdir(input_dir):
            if filename.endswith(INPUT_MODEL_FILE_EXTENSION):
                model_path = os.path.join(input_dir, filename)
                break
        if model_path is None:
            raise ValueError("No .safetensors files found in the input directory")

        # Load safetensors to extract tensor information and derive input shape.
        with safe_open(model_path, framework="pt", device="cpu") as f:
            first_tensor_name = next(iter(f.keys()))
            tensor_info = f.get_tensor(first_tensor_name)
            input_shape = (1,) + tuple(tensor_info.shape)

        # Create the model using the provided helper.
        model = create_model(input_shape, input_dir)

        # Set up conversion options.
        conversion_options = {"opset": opset}
        if fold_const:
            conversion_options["fold_const"] = True

        output_console_text = f"TF to ONNX Conversion Options: opset {opset}\n"

        if hasattr(model, 'layers'):
            # For Keras-based models.
            spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
            if isinstance(model, tf.keras.Sequential):
                # Direct conversion for Sequential models.
                model_proto, _ = tf2onnx.convert.from_keras(
                    model, input_signature=spec, output_path=output_path, opset=opset
                )
            else:
                # Wrap the model call to force conversion of inputs.
                def wrapped_call(*args, **kwargs):
                    # Infer input types from model layers.
                    input_types = []
                    for layer in model.layers:
                        if isinstance(layer, tf.keras.layers.InputLayer):
                            input_types.append(layer.dtype)

                    # if no layers return a dummy type.
                    if not input_types:
                        input_types.append(tf.float32)

                    # Convert inputs to concrete tensors with inferred types.
                    def convert_to_tensor_with_type(tensor, dtype):
                        return tf.convert_to_tensor(tensor, dtype=dtype)

                    converted_args = tf.nest.map_structure(convert_to_tensor_with_type, args, [input_types[0]])

                    return model.call(*converted_args, **kwargs)

                concrete_func = tf.function(wrapped_call).get_concrete_function(*spec)
                model_proto, _ = tf2onnx.convert.from_function(
                    concrete_func, input_signature=spec, output_path=output_path, **conversion_options
                )
        elif isinstance(model, Qwen2VLForConditionalGeneration):
            # For non-Keras models, such as a transformers-based model.
            # Infer input types from model layers.
            input_types = []
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    input_types.append(layer.dtype)

            # Define the input specifications based on the inferred types.
            spec = []
            if len(input_types) >= 3:
                spec.append(tf.TensorSpec((None, None, 3, 224, 224), input_types[0], name="pixel_values"))
                spec.append(tf.TensorSpec((None, None), input_types[1], name="input_ids"))
                spec.append(tf.TensorSpec((None, None), input_types[2], name="attention_mask"))

            if spec:
                def wrapped_call(*args, **kwargs):
                    # Convert inputs to concrete tensors with inferred types.
                    def convert_to_tensor_with_type(tensor, dtype):
                        return tf.convert_to_tensor(tensor, dtype=dtype)

                    converted_args = tf.nest.map_structure(convert_to_tensor_with_type, args, [input_types[0]] if len(input_types) > 0 else [])
                    return model.call(*converted_args, **kwargs)

                concrete_func = tf.function(wrapped_call).get_concrete_function(*spec)
                model_proto, _ = tf2onnx.convert.from_function(
                    concrete_func, input_signature=spec, output_path=output_path, **conversion_options
                )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        return 0, output_console_text

    except Exception as e:
        output_console_text = f"Error during TF to ONNX conversion: {e}\n"
        return 1, output_console_text


# Optionally, if you need a custom class (like Qwen2VLForConditionalGeneration) import it here
# from your_custom_library import Qwen2VLForConditionalGeneration

def create_model(input_shape, input_dir):
    """
    Create a model based on configuration in the input directory.

    If a config.json is present and indicates a transformer/VLM model,
    use the transformers library to load it accordingly.
    Otherwise, build a dummy TensorFlow Keras model based on input_shape.

    Args:
        input_shape (tuple): Shape of the input tensor.
        input_dir (str): Directory containing the model configuration (config.json) and weights.

    Returns:
        A transformer model (e.g., PreTrainedModel) if config indicates a VLM,
        otherwise a tf.keras.Model.
    """
    config_file = os.path.join(input_dir, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config_data = json.load(f)

        # Determine the model type from the config.
        model_type = config_data.get("model_type", "").lower()
        architectures = config_data.get("architectures", [])
        auto_map = config_data.get("auto_map", {})

        # If the config indicates a transformer/VLM model, load it accordingly.
        if architectures or auto_map:
            # Create a transformers AutoConfig instance from the directory.
            auto_config = AutoConfig.from_pretrained(input_dir)

            # For example, if the config auto_map has an entry for causal language modeling:
            if "AutoModelForCausalLM" in auto_map:
                model = AutoModelForCausalLM.from_pretrained(input_dir, config=auto_config)
                return model
            if auto_config.model_type == "qwen2_5_vl":
                auto_config.architectures[0] = 'Qwen2_VLForConditionalGeneration'
                auto_config.model_type = "qwen2_vl"
            # Otherwise, try the generic AutoModel
            model = AutoModel.from_pretrained(input_dir, config=auto_config)
            if auto_config.model_type == "qwen2_vl":
                pass
            return model

    # Fallback: create a dummy TensorFlow Keras model based on input_shape.
    num_outputs = input_shape[0] if len(input_shape) > 0 else 1

    if len(input_shape) == 1:
        # Dense model for 1D input.
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Dense(num_outputs, activation='relu', name='output')
        ])
    elif len(input_shape) == 2:
        # LSTM model for 2D input.
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LSTM(units=num_outputs, return_sequences=True),
            tf.keras.layers.Dense(num_outputs, activation='relu', name='output')
        ])
    elif len(input_shape) == 3:
        # Conv2D model for 3D input (e.g., image data).
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_outputs, activation='relu', name='output')
        ])
    else:
        # Default to a simple Dense model.
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Dense(num_outputs, activation='relu', name='output')
        ])

    return model


def convert_onnx_to_pytorch(onnx_path, pytorch_path):
    """
    Converts an ONNX model to a PyTorch model.

    Args:
        onnx_path (str): Path to the ONNX model.
        pytorch_path (str): Path to save the PyTorch model.

    Returns:
        tuple: Exit code and output text.
    """
    try:
        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)

        # Convert to PyTorch
        pytorch_model = ConvertModel(onnx_model)

        # Save the PyTorch model
        import torch
        torch.save(pytorch_model.state_dict(), pytorch_path)

        output_console_text = ""
        return 0, output_console_text
    except Exception as e:
        output_console_text = f"Error during ONNX to PyTorch conversion: {e}\n"
        return 1, output_console_text
