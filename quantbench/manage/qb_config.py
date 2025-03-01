from typing import List, Literal

from integration.data.config import INSTALL_PATH

# Constants
CONVERT_HF_TO_GGUF_SCRIPT = "convert_hf_to_gguf.py"
INPUT_MODEL_FILE_EXTENSION = ".safetensors"
GGUF_FILE_EXTENSION = ".gguf"
IMATRIX_FILE_NAME = "imatrix.dat"
IMATRIX_CUSTOM_FILE_NAME = "imatrix-custom.dat"
IMATRIX_DATASET_FILE = f"{INSTALL_PATH}/agents/res/examples/train-data.txt"  # Hardcoded dataset file for now
ONNX_FILE_EXTENSION = ".onnx"
PYTORCH_FILE_EXTENSION = ".pt"

# Model Formats
SAFETENSORS: Literal["SAFETENSORS"] = "SAFETENSORS"
GGUF: Literal["GGUF"] = "GGUF"
PYTORCH: Literal["PYTORCH"] = "PYTORCH"
ONNX: Literal["ONNX"] = "ONNX"
GPTQ: Literal["GPTQ"] = "GPTQ"
UNSUPPORTED: Literal["UNSUPPORTED"] = "UNSUPPORTED"


class SupportedFeatures:
    FP_QUANT_TYPES: List[str] = ["F32", "F16"]
    HIGHEST_FP_QUANT_TYPE: str = "F32"

    MODEL_FORMAT_TYPES = SAFETENSORS or GGUF or PYTORCH or ONNX or GPTQ

    INPUT_FORMATS = [
        SAFETENSORS,
        GGUF,
        PYTORCH,
        ONNX,
        GPTQ
    ]

    OUTPUT_FORMATS = [
        GGUF,
        PYTORCH,
        ONNX,
        GPTQ
    ]

    # Define conversion paths based on input format
    CONVERSION_PATHS = {
        # SAFETENSORS can be converted to all other formats except itself
        SAFETENSORS: [GGUF, PYTORCH, ONNX, GPTQ],
        # Other formats have specific conversion capabilities
        GGUF: [PYTORCH, ONNX, GPTQ],
        PYTORCH: [GPTQ],  # TORCH -> ONNX is tricky!
        ONNX: [PYTORCH],
        GPTQ: [],  # GPTQ can't be converted to other formats
    }

    GGUF_AVAILABLE_QUANT_TYPES = [
        "Q8_0",  # 8-bit integer quantization: Balances performance and precision.
        "Q5_0",  # 5-bit integer quantization: Reduces memory with minimal precision loss.
        "Q5_1",  # 5-bit integer quantization with variant 1: Slightly different scaling.
        "Q4_0",  # 4-bit integer quantization: Further reduces memory footprint.
        "Q4_1",  # 4-bit integer quantization with variant 1: Alternative scaling method.
        "Q2_K",  # 2-bit integer quantization: Significant memory savings with precision trade-off.
        "Q3_K",  # 3-bit integer quantization: Middle ground between Q2_K and Q4_K.
        "Q4_K",  # 4-bit integer quantization with K-based method: Optimized for specific hardware.
        "Q5_K",  # 5-bit integer quantization with K-based method: Balances memory and precision.
        "Q6_K",  # 6-bit integer quantization with K-based method: Higher precision with moderate memory usage.
        "Q2_K_S",  # 2-bit integer quantization with K-based method and signed values.
        "Q3_K_S",  # 3-bit integer quantization with K-based method and signed values.
        "Q4_K_S",  # 4-bit integer quantization with K-based method and signed values.
        "Q5_K_S",  # 5-bit integer quantization with K-based method and signed values.
        "IQ1_S",  # 1-bit integer quantization with small block size.
        "IQ2_XXS",  # 2-bit integer quantization with extra-extra-small block size.
        "IQ2_XS",  # 2-bit integer quantization with extra-small block size.
        "IQ2_S",  # 2-bit integer quantization with small block size.
        "IQ2_M",  # 2-bit integer quantization with medium block size.
        "IQ3_XXS",  # 3-bit integer quantization with extra-extra-small block size.
        "IQ3_XS",  # 3-bit integer quantization with extra-small block size.
        "IQ3_S",  # 3-bit integer quantization with small block size.
        "IQ3_M",  # 3-bit integer quantization with medium block size.
        "IQ4_XS",  # 4-bit integer quantization with extra-small block size.
    ]
    IMATRIX_OPTIONS = ["None (Lower Quality)", "From Input Directory", "Build Custom Matrix"]

    OUTPUTS = [GGUF, PYTORCH, ONNX, GPTQ]
    RESULTS_TABLE_HEADERS = [
        "Model File Path",
        "Type",
        "TTQ",
        "Size",
        "% of FP",
        "Test Runs",
        "Model Load (s)",
        "Encode (s/mpx)",
        "Prompt (t/s)",
        "Resp. (t/s)",
        "Quality (%)",
    ]
    RESULTS_TABLE_COLUMNS_WIDTH = ["16%", "8%", "8%", "8%", "8%", "7%", "9%", "9%", "9%", "9%", "9%"]


def get_available_quant_types():
    """
    Returns the list of available quantization types.

    Returns:
        list: List of quantization types.
    """
    return SupportedFeatures.GGUF_AVAILABLE_QUANT_TYPES
