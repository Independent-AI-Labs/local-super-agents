from integration.data.config import INSTALL_PATH

# Constants
CONVERT_HF_TO_GGUF_SCRIPT = "convert_hf_to_gguf.py"
INPUT_MODEL_FILE_EXTENSION = ".safetensors"
GGUF_FILE_EXTENSION = ".gguf"
FP16_QUANT_TYPE = "F16"
IMATRIX_FILE_NAME = "imatrix.dat"
IMATRIX_CUSTOM_FILE_NAME = "imatrix-custom.dat"
IMATRIX_DATASET_FILE = f"{INSTALL_PATH}/agents/res/examples/train-data.txt"  # Hardcoded dataset file for now
ONNX_FILE_EXTENSION = ".onnx"
PYTORCH_FILE_EXTENSION = ".pt"


class SupportedFeatures:
    AVAILABLE_QUANT_TYPES = [
        "F16", "Q8_0", "Q4_0", "IQ4_XS", "Q4_K_M", "Q4_K_S", "Q6_K", "Q5_K_M",
        "Q5_K_S", "IQ4_NL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K"
    ]
    IMATRIX_OPTIONS = ["None (Reduced Quality)", "Use Included", "Generate Custom Matrix"]
    OUTPUT_FORMATS = ["GGUF", "ONNX", "GPTQ"]
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
    return SupportedFeatures.AVAILABLE_QUANT_TYPES
