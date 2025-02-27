# ⚛️ QuantBench

**LLM/VLM Quantization and Benchmarking Tool**

QuantBench is a user-friendly tool for quantizing and benchmarking Large Language Models (LLMs) and Vision Language Models (VLMs). It provides a Gradio interface to easily configure quantization settings, run quantization processes, and evaluate the performance of quantized models.

![](https://github.com/Independent-AI-Labs/local-super-agents/blob/feat/additional-output-formats/res/docs/screens/quantbench_updated.png)

**Key Features:**

* **Intuitive Gradio UI:** Simple and easy-to-use web interface for quantization and benchmarking.
* **Quantization Type Selection:** Supports various quantization types (determined by `get_available_quant_types` in `quantbench`).
* **Output Console:** Real-time output console to monitor the quantization and benchmarking process.
* **Benchmark Results Table:** Displays benchmark results in a structured table, including key metrics like quantization time (TTQ), size reduction, and performance indicators.
* **Natural-Language Rule-Base Dataset Generation:** Can be used for benchmarking, imatrix quantization and fine-tuning.

🚧 **imatrix and benchmarking features are currently under development and being actively implemented.** 🚧

The core quantization functionality is in place, allowing users to quantize models.  Benchmarking capabilities are being added to provide comprehensive performance evaluation of quantized models.

**Usage:**

0.  Environment Setup:

* Install **`requirements.txt`**.
* Set env. var  **`OLLAMA_PATH`:**
    *   **Description:** Path to the `ollama` executable.
    *   **How to Configure:** Set the `OLLAMA_PATH` environment variable to the full path to the `ollama` binary.
    *   **Default Value (Windows):** `{INSTALL_PATH}\agents\ollama\ollama.exe` (relative to `INSTALL_PATH`).

* Set env. var  **`LLAMACPP_QUANTIZE_PATH`:**
    *   **Description:** Path to the `llama-quantize` executable from llama.cpp. This tool is likely used to quantize language models for reduced size and potentially faster inference.
    *   **How to Configure:** Set the `LLAMACPP_QUANTIZE_PATH` environment variable to the full path to the `llama-quantize.exe` binary.
    *   **Default Value (Windows):** `{INSTALL_PATH}\build\llama.cpp\build\bin\Release\llama-quantize.exe` (relative to `INSTALL_PATH`).

1.  Run the `quantbench/gradio_ui.py` script.
2.  Access the Gradio UI in your browser (usually at `http://127.0.0.1:7860` or the address shown in the console).
3.  Navigate through the tabs to configure quantization settings, initiate quantization, and view results.
