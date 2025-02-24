def get_available_quant_types():
    return [
        "F16", "Q8_0", "Q4_0", "IQ4_XS", "Q4_K_M", "Q6_K", "Q5_K_M",
        "Q5_K_S", "IQ4_NL", "Q4_K_S", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K"
    ]

# TODO Other types may require additional configuration, such as importance matrices.


def calculate_file_size_and_percentage(original_size_bytes, quantized_size_bytes):
    """Calculates file size in GB and percentage of original size."""
    if original_size_bytes == 0:
        return "0.00 GB", "0.00%"  # Avoid division by zero
    quantized_size_gb = quantized_size_bytes / (1024 ** 3)
    percentage_of_original = (quantized_size_bytes / original_size_bytes) * 100
    return f"{quantized_size_gb:.2f} GB", f"{percentage_of_original:.2f}%"
