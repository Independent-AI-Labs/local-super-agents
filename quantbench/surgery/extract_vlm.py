import argparse
from typing import Dict

import torch
from gguf import *
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    AutoProcessor,
    Qwen2_5_VLConfig,
    Qwen2VLImageProcessor
)

VISION = "clip.vision"

# Global variable to hold the model input directory
MODEL_INPUT_DIR = r"C:\Users\vdonc\Desktop\Qwen2.5-VL-3B-Instruct"


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def to_gguf_name(name: str) -> str:
    og = name
    name = name.replace("text_model", "t").replace("vision_model", "v")
    name = name.replace("blocks", "blk").replace("embeddings.", "")
    name = name.replace("attn.", "attn_")
    name = name.replace("mlp.gate_proj", "ffn_gate").replace("mlp.up_proj", "ffn_up").replace("mlp.down_proj", "ffn_down")
    name = name.replace("proj.", "out.")
    # Replace norm names so that layernorms become ln1/ln2
    name = name.replace("norm1", "ln1").replace("norm2", "ln2")
    name = name.replace("merger.mlp", "mm")
    print(f"[to_gguf_name] {og} --> {name}")
    return name


def find_vision_tensors(qwen2vl, np_dtype) -> Dict[str, np.ndarray]:
    vision_model = qwen2vl.visual
    tensor_map = {}

    for name, ten in vision_model.state_dict().items():
        ten = ten.numpy()

        if 'qkv' in name:
            # Split qkv tensor into q, k, v
            if ten.ndim == 2:  # weight
                c3, _ = ten.shape
            else:  # bias
                c3 = ten.shape[0]
            assert c3 % 3 == 0, f"qkv tensor shape mismatch in {name}"
            c = c3 // 3
            wq = ten[:c]
            wk = ten[c: c * 2]
            wv = ten[c * 2:]
            base_name = to_gguf_name(f"vision_model.{name}")
            tensor_map[base_name.replace("qkv", "q")] = wq
            tensor_map[base_name.replace("qkv", "k")] = wk
            tensor_map[base_name.replace("qkv", "v")] = wv

        elif 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name:
            # Handle the MLP structure with gate/up/down projections
            tensor_map[to_gguf_name(f"vision_model.{name}")] = ten

        elif 'merger' in name:
            # Map merger layernorm parameters to post_ln keys
            if name.endswith("ln_q.weight"):
                tensor_map['v.post_ln.weight'] = ten
            elif name.endswith("ln_q.bias"):
                tensor_map['v.post_ln.bias'] = ten
            elif 'mlp' in name:
                # Handle the merger MLP layers
                if name.endswith("mlp.0.weight") or name.endswith("mlp.0.bias"):
                    # First linear layer in Sequential
                    new_name = name.replace("mlp.0", "mm.0")
                    tensor_map[to_gguf_name(new_name)] = ten
                elif name.endswith("mlp.2.weight") or name.endswith("mlp.2.bias"):
                    # Second linear layer in Sequential (after GELU)
                    new_name = name.replace("mlp.2", "mm.2")
                    tensor_map[to_gguf_name(new_name)] = ten
                else:
                    tensor_map[to_gguf_name(name)] = ten
            else:
                tensor_map[to_gguf_name(name)] = ten

        elif 'patch_embed.proj.weight' in name:
            # For the Conv3d, split the temporal kernel dimension (which is 2)
            c1, c2, kt, kh, kw = ten.shape
            assert kt == 2, "Current implementation only supports temporal_patch_size of 2"
            tensor_map["v.patch_embd.weight"] = ten[:, :, 0, ...]
            tensor_map["v.patch_embd.weight.1"] = ten[:, :, 1, ...]

        else:
            tensor_map[to_gguf_name(f"vision_model.{name}")] = ten

    # Ensure biases and layer norm weights remain in fp32
    for new_name, ten in tensor_map.items():
        if (ten.ndim <= 1 or
                new_name.endswith("ln1.weight") or
                new_name.endswith("ln1.bias") or
                new_name.endswith("ln2.weight") or
                new_name.endswith("ln2.bias")):
            tensor_map[new_name] = ten.astype(np.float32)
        else:
            tensor_map[new_name] = ten.astype(np_dtype)

    # Dummy tensor as a placeholder for position embeddings
    # Required even when using rotary embeddings
    tensor_map["v.position_embd.weight"] = np.zeros([10, 10], dtype=np.float32)

    return tensor_map


def main(args):
    global MODEL_INPUT_DIR
    if args.data_type == 'fp32':
        dtype = torch.float32
        np_dtype = np.float32
        ftype = 0
    elif args.data_type == 'fp16':
        dtype = torch.float32  # load model in fp32 then convert selected tensors to fp16
        np_dtype = np.float16
        ftype = 1
    else:
        raise ValueError("Unsupported data type")

    model_path = ""
    model_name = args.model_name
    print("model_name: ", model_name)

    if MODEL_INPUT_DIR is not None:
        model_path = MODEL_INPUT_DIR
        print(f"Loading model from local directory: {model_path}")
        qwen2vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype
        )
    else:
        print("Loading model from Hugging Face Hub (default behavior)")
        qwen2vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )

    cfg: Qwen2_5_VLConfig = qwen2vl.config
    vcfg = cfg.vision_config

    if MODEL_INPUT_DIR is not None:
        model_name = os.path.basename(model_path.rstrip(os.sep))

    fname_out = f"{model_name.replace('/', '-').lower()}-vision.gguf"

    fout = GGUFWriter(path=fname_out, arch="clip")
    fout.add_description("Image encoder for Qwen2.5VL")
    fout.add_file_type(ftype)
    fout.add_bool("clip.has_text_encoder", False)
    fout.add_bool("clip.has_vision_encoder", True)
    fout.add_bool("clip.has_qwen2vl_merger", True)
    fout.add_string("clip.projector_type", "qwen2vl_merger")

    print(vcfg)

    tensor_map = find_vision_tensors(qwen2vl, np_dtype)
    for name, data in tensor_map.items():
        fout.add_tensor(name, data)

    fout.add_uint32("clip.vision.patch_size", vcfg.patch_size)
    fout.add_uint32("clip.vision.image_size", 560)
    fout.add_uint32("clip.vision.projection_dim", 1536)
    fout.add_uint32("clip.vision.embedding_length", vcfg.hidden_size)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), vcfg.num_heads)
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), 1e-6)
    fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), vcfg.depth)
    # For Qwen2.5VL the feed forward dim is 0 since we handle the MLP differently
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), 0)
    fout.add_name(model_name)

    fout.add_string("clip.vision.mm_patch_merge_type", "qwen2vl_merger")
    # Set the appropriate crop resolution based on image_size
    fout.add_uint32("clip.vision.image_crop_resolution", 560)

    if MODEL_INPUT_DIR is not None:
        processor: Qwen2_5_VLProcessor = Qwen2VLImageProcessor.from_pretrained(model_path)
    else:
        processor: Qwen2_5_VLProcessor = AutoProcessor.from_pretrained(model_name)

    fout.add_array("clip.vision.image_mean", processor.image_mean)
    fout.add_array("clip.vision.image_std", processor.image_std)

    # Set the activation function flags based on the model config
    if hasattr(vcfg, 'hidden_act') and 'silu' in vcfg.hidden_act.lower():
        fout.add_bool("clip.use_silu", True)
        fout.add_bool("clip.use_gelu", False)
    else:
        fout.add_bool("clip.use_silu", False)
        fout.add_bool("clip.use_gelu", False)  # Use defaults from dump

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()
    print("Saved model as:", fname_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", nargs='?')
    parser.add_argument("--data_type", nargs='?', choices=['fp32', 'fp16'], default="fp32")
    parser.add_argument("--input_dir", type=str, help="Path to the local model directory")
    args = parser.parse_args()

    # Update the global MODEL_INPUT_DIR if provided.
    if args.input_dir:
        if os.path.isdir(args.input_dir):
            MODEL_INPUT_DIR = args.input_dir
        else:
            raise ValueError(f"Input directory not found: {args.input_dir}")

    main(args)
