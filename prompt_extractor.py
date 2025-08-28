import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomLoadImageWithPathNode:
    # ... (このクラスは変更ありませんので、省略します)
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return {"required": {"image": (sorted(files), {"image_upload": True}),}}
    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "image_path")
    FUNCTION = "load_image"
    def load_image(self, image):
        try:
            image_path = folder_paths.get_annotated_filepath(image)
            img = Image.open(image_path)
            i = img.convert("RGB")
            image_array = np.array(i).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]
            if "A" in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32)
            return (image_tensor, mask.unsqueeze(0), image)
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise

class PromptExtractorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {}), "image_path": ("STRING", {"default": "Not found"}),}}
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("positive_prompt", "image")
    FUNCTION = "extract_prompt"

    def extract_prompt(self, image, image_path):
        try:
            full_path = folder_paths.get_annotated_filepath(image_path)
            img = Image.open(full_path)
            metadata = img.info
            positive_prompt = "Not found"

            if "prompt" not in metadata:
                return ("Not found: No API metadata", image)
            
            prompt_data = json.loads(metadata["prompt"])

            # --- ★★★ 最重要修正箇所 ★★★ ---
            # プロンプトテキストが含まれる可能性のあるキーのリスト
            PROMPT_TEXT_KEYS = ["text", "value", "item", "prefix_tags", "main_text"]

            def resolve_value(value, prompt_data, visited):
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    return resolve_node(value[0], prompt_data, visited)
                elif isinstance(value, str):
                    return value
                return ""

            def resolve_node(node_id, prompt_data, visited):
                if node_id in visited: return ""
                visited.add(node_id)

                if node_id not in prompt_data: return ""
                
                node = prompt_data[node_id]
                class_type = node.get("class_type")
                inputs = node.get("inputs", {})

                if class_type in ["PromptCombinerNode", "SimpleTextCombinerNode"]:
                    parts = []
                    # inputs辞書の全ての値をループ処理
                    for key, value in inputs.items():
                        if key == 'separator': continue
                        part = resolve_value(value, prompt_data, visited.copy())
                        if part:
                            parts.append(part)
                    
                    separator = inputs.get("separator", ", ")
                    return separator.join(filter(None, parts))
                else:
                    # その他のノードの場合、指定されたキーのリストを順番に探し、
                    # 最初に見つかった文字列を返す。これにより設定値("auto", "Random"等)を無視する。
                    for key in PROMPT_TEXT_KEYS:
                        if key in inputs and isinstance(inputs[key], str):
                            return inputs[key]
                
                return ""

            # --- 探索開始 ---
            sampler_node_id = None
            for node_id, node in prompt_data.items():
                if node.get("class_type") == "KSampler":
                    sampler_node_id = node_id
                    break

            if not sampler_node_id:
                return ("Not found: No KSampler", image)

            sampler_inputs = prompt_data[sampler_node_id].get("inputs", {})
            positive_link = sampler_inputs.get("positive")
            if not positive_link:
                return ("Not found: No positive input on KSampler", image)

            positive_clip_id = positive_link[0]
            positive_clip_node = prompt_data.get(positive_clip_id, {})
            
            text_link = positive_clip_node.get("inputs", {}).get("text")
            if not text_link:
                 return ("Not found: No text input on CLIPTextEncode", image)

            resolved = resolve_value(text_link, prompt_data, set())
            if resolved:
                positive_prompt = resolved

            logger.info(f"Final positive_prompt: {positive_prompt}")
            return (positive_prompt, image)

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return ("Error", image)

NODE_CLASS_MAPPINGS = {
    "CustomLoadImageWithPathNode": CustomLoadImageWithPathNode,
    "PromptExtractorNode": PromptExtractorNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomLoadImageWithPathNode": "Custom Load Image With Path",
    "PromptExtractorNode": "Prompt Extractor Node"
}