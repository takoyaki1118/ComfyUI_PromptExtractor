from PIL import Image
import json
import torch
import os

class PromptImageExtractorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "image_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("positive_prompt", "image")
    FUNCTION = "extract_prompt_and_image"
    CATEGORY = "Custom Nodes"

    def extract_prompt_and_image(self, image, image_path):
        try:
            # 画像パスが有効か確認
            if not image_path or not os.path.exists(image_path):
                return "No valid image path provided", image

            # 画像からメタデータを抽出
            img = Image.open(image_path)
            metadata = img.info

            positive_prompt = "Not found"

            # メタデータからプロンプトを抽出
            if "prompt" in metadata:
                prompt_data = json.loads(metadata["prompt"])
                # PromptCombinerNode (ID: 15) の出力を探す
                for node_id, node in prompt_data.items():
                    if node.get("class_type") == "PromptCombinerNode":
                        positive_prompt = node.get("inputs", {}).get("prompt", "Not found")
                        break
                    elif node.get("class_type") == "ShowText|pysssss":
                        positive_prompt = node.get("inputs", {}).get("text", "Not found")
                        break

            elif "workflow" in metadata:
                workflow_data = json.loads(metadata["workflow"])
                # PromptCombinerNode または ShowText の widgets_values を探す
                for node in workflow_data.get("nodes", []):
                    if node.get("type") == "PromptCombinerNode":
                        positive_prompt = node.get("widgets_values", ["Not found"])[0]
                        break
                    elif node.get("type") == "ShowText|pysssss":
                        positive_prompt = node.get("widgets_values", ["Not found"])[0]
                        break

            # 入力画像をそのまま出力
            return positive_prompt, image

        except Exception as e:
            return f"Error: {str(e)}", image

# ノードのマッピング
NODE_CLASS_MAPPINGS = {
    "PromptImageExtractorNode": PromptImageExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptImageExtractorNode": "Prompt and Image Extractor Node"
}