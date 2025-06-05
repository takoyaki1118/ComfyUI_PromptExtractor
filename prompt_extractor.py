import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import logging
import json

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# カスタム画像読み込みノード
class CustomLoadImageWithPathNode:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "image_path")
    FUNCTION = "load_image"

    def load_image(self, image):
        try:
            if not image:
                raise ValueError("No image file selected")

            image_path = os.path.join(folder_paths.get_input_directory(), image)
            if not os.path.exists(image_path):
                raise ValueError(f"Image file does not exist: {image_path}")

            logger.info(f"Loading image: {image_path}")
            img = Image.open(image_path)
            output_images = []
            output_masks = []

            i = img.convert("RGB")
            image_array = np.array(i).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]

            if "A" in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64))

            output_images.append(image_tensor)
            output_masks.append(mask)

            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)

            logger.info(f"Image loaded successfully: {image}")
            return (output_image, output_mask, image)

        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise ValueError(f"Error loading image: {str(e)}")

# プロンプト抽出ノード
class PromptExtractorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "image_path": ("STRING", {"default": "Not found"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("positive_prompt", "image")
    FUNCTION = "extract_prompt"

    def extract_prompt(self, image, image_path):
        try:
            # image_path の型チェック
            if not isinstance(image_path, str):
                raise TypeError(f"image_path must be a string, got {type(image_path).__name__}")

            full_path = os.path.join(folder_paths.get_input_directory(), image_path)
            if not os.path.exists(full_path):
                full_path = image_path
                if not os.path.exists(full_path):
                    raise ValueError(f"Image file does not exist: {full_path}")

            logger.info(f"Processing image: {full_path}")
            if image.shape[-1] != 3:
                raise ValueError("Image must have 3 channels (RGB)")

            img = Image.open(full_path)
            metadata = img.info
            logger.info(f"Metadata keys: {list(metadata.keys())}")
            positive_prompt = "Not found"

            if "prompt" in metadata:
                try:
                    prompt_data = json.loads(metadata["prompt"])
                    logger.debug(f"Prompt metadata nodes: {len(prompt_data)}")

                    def resolve_prompt(value, prompt_data, visited=None):
                        if visited is None:
                            visited = set()
                        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                            node_id = value[0]
                            if node_id in visited:
                                logger.warning(f"Circular reference detected for node_id: {node_id}")
                                return "Not found"
                            visited.add(node_id)
                            if node_id in prompt_data:
                                node = prompt_data[node_id]
                                logger.debug(f"Resolving node_id: {node_id}, class_type: {node.get('class_type')}")
                                if node.get("class_type") == "PromptCombinerNode":
                                    prompt = node.get("inputs", {}).get("prompt", "Not found")
                                    logger.debug(f"Found PromptCombinerNode prompt: {prompt}")
                                    return prompt if isinstance(prompt, str) else str(prompt)
                                elif node.get("class_type") == "ShowText|pysssss":
                                    text_input = node.get("inputs", {}).get("text", "Not found")
                                    logger.debug(f"ShowText|pysssss text_input: {text_input}")
                                    return resolve_prompt(text_input, prompt_data, visited)
                        return value if isinstance(value, str) else str(value)

                    for node_id, node in prompt_data.items():
                        if node.get("class_type") == "PromptCombinerNode":
                            prompt = node.get("inputs", {}).get("prompt", "Not found")
                            logger.debug(f"PromptCombinerNode (ID {node_id}) prompt: {prompt}")
                            positive_prompt = resolve_prompt(prompt, prompt_data)
                            if positive_prompt != "Not found":
                                break

                    if positive_prompt == "Not found":
                        for node_id, node in prompt_data.items():
                            if node.get("class_type") == "ShowText|pysssss":
                                widgets_values = node.get("widgets_values", ["Not found"])
                                logger.debug(f"ShowText|pysssss (ID {node_id}) widgets_values: {widgets_values}")
                                if isinstance(widgets_values, list) and widgets_values:
                                    prompt = widgets_values[0]
                                    if isinstance(prompt, str) and prompt != "Not found":
                                        positive_prompt = prompt
                                        logger.debug(f"Extracted prompt from widgets_values: {positive_prompt}")
                                        break
                                text_input = node.get("inputs", {}).get("text", "Not found")
                                logger.debug(f"ShowText|pysssss text_input: {text_input}")
                                prompt = resolve_prompt(text_input, prompt_data)
                                if isinstance(prompt, str) and prompt != "Not found":
                                    positive_prompt = prompt
                                    logger.debug(f"Resolved prompt from ShowText|pysssss: {positive_prompt}")
                                    break

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in prompt metadata: {str(e)}")
                    raise ValueError(f"Invalid JSON in prompt metadata: {str(e)}")

            if positive_prompt == "Not found" and "workflow" in metadata:
                try:
                    workflow_data = json.loads(metadata["workflow"])
                    logger.debug(f"Workflow metadata nodes: {len(workflow_data.get('nodes', []))}")
                    for node in workflow_data.get('nodes', []):
                        if node.get("type") == "PromptCombinerNode":
                            widgets_values = node.get("widgets_values", ["Not found"])
                            logger.debug(f"PromptCombinerNode widgets_values: {widgets_values}")
                            if isinstance(widgets_values, list) and widgets_values:
                                prompt = widgets_values[0]
                                positive_prompt = prompt if isinstance(prompt, str) else str(prompt)
                                logger.debug(f"Extracted prompt from workflow PromptCombinerNode: {positive_prompt}")
                                break
                        elif node.get("type") == "ShowText|pysssss":
                            widgets_values = node.get("widgets_values", ["Not found"])
                            logger.debug(f"ShowText|pysssss widgets_values: {widgets_values}")
                            if isinstance(widgets_values, list) and widgets_values:
                                prompt = widgets_values[0]
                                positive_prompt = prompt if isinstance(prompt, str) else str(prompt)
                                logger.debug(f"Extracted prompt from workflow ShowText|pysssss: {positive_prompt}")
                                break

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in workflow metadata: {str(e)}")
                    raise ValueError(f"Invalid JSON in workflow metadata: {str(e)}")

            if isinstance(positive_prompt, list):
                if len(positive_prompt) == 1:
                    positive_prompt = positive_prompt[0] if isinstance(positive_prompt[0], str) else str(positive_prompt[0])
                    logger.debug(f"Extracted single-element list prompt: {positive_prompt}")
                else:
                    positive_prompt = ", ".join(str(p) for p in positive_prompt) if positive_prompt else "Not found"
                    logger.debug(f"Joined multi-element list prompt: {positive_prompt}")

            logger.info(f"Final positive_prompt: {positive_prompt}")
            return positive_prompt, image

        except Exception as e:
            logger.error(f"Error extracting prompt: {str(e)}")
            raise ValueError(f"Error extracting prompt: {str(e)}")

# ノードのマッピング
NODE_CLASS_MAPPINGS = {
    "CustomLoadImageWithPathNode": CustomLoadImageWithPathNode,
    "PromptExtractorNode": PromptExtractorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomLoadImageWithPathNode": "Custom Load Image With Path",
    "PromptExtractorNode": "Prompt Extractor Node"
}