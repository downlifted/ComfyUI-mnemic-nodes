import os
import random
import json
import numpy as np
import torch
from colorama import Fore, Style
import sys
from ..utils.string_clean import process_text
from ..utils.image_utils import tensor_to_pil, encode_image
import configparser
import requests

class GroqAPIVLM:
    DEFAULT_PROMPT = "Use [system_message] and [user_input]"
    
    VLM_MODELS = [
        "llava-v1.5-7b-4096-preview",
        "llama-3.2-11b-vision-preview",
        "llava-v1.6-34b",
        "llava-v1.6-mistral-7b",
        "llama-3.1-70b-versatile",
        "gemma2-9b-it"
    ]
    
    def __init__(self):
        self.api_keys = self.load_api_keys()
        self.current_key_index = 0
        self.prompt_options = self.load_prompt_options()

    def load_api_keys(self):
        config = configparser.ConfigParser()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_paths = [
            os.path.join(current_dir, 'GroqConfig.ini'),
            os.path.join(current_dir, 'groq', 'GroqConfig.ini'),
            os.path.join(current_dir, '..', 'groq', 'GroqConfig.ini'),
            '/root/comfy/ComfyUI/custom_nodes/ComfyUI-mnemic-nodes/nodes/groq/GroqConfig.ini',
            r"C:\Users\bewiz\OneDrive\Desktop\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-mnemic-nodes\nodes\groq\GroqConfig.ini"
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                config.read(path)
                if 'API_KEYS' in config:
                    keys = config['API_KEYS']['groq_keys'].split(',')
                    return [key.strip() for key in keys]
        
        # If no config file is found, use the provided default key
        return ["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"]

    def get_next_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return self.api_keys[self.current_key_index]

    def load_prompt_options(self):
        try:
            current_directory = os.path.dirname(os.path.realpath(__file__))
            groq_directory = os.path.join(current_directory, 'groq')
            prompt_files = [
                os.path.join(groq_directory, 'DefaultPrompts_VLM.json'),
                os.path.join(groq_directory, 'UserPrompts_VLM.json')
            ]
            prompt_options = {}
            for file in prompt_files:
                if os.path.exists(file):
                    with open(file, 'r') as f:
                        prompt_options.update(json.load(f))
            return prompt_options
        except Exception as e:
            print(Fore.RED + f"Failed to load prompt options: {e}" + Style.RESET_ALL)
            return {}

    @staticmethod
    def get_prompt_content(prompt_options, preset):
        return prompt_options.get(preset, "")

    @classmethod
    def load_prompt_options_class(cls):
        try:
            current_directory = os.path.dirname(os.path.realpath(__file__))
            groq_directory = os.path.join(current_directory, 'groq')
            prompt_files = [
                os.path.join(groq_directory, 'DefaultPrompts_VLM.json'),
                os.path.join(groq_directory, 'UserPrompts_VLM.json')
            ]
            prompt_options = {}
            for file in prompt_files:
                if os.path.exists(file):
                    with open(file, 'r') as f:
                        prompt_options.update(json.load(f))
            return prompt_options
        except Exception as e:
            print(f"Failed to load prompt options: {e}")
            return {}

    @classmethod
    def INPUT_TYPES(cls):
        prompt_options = cls.load_prompt_options_class()
        return {
            "required": {
                "model": (cls.VLM_MODELS, {"tooltip": "Select the Vision-Language Model (VLM) to use."}),
                "preset": ([cls.DEFAULT_PROMPT] + list(prompt_options.keys()), {"tooltip": "Select a preset prompt or use a custom prompt for the model."}),
                "system_message": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional system message to guide model behavior."}),
                "user_input": ("STRING", {"multiline": True, "default": "", "tooltip": "User input or prompt for the model to generate a response."}),
                "image": ("IMAGE", {"label": "Image (required for VLM models)", "tooltip": "Upload an image for processing by the VLM model."}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Controls randomness in responses.\n\nA higher temperature makes the model take more risks, leading to more creative or varied answers.\n\nA lower temperature (closer to 0.1) makes the model more focused and predictable."}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 131072, "step": 1, "tooltip": "Maximum number of tokens to generate in the output."}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Limits the pool of words the model can choose from based on their combined probability.\n\nSet it closer to 1 to allow more variety in output. Lowering this (e.g., 0.9) will restrict the output to the most likely words, making responses more focused."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295, "tooltip": "Seed for random number generation, ensuring reproducibility."}),
                "max_retries": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1, "tooltip": "Maximum number of retries in case of failures."}),
                "stop": ("STRING", {"default": "", "tooltip": "Stop generation when the specified sequence is encountered."}),
                "json_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable JSON mode for structured output.\n\nIMPORTANT: Requires you to use the word 'JSON' in the prompt."}),
            }
        }    

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("api_response", "success", "status_code")
    OUTPUT_TOOLTIPS = ("The API response. This is the description of your input image generated by the model", "Whether the request was successful", "The status code of the request")
    FUNCTION = "process_completion_request"
    CATEGORY = "⚡ MNeMiC Nodes"
    DESCRIPTION = "Uses Groq API for image processing."
    
    def process_completion_request(self, model, image, temperature, max_tokens, top_p, seed, max_retries, stop, json_mode, preset="", system_message="", user_input=""):
        # Set the seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if preset == self.DEFAULT_PROMPT:
            system_message = system_message
        else:
            system_message = self.get_prompt_content(self.prompt_options, preset)

        url = 'https://api.groq.com/openai/v1/chat/completions'
        
        if image is not None and isinstance(image, torch.Tensor):
            # Process the image
            image_pil = tensor_to_pil(image)
            base64_image = encode_image(image_pil)
            if base64_image:
                combined_message = f"{system_message}\n{user_input}"
                # Send one single message containing both text and image
                image_content = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": combined_message},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
                messages = [image_content]
            else:
                print(Fore.RED + "Failed to encode image." + Style.RESET_ALL)
                messages = []
        else:
            print(Fore.RED + "Image is required for VLM models." + Style.RESET_ALL)
            return "Image is required for VLM models.", False, "400 Bad Request"
       
        data = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'seed': seed
        }
        
        if stop:
            data['stop'] = stop

        for _ in range(max_retries):
            try:
                headers = {'Authorization': f'Bearer {self.api_keys[self.current_key_index]}'}
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content'], True, response.status_code
            except requests.exceptions.RequestException as e:
                if 'rate limit' in str(e).lower():
                    print(Fore.YELLOW + f"Rate limit reached for API key {self.current_key_index + 1}. Switching to next key." + Style.RESET_ALL)
                    self.get_next_api_key()
                else:
                    print(Fore.RED + f"Error: {e}" + Style.RESET_ALL)
            
        print(Fore.RED + "Max retries reached. Unable to complete the request." + Style.RESET_ALL)
        return "Max retries reached. Unable to complete the request.", False, "429 Too Many Requests"
