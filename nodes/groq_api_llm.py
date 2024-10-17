import os
import json
import random
import numpy as np
import torch
from colorama import init, Fore, Style
from configparser import ConfigParser
import requests

from ..utils.api_utils import make_api_request, load_prompt_options, get_prompt_content

init()  # Initialize colorama

class GroqAPILLM:
    DEFAULT_PROMPT = "Use [system_message] and [user_input]"
    
    LLM_MODELS = [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "llama-guard-3-8b",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama3-groq-70b-8192-tool-use-preview",
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "gemma2-9b-it",
        "llama-3.2-1b-preview",
        "llama-3.2-3b-preview",
        "llama-3.2-11b-text-preview",
        "llama-3.2-90b-text-preview",
    ]
    
    def __init__(self):
        self.api_keys = self.load_api_keys()
        self.current_key_index = 0
        self.prompt_options = self.load_prompt_options()

    def load_api_keys(self):
        config = ConfigParser()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        groq_directory = os.path.join(current_dir, 'groq')
        config_paths = [
            os.path.join(groq_directory, 'GroqConfig.ini'),
            os.path.join(current_dir, '..', 'groq', 'GroqConfig.ini')
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                config.read(path)
                if 'API_KEYS' in config:
                    keys = config['API_KEYS']['groq_keys'].split(',')
                    return [key.strip() for key in keys]
        
        # If no config file is found, use the provided default key
        return ["default_api_key"]
    def get_next_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return self.api_keys[self.current_key_index]

    def load_prompt_options(self):
        try:
            current_directory = os.path.dirname(os.path.realpath(__file__))
            groq_directory = os.path.join(current_directory, 'groq')
            prompt_files = [
                os.path.join(groq_directory, 'DefaultPrompts.json'),
                os.path.join(groq_directory, 'UserPrompts.json')
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

    @classmethod
    def INPUT_TYPES(cls):
        try:
            current_directory = os.path.dirname(os.path.realpath(__file__))
            groq_directory = os.path.join(current_directory, 'groq')
            prompt_files = [
                os.path.join(groq_directory, 'DefaultPrompts.json'),
                os.path.join(groq_directory, 'UserPrompts.json')
            ]
            prompt_options = load_prompt_options(prompt_files)
        except Exception as e:
            print(Fore.RED + f"Failed to load prompt options: {e}" + Style.RESET_ALL)
            prompt_options = {}
    
        return {
            "required": {
                "model": (cls.LLM_MODELS, {"tooltip": "Select the Large Language Model (LLM) to use."}),
                "preset": ([cls.DEFAULT_PROMPT] + list(prompt_options.keys()), {"tooltip": "Select a preset or custom prompt for guiding the LLM."}),
                "system_message": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional system message to guide the LLM's behavior."}),
                "user_input": ("STRING", {"multiline": True, "default": "", "tooltip": "User input or prompt to generate a response from the LLM."}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Controls randomness in responses.\n\nA higher temperature makes the model take more risks, leading to more creative or varied answers.\n\nA lower temperature (closer to 0.1) makes the model more focused and predictable."}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 131072, "step": 1, "tooltip": "Maximum number of tokens to generate in the response."}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01, "tooltip": "Limits the pool of words the model can choose from based on their combined probability.\n\nSet it closer to 1 to allow more variety in output. Lowering this (e.g., 0.9) will restrict the output to the most likely words, making responses more focused."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295, "tooltip": "Seed for random number generation, ensuring reproducibility."}),
                "max_retries": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1, "tooltip": "Maximum number of retries in case of request failure."}),
                "stop": ("STRING", {"default": "", "tooltip": "Stop generation when the specified sequence is encountered."}),
                "json_mode": ("BOOLEAN", {"default": False, "tooltip": "Enable JSON mode for structured output.\n\nIMPORTANT: Requires you to use the word 'JSON' in the prompt."}),
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("api_response", "success", "status_code")
    OUTPUT_TOOLTIPS = ("The API response. This is the text generated by the model", "Whether the request was successful", "The status code of the request")
    FUNCTION = "process_completion_request"
    CATEGORY = "⚡ MNeMiC Nodes"
    DESCRIPTION = "Uses Groq API to generate text from language models."
    
    def process_completion_request(self, model, preset, system_message, user_input, temperature, max_tokens, top_p, seed, max_retries, stop, json_mode):
        # Set the seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
        if preset == self.DEFAULT_PROMPT:
            system_message = system_message
        else:
            system_message = get_prompt_content(self.prompt_options, preset)
    
        url = 'https://api.groq.com/openai/v1/chat/completions'
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_input}
        ]
       
        data = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
            'seed': seed
        }
        
        if stop:  # Only add stop if it's not empty
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