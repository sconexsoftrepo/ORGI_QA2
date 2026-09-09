import json
import os
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_prompt(prompt_path, filename):
    """Load a prompt from a file in the prompt folder."""
    try:
        prompt_file = os.path.join(prompt_path, filename)
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        logger.info(f"Loaded prompt from {prompt_file}")
        return prompt_content
    except Exception as e:
        logger.error(f"Failed to load prompt from {filename}: {e}")
        raise

def load_config(config_path='config.json'):
    """Load configuration from a JSON file and include prompt contents."""
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")

        # Load prompt files specified in ollama_config
        # Note: visicooler detection prompts removed as detection now handled by mobile app
        ollama_config = config.get('ollama_config', {})
        prompt_path = ollama_config.get('prompt_path', '')
        prompt_files = ollama_config.get('prompt_files', {})
        
        if prompt_path and prompt_files:
            config['ollama_config']['prompts'] = {
                # Removed: 'visicooler_detect' and 'visicooler_attrs'
                # Visicooler detection now handled by mobile app
                'extended_visibility_group1': load_prompt(prompt_path, prompt_files.get('extended_visibility_group1', 'extended_group1.txt')),
                'extended_visibility_group2': load_prompt(prompt_path, prompt_files.get('extended_visibility_group2', 'extended_group2.txt'))
            }
        else:
            logger.warning("Prompt path or prompt files not specified in config. Skipping prompt loading.")
            config['ollama_config']['prompts'] = {
                'extended_visibility_group1': '',
                'extended_visibility_group2': ''
            }

        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def load_yaml_classes(yaml_path):
    """Load class IDs from a YAML file."""
    try:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        class_ids = yaml_data.get('names', {})
        if not class_ids:
            raise ValueError(f"No 'names' field found in YAML file: {yaml_path}")
        if isinstance(class_ids, list):
            class_ids = {str(i): name for i, name in enumerate(class_ids)}
        logger.info(f"Loaded class IDs from {yaml_path}")
        return class_ids
    except Exception as e:
        logger.error(f"Failed to load class IDs from {yaml_path}: {e}")
        raise

def load_json_classes(json_path):
    """Load class IDs from a JSON file."""
    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            class_ids = json.load(f)
        if not isinstance(class_ids, dict):
            raise ValueError(f"Invalid JSON format in {json_path}: Expected a dictionary")
        logger.info(f"Loaded class IDs from {json_path}")
        return class_ids
    except Exception as e:
        logger.error(f"Failed to load class IDs from {json_path}: {e}")
        raise