#!/usr/bin/env python3
"""
Cosmos Reason 2 Remote Client
Analyze images and videos using Cosmos Reason 2 running on a remote server.

Usage:
  python3 cosmos_client.py <file_path> [prompt]
  python3 cosmos_client.py <file_path> --prompt-file <yaml_file>
  python3 cosmos_client.py test

Examples:
  python3 cosmos_client.py video.mp4
  python3 cosmos_client.py video.mp4 "What happens in this video?"
  python3 cosmos_client.py image.jpg --prompt-file prompts/caption.yaml
  python3 cosmos_client.py test
"""

import requests
import base64
import sys
from pathlib import Path
import re
import yaml

# Configuration
WORKSTATION_IP = "10.0.0.46"
API_URL = f"http://{WORKSTATION_IP}:8000/v1/chat/completions"

# Supported file types
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.gif']

def load_yaml_prompt(yaml_path):
    """Load prompt from YAML file"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        system_prompt = data.get('system_prompt', 'You are a helpful assistant.')
        user_prompt = data.get('user_prompt', '')
        
        return system_prompt, user_prompt
    except Exception as e:
        print(f"❌ Error loading YAML file: {e}")
        return None, None

def parse_reasoning(content):
    """Parse <think> and <answer> sections from response if present"""
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
    
    if think_match and answer_match:
        thinking = think_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return thinking, answer, True
    
    return None, content.strip(), False

def display_result(content, show_reasoning=True):
    """Display result with optional reasoning sections"""
    thinking, answer, has_sections = parse_reasoning(content)
    
    if has_sections and show_reasoning:
        print("=" * 70)
        print("🤔 REASONING:")
        print("=" * 70)
        print(thinking)
        print()
        print("=" * 70)
        print("✅ ANSWER:")
        print("=" * 70)
        print(answer)
        print("=" * 70)
    else:
        print("=" * 70)
        print("✅ RESULT:")
        print("=" * 70)
        print(answer)
        print("=" * 70)

def process_file(file_path, prompt=None, yaml_file=None, show_reasoning=True):
    """Process video or image file with Cosmos Reason 2"""
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    # Determine file type
    ext = file_path.suffix.lower()
    
    if ext in VIDEO_EXTENSIONS:
        file_type = 'video'
        default_prompt = "Caption this video in detail."
        print(f"📹 Processing video: {file_path.name}")
    elif ext in IMAGE_EXTENSIONS:
        file_type = 'image'
        default_prompt = "Describe this image in detail."
        print(f"🖼️  Processing image: {file_path.name}")
    else:
        print(f"❌ Error: Unsupported file type: {ext}")
        return None
    
    # Read and encode file
    print("📦 Encoding file...")
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
            file_size_mb = len(file_data) / (1024 * 1024)
            print(f"   File size: {file_size_mb:.2f} MB")
            data_b64 = base64.b64encode(file_data).decode()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None
    
    # Build content based on file type
    if file_type == 'video':
        content_item = {
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{data_b64}"}
        }
    else:
        mime_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }
        mime_type = mime_map.get(ext, 'image/jpeg')
        content_item = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{data_b64}"}
        }
    
    # Determine prompts
    if yaml_file:
        print(f"📄 Loading prompt from: {yaml_file}")
        system_prompt, user_prompt = load_yaml_prompt(yaml_file)
        if system_prompt is None:
            return None
    else:
        # Simple default system prompt
        system_prompt = "You are a helpful assistant that analyzes images and videos."
        user_prompt = prompt if prompt else default_prompt
    
    # Build API request
    request_data = {
        "model": "nvidia/Cosmos-Reason2-2B",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    content_item,
                    {"type": "text", "text": user_prompt}
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.7
    }
    
    print(f"🚀 Sending to server at {WORKSTATION_IP}:8000...")
    if yaml_file:
        print(f"💭 Using YAML prompt: {yaml_file}")
    else:
        print(f"💭 Prompt: {user_prompt}")
    print()
    
    # Send request
    try:
        response = requests.post(API_URL, json=request_data, timeout=180)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        display_result(content, show_reasoning)
        return content
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to server at {WORKSTATION_IP}:8000")
        print("   Make sure the vLLM server is running on your workstation.")
        return None
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        return None
    except KeyError as e:
        print(f"❌ Error parsing response: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_connection():
    """Test connection to the server"""
    print(f"🔍 Testing connection to {WORKSTATION_IP}:8000...")
    try:
        response = requests.get(f"http://{WORKSTATION_IP}:8000/v1/models", timeout=5)
        response.raise_for_status()
        models = response.json()
        print(f"✅ Connected! Available models:")
        for model in models.get('data', []):
            print(f"   • {model.get('id', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Make sure vLLM server is running at {WORKSTATION_IP}:8000")
        return False

def show_usage():
    """Display usage information"""
    print(__doc__)
    print("\nFlags:")
    print("  --prompt-file FILE    Load prompt from YAML file")
    print("  --no-reasoning        Hide <think> sections if present")
    print()
    print("Common Use Cases:")
    print("  Test connection:        python3 cosmos_client.py test")
    print("  Quick analysis:         python3 cosmos_client.py video.mp4")
    print("  Custom prompt:          python3 cosmos_client.py image.jpg 'Describe this'")
    print("  Use YAML template:      python3 cosmos_client.py video.mp4 --prompt-file prompts/caption.yaml")
    print()

def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)
    
    # Special commands
    if sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
        sys.exit(0)
    
    if sys.argv[1] in ['test', '--test']:
        test_connection()
        sys.exit(0)
    
    # Parse flags
    show_reasoning = '--no-reasoning' not in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != '--no-reasoning']
    
    # Check for --prompt-file flag
    yaml_file = None
    if '--prompt-file' in args:
        idx = args.index('--prompt-file')
        if idx + 1 < len(args):
            yaml_file = args[idx + 1]
            args = args[:idx] + args[idx+2:]
        else:
            print("❌ Error: --prompt-file requires a file path")
            sys.exit(1)
    
    if len(args) < 1:
        print("❌ Error: No file specified")
        show_usage()
        sys.exit(1)
    
    file_path = args[0]
    custom_prompt = ' '.join(args[1:]) if len(args) > 1 and not yaml_file else None
    
    # Process the file
    result = process_file(file_path, custom_prompt, yaml_file, show_reasoning)
    
    if result is None:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
