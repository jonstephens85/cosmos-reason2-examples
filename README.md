# Cosmos Reason 2 Utilities

A collection of practical tools for working with NVIDIA's [Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2) world model for physical AI applications.

## What's Included

This repository provides two complementary tools:

1. **Video Validator** - Batch process videos to evaluate physics compliance
2. **Remote Client** - Analyze images and videos from any device via a remote vLLM server

## Prerequisites

- Python 3.9+
- Access to a system running Cosmos Reason 2 via vLLM server
- Required Python packages: `requests`, `pandas`, `pyyaml`

```bash
pip install requests pandas pyyaml
```

---

## 1. Video Validator

### Overview

The Video Validator is a batch processing tool designed to evaluate whether videos conform to real-world physics. It processes folders of videos, analyzes each one using Cosmos Reason 2, and outputs structured results in CSV format.

**Key Features:**
- Batch process entire directories of videos
- Customizable physics evaluation prompts
- CSV output with verdicts, reasoning, and timestamps
- Progress tracking and verbose debugging mode

**Use Cases:**
- Validating synthetic training data for robotics
- Quality control for AI-generated videos
- Evaluating world model outputs
- Dataset curation for physical AI applications

### Usage

#### Basic Usage

```bash
python video_evaluator.py -i /path/to/videos
```

This will:
1. Process all `.mp4` files in the input directory
2. Evaluate each video for physics compliance
3. Output results to `physics_evaluation_results.csv`

#### Advanced Options

```bash
# Specify output directory
python video_evaluator.py -i /path/to/videos -o /path/to/output

# Enable verbose mode for debugging
python video_evaluator.py -i /path/to/videos -v

# Full example
python video_evaluator.py -i ./test_videos -o ./results -v
```

#### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-i`, `--input` | Input directory containing videos | Required |
| `-o`, `--output` | Output directory for results | Same as input |
| `-v`, `--verbose` | Enable verbose output | Off |

#### Output Format

The script generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `filename` | Name of the video file |
| `verdict` | YES (compliant) or NO (violations found) |
| `answer` | Full model response |
| `reasoning` | Extracted reasoning (if available) |
| `timestamp` | Processing timestamp |

#### Example Output

```csv
filename,verdict,answer,reasoning,timestamp
robot_walk.mp4,YES,"Yes, this video follows real-world physics...","The robot's gait shows proper...",2025-12-30 15:23:45
drone_flight.mp4,NO,"No, the drone exhibits impossible...","The sudden acceleration violates...",2025-12-30 15:24:12
```

### Configuration

**Server Connection:**

Edit the script to point to your vLLM server:

```python
SERVER_URL = "http://10.0.0.46:8000/v1/chat/completions"
```

**Custom Prompts:**

Modify the evaluation prompt in the script:

```python
prompt = """Does this video conform to real world physics? 
Analyze the motion, interactions, and physical behaviors shown. 
Point out any violations of physical laws or unrealistic elements."""
```

### Requirements

- Videos must be in `.mp4` format
- vLLM server must be running and accessible
- Sufficient disk space for CSV output

---

## 2. Remote Inference Client

### Overview

The Remote Inference Client enables you to analyze images and videos using Cosmos Reason 2 running on a remote server. This is useful when you want to:

- Run inference from a laptop while your GPU workstation handles the compute
- Access the model from multiple devices
- Demo the model without local GPU requirements
- Build applications that query a central inference server

**Cross-Platform:** Works on Mac, Windows, Linux, or any system with Python 3.9+

### Usage

#### Quick Start

```bash
# Test connection to server
python3 cosmos_client.py test

# Analyze an image
python3 cosmos_client.py image.jpg

# Analyze a video
python3 cosmos_client.py video.mp4

# Custom prompt
python3 cosmos_client.py scene.jpg "What should the robot do next?"
```

#### Using YAML Prompt Templates

YAML files give you full control over system and user prompts:

```bash
# Use a prompt template
python3 cosmos_client.py video.mp4 --prompt-file prompts/caption.yaml

# Download official prompts from the Cosmos Reason 2 repo
curl -o prompts/caption.yaml https://raw.githubusercontent.com/nvidia-cosmos/cosmos-reason2/main/prompts/caption.yaml
```

#### Command-Line Options

```bash
python3 cosmos_client.py <file> [options]

Options:
  --prompt-file FILE    Load prompt from YAML file
  --no-reasoning        Hide <think> sections if present
  -h, --help           Show help message
  test                 Test connection to server
```

#### Examples

```bash
# Quick image description
python3 cosmos_client.py warehouse.jpg

# Analyze driving scene with custom question
python3 cosmos_client.py dashcam.mp4 "What traffic hazards are present?"

# Use official embodied reasoning template
python3 cosmos_client.py robot_scene.png --prompt-file prompts/embodied_reasoning.yaml

# Process without showing reasoning sections
python3 cosmos_client.py video.mp4 --no-reasoning
```

### Configuration

**Server IP Address:**

Edit the script to point to your vLLM server:

```python
WORKSTATION_IP = "10.0.0.46"  # Change to your server's IP
```

Or create a configuration file (future enhancement):

```python
# config.yaml
server:
  host: "10.0.0.46"
  port: 8000
```

### Supported File Types

**Videos:**
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

**Images:**
- `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`

### Network Setup

If connecting from a different machine, ensure:

1. **vLLM server is accessible:**
   ```bash
   # On server, allow external connections
   vllm serve nvidia/Cosmos-Reason2-2B --host 0.0.0.0 --port 8000
   ```

2. **Firewall allows port 8000:**
   ```bash
   # On Ubuntu server
   sudo ufw allow 8000/tcp
   sudo ufw reload
   ```

3. **Test connectivity:**
   ```bash
   python3 cosmos_client.py test
   ```

---

## Server Setup

Both tools require a running vLLM server with Cosmos Reason 2. Here's how to set it up:

### Installation

```bash
# Clone the repository
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
cd cosmos-reason2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Start the Server

```bash
# Basic server (local access only)
vllm serve nvidia/Cosmos-Reason2-2B \
  --max-model-len 16384 \
  --media-io-kwargs '{"video": {"num_frames": -1}}'

# Server accessible from other devices
vllm serve nvidia/Cosmos-Reason2-2B \
  --host 0.0.0.0 \
  --port 8000 \
  --allowed-local-media-path "$(pwd)" \
  --max-model-len 16384 \
  --media-io-kwargs '{"video": {"num_frames": -1}}'
```

### Hardware Requirements

- **Cosmos Reason 2B:** ~24GB VRAM (RTX 6000 Ada, RTX 4090, etc.)
- **Cosmos Reason 8B:** ~32GB VRAM (A100, H100, etc.)

---

## Troubleshooting

### Connection Issues

**Problem:** `Cannot connect to server`

**Solutions:**
1. Verify server is running: `curl http://SERVER_IP:8000/v1/models`
2. Check firewall settings
3. Confirm IP address in scripts matches server
4. Test with `python3 cosmos_client.py test`

### Video Processing

**Problem:** Videos fail to process

**Solutions:**
1. Ensure videos are in supported format (`.mp4` recommended)
2. Check video file isn't corrupted
3. Verify server has sufficient memory
4. Try smaller videos first

### Import Errors

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
pip install requests pandas pyyaml
```

---

## Examples

### Use Case: Validate Generated Training Data

```bash
# Process a batch of synthetic videos
python video_evaluator.py -i ./synthetic_data/batch_001 -o ./validation_results

# Review results
cat validation_results/physics_evaluation_results.csv | grep "NO"
```

### Use Case: Remote Robot Telemetry Analysis

```bash
# From your laptop, analyze robot camera feed
python3 cosmos_client.py robot_camera.jpg "Identify any obstacles or hazards"
```

### Use Case: Real-time Demo

```bash
# Quick analysis for presentations
python3 cosmos_client.py demo_video.mp4 --prompt-file prompts/caption.yaml
```

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## Credits

Built by [Jonathan Stephens](https://github.com/yourusername) for the [Pixel Reconstruct](https://pixelreconstruct.com) community.
Based on [NVIDIA Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2).

---

## License

MIT License - See LICENSE file for details
