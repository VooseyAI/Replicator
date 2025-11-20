# Replicator

Automated deployment pipeline for converting research models and Jupyter notebooks into Replicate-hosted inference APIs.

## Overview

This project provides a streamlined workflow to convert complex machine learning models (particularly those distributed as Jupyter notebooks or research code) into production-ready Replicate APIs using Cog containerization.

## Current Models

### 1. SAM 3D Objects
- **Source**: [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects)
- **Description**: Foundation model for reconstructing full 3D shape geometry, texture, and layout from a single image
- **Output Format**: 3D Gaussian splat representations (PLY format)
- **Location**: `./sam3d-objects/`

### 2. Depth Anything V3
- **Source**: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Description**: Unified transformer model for depth estimation with Gaussian splat support
- **Capabilities**: Monocular/multi-view depth, camera pose estimation, 3D Gaussian prediction
- **Location**: `./depth-anything-v3/`

## Project Structure

```
Replicator/
├── README.md                 # This file
├── common/                   # Shared utilities and helpers
├── sam3d-objects/           # SAM 3D Objects deployment
│   ├── cog.yaml             # Cog configuration
│   ├── predict.py           # Prediction interface
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Model-specific documentation
└── depth-anything-v3/       # Depth Anything V3 deployment
    ├── cog.yaml             # Cog configuration
    ├── predict.py           # Prediction interface
    ├── requirements.txt     # Python dependencies
    └── README.md            # Model-specific documentation
```

## Prerequisites

1. **Cog**: Install via `sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m) && sudo chmod +x /usr/local/bin/cog`
2. **Replicate Account**: Sign up at [replicate.com](https://replicate.com)
3. **Replicate API Token**: Set as `REPLICATE_API_TOKEN` environment variable
4. **Docker**: Required for Cog to build containers

## Quick Start

### Local Testing

```bash
# Test SAM 3D Objects
cd sam3d-objects
cog predict -i image=@input.jpg -i mask=@mask.png

# Test Depth Anything V3
cd depth-anything-v3
cog predict -i image=@input.jpg
```

### Deploy to Replicate

```bash
# SAM 3D Objects
cd sam3d-objects
cog login
cog push r8.im/yourusername/sam3d-objects

# Depth Anything V3
cd depth-anything-v3
cog login
cog push r8.im/yourusername/depth-anything-v3
```

## Development Workflow

1. **Research Phase**: Examine source repository structure and dependencies
2. **Configuration**: Create `cog.yaml` with appropriate Python version and packages
3. **Implementation**: Build `predict.py` with setup() and predict() methods
4. **Local Testing**: Use `cog predict` to verify functionality
5. **Deployment**: Push to Replicate with `cog push`
6. **Monitoring**: Track performance via Replicate dashboard

## Common Utilities

The `common/` directory contains shared code:
- Model download helpers
- Input/output format converters
- Validation utilities
- Deployment scripts

## License

Each model retains its original license:
- SAM 3D Objects: SAM License
- Depth Anything V3: Apache 2.0 / CC BY-NC 4.0

## Resources

- [Replicate Documentation](https://replicate.com/docs)
- [Cog Documentation](https://github.com/replicate/cog)
- [Replicate Python Client](https://github.com/replicate/replicate-python)
