# Depth Anything V3 - Replicate Deployment

Unified transformer model for depth estimation with 3D Gaussian splat support.

## Source

- **Repository**: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Paper**: Depth Anything 3
- **License**: Apache 2.0 / CC BY-NC 4.0

## Capabilities

- **Monocular depth estimation** from single images
- **Multi-view depth estimation** for consistent geometry
- **Camera pose estimation** (extrinsics and intrinsics)
- **3D Gaussian prediction** for novel view synthesis
- **Pose-conditioned depth** when camera parameters are known

## Model Variants

- **DA3-Giant** (1.15B params): Best quality, highest accuracy
- **DA3-Large** (335M params): Balanced performance
- **DA3-Base** (97M params): Fast inference
- **DA3-Small** (80M params): Fastest, lightweight

This deployment uses **DA3-Giant** with Gaussian splat support for maximum quality.

## Input Formats

- **image**: Single RGB image (JPG, PNG, WebP)
- **mode**: Inference mode (monocular, multi-view, gaussian-splat)
- **model_size**: Model variant to use (giant, large, base, small)

## Output Formats

Depending on mode:
- **Depth map**: Float32 numpy array or visualization image
- **Point cloud**: PLY format
- **3D Gaussian splat**: For novel view synthesis
- **Camera parameters**: Intrinsics and extrinsics (JSON)
- **GLB export**: 3D model file

## Hardware Requirements

- GPU: A100 (40GB) recommended for Giant model
- Alternative: A10G (24GB) for Large model
- Minimum: T4 GPU for Base/Small models

## Usage Example

```python
import replicate

# Monocular depth estimation
output = replicate.run(
    "yourusername/depth-anything-v3:latest",
    input={
        "image": open("input.jpg", "rb"),
        "mode": "monocular",
        "model_size": "giant"
    }
)

# With Gaussian splat support
output = replicate.run(
    "yourusername/depth-anything-v3:latest",
    input={
        "image": open("input.jpg", "rb"),
        "mode": "gaussian-splat",
        "model_size": "giant",
        "export_format": "glb"
    }
)
```

## Local Testing

```bash
# Basic depth estimation
cog predict -i image=@input.jpg -i mode=monocular

# With Gaussian splat
cog predict -i image=@input.jpg -i mode=gaussian-splat -i export_format=ply
```

## Deployment

```bash
cog login
cog push r8.im/yourusername/depth-anything-v3
```

## Notes

- First run downloads model weights from Hugging Face (~4GB for Giant)
- Gaussian splat mode requires additional dependencies
- Supports batch processing for multi-view reconstruction
- Metric depth models provide real-world scale
