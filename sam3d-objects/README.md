# SAM 3D Objects - Replicate Deployment

Foundation model that reconstructs full 3D shape geometry, texture, and layout from a single image.

## Source

- **Repository**: [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects)
- **Paper**: SAM 3D Objects
- **License**: SAM License (Research and Commercial use with attribution)

## Capabilities

- Reconstruct 3D objects from single 2D images
- Generate 3D Gaussian splat representations
- Output PLY format files with geometry and texture
- Handle multiple objects with mask-based selection

## Input Formats

- **image**: RGB image (JPG, PNG)
- **mask**: Binary mask indicating object(s) to reconstruct (optional, if not provided will process full image)

## Output Formats

- **PLY file**: 3D Gaussian splat representation
- Contains geometry, texture, and pose information

## Model Checkpoints

The model uses pre-trained weights from the Hugging Face hub, automatically downloaded on first run.

## Hardware Requirements

- GPU: A100 (40GB) recommended for best performance
- Minimum: T4 GPU for inference

## Usage Example

```python
import replicate

output = replicate.run(
    "yourusername/sam3d-objects:latest",
    input={
        "image": open("input.jpg", "rb"),
        "mask": open("mask.png", "rb")
    }
)

# Download the PLY file
with open("output.ply", "wb") as f:
    f.write(output.read())
```

## Local Testing

```bash
cog predict -i image=@input.jpg -i mask=@mask.png
```

## Deployment

```bash
cog login
cog push r8.im/yourusername/sam3d-objects
```

## Notes

- First run will download model checkpoints (~several GB)
- Processing time varies with image resolution and complexity
- Supports batch processing of multiple objects via separate masks
