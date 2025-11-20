# Deployment Guide

Complete guide for deploying models to Replicate using this project.

## Prerequisites

### 1. Install Cog

```bash
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
sudo chmod +x /usr/local/bin/cog
```

Verify installation:
```bash
cog --version
```

### 2. Install Docker

Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/) for your platform.

Verify installation:
```bash
docker --version
```

### 3. Set Up Replicate Account

1. Create account at [replicate.com](https://replicate.com)
2. Get your API token from [Account Settings](https://replicate.com/account/api-tokens)
3. Set environment variable:

```bash
export REPLICATE_API_TOKEN=<your-token-here>
```

Add to your shell profile (~/.bashrc, ~/.zshrc) to persist:
```bash
echo 'export REPLICATE_API_TOKEN=<your-token>' >> ~/.zshrc
```

## Deployment Methods

### Method 1: Automated Deployment Script (Recommended)

Use the provided Python script for automated deployment:

```bash
# Deploy a single model with testing
python3 common/deploy.py sam3d-objects \
    --username your-replicate-username \
    --test-image path/to/test-image.jpg

# Deploy without testing
python3 common/deploy.py depth-anything-v3 \
    --username your-replicate-username \
    --skip-test

# Deploy all models
python3 common/deploy.py all \
    --username your-replicate-username

# Test only (no deployment)
python3 common/deploy.py sam3d-objects \
    --username your-replicate-username \
    --test-image test.jpg \
    --test-only
```

### Method 2: Manual Deployment

#### Step 1: Local Testing

```bash
cd sam3d-objects  # or depth-anything-v3

# Build the container
cog build

# Test with sample input
cog predict -i image=@test-image.jpg
```

#### Step 2: Create Model on Replicate

1. Visit [replicate.com/create](https://replicate.com/create)
2. Choose a model name (e.g., `sam3d-objects`)
3. Select visibility (start with Private)
4. Choose hardware (A100 40GB recommended)

#### Step 3: Push Model

```bash
# Login to Replicate
cog login

# Push the model (replace 'username' with your actual username)
cog push r8.im/username/sam3d-objects
```

#### Step 4: Create Deployment (Optional)

For production use:

1. Go to your model page: `replicate.com/username/sam3d-objects`
2. Click "Deploy" button
3. Configure:
   - Deployment name
   - GPU hardware type
   - Min/max instances for autoscaling
4. Click "Create deployment"

## Model-Specific Deployment

### SAM 3D Objects

**Hardware Requirements:**
- Development: T4 GPU
- Production: A100 40GB

**Test Command:**
```bash
cd sam3d-objects
cog predict -i image=@input.jpg -i mask=@mask.png
```

**Expected Output:** PLY file with 3D Gaussian splat

### Depth Anything V3

**Hardware Requirements:**
- Giant model: A100 40GB
- Large model: A10G 24GB
- Base/Small: T4 GPU

**Test Commands:**
```bash
cd depth-anything-v3

# Monocular depth
cog predict -i image=@input.jpg -i mode=monocular -i model_size=giant

# Gaussian splat mode
cog predict -i image=@input.jpg -i mode=gaussian-splat -i export_format=ply
```

**Expected Outputs:** Depth maps, PLY, GLB, or NPZ files

## Troubleshooting

### Build Failures

**Issue:** Dependency installation fails

```bash
# Check cog.yaml syntax
cat cog.yaml

# Try building with verbose output
cog build --verbose
```

**Issue:** CUDA version mismatch

- Update PyTorch version in cog.yaml
- Ensure CUDA version matches your model requirements

### Push Failures

**Issue:** Authentication failed

```bash
# Re-login to Replicate
cog login

# Verify token is set
echo $REPLICATE_API_TOKEN
```

**Issue:** Disk space

```bash
# Clean up old Docker images
docker system prune -a
```

### Runtime Errors

**Issue:** Model weights not downloading

- Check internet connectivity
- Verify Hugging Face access
- Try downloading weights manually

**Issue:** Out of memory

- Reduce batch size in predict.py
- Use smaller model variant
- Upgrade to larger GPU instance

## Monitoring and Updates

### Check Deployment Status

```bash
# View via web dashboard
open https://replicate.com/username/model-name

# Or use Replicate API
curl -X GET https://api.replicate.com/v1/models/username/model-name \
  -H "Authorization: Token $REPLICATE_API_TOKEN"
```

### Update Model

```bash
cd model-directory

# Make changes to predict.py or cog.yaml

# Push update (creates new version)
cog push r8.im/username/model-name
```

**Note:** Deployments can use rolling updates for zero-downtime deployments.

### View Logs

1. Go to model page on Replicate
2. Click on "Predictions" tab
3. View individual prediction logs

## Cost Optimization

### Development
- Use T4 GPUs ($0.000225/sec)
- Set min instances to 0
- Use Private visibility

### Production
- Choose appropriate GPU tier
- Configure autoscaling (min/max instances)
- Monitor usage via dashboard
- Consider reserved capacity for high volume

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Replicate

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Install Cog
        run: |
          sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_Linux_x86_64
          sudo chmod +x /usr/local/bin/cog

      - name: Login to Replicate
        env:
          REPLICATE_API_TOKEN: ${{ secrets.REPLICATE_API_TOKEN }}
        run: cog login

      - name: Push model
        run: |
          cd sam3d-objects
          cog push r8.im/${{ secrets.REPLICATE_USERNAME }}/sam3d-objects
```

## Next Steps

1. **Test your deployment**
   ```python
   import replicate

   output = replicate.run(
       "username/model-name:latest",
       input={"image": open("test.jpg", "rb")}
   )
   ```

2. **Monitor performance**
   - Track latency and costs via Replicate dashboard
   - Optimize hardware selection based on usage

3. **Iterate and improve**
   - Update predict.py for better performance
   - Add more input/output options
   - Optimize model loading and caching

## Resources

- [Replicate Documentation](https://replicate.com/docs)
- [Cog GitHub](https://github.com/replicate/cog)
- [Replicate Python Client](https://github.com/replicate/replicate-python)
- [Example Models](https://replicate.com/explore)
