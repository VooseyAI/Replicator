#!/usr/bin/env python3
"""
Deployment automation for Replicator models.

This script helps automate the deployment of models to Replicate using Cog.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional


class ReplicateDeployer:
    """Automated deployment for Replicate models"""

    def __init__(self, model_dir: str, replicate_username: str):
        self.model_dir = Path(model_dir).resolve()
        self.replicate_username = replicate_username
        self.model_name = self.model_dir.name

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        print("Checking prerequisites...")

        # Check for cog
        try:
            result = subprocess.run(["cog", "--version"], capture_output=True, text=True)
            print(f"✓ Cog found: {result.stdout.strip()}")
        except FileNotFoundError:
            print("✗ Cog not found. Install it first:")
            print("  sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)")
            print("  sudo chmod +x /usr/local/bin/cog")
            return False

        # Check for Docker
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            print(f"✓ Docker found: {result.stdout.strip()}")
        except FileNotFoundError:
            print("✗ Docker not found. Please install Docker Desktop.")
            return False

        # Check for required files
        if not (self.model_dir / "cog.yaml").exists():
            print(f"✗ cog.yaml not found in {self.model_dir}")
            return False
        print("✓ cog.yaml found")

        if not (self.model_dir / "predict.py").exists():
            print(f"✗ predict.py not found in {self.model_dir}")
            return False
        print("✓ predict.py found")

        # Check for Replicate API token
        if not os.getenv("REPLICATE_API_TOKEN"):
            print("⚠ REPLICATE_API_TOKEN not set. You'll need it for deployment.")
            print("  Get your token from: https://replicate.com/account/api-tokens")
            print("  Then run: export REPLICATE_API_TOKEN=<your-token>")

        return True

    def test_local(self, test_image: Optional[str] = None):
        """Test the model locally using cog predict"""
        print(f"\n{'='*60}")
        print(f"Testing {self.model_name} locally...")
        print(f"{'='*60}\n")

        os.chdir(self.model_dir)

        # Build the container first
        print("Building Docker container...")
        result = subprocess.run(["cog", "build"], capture_output=False)

        if result.returncode != 0:
            print("✗ Build failed!")
            return False

        print("✓ Build successful!")

        # Run a test prediction if test image provided
        if test_image:
            print(f"\nRunning test prediction with {test_image}...")
            result = subprocess.run(
                ["cog", "predict", f"-i", f"image=@{test_image}"],
                capture_output=False
            )

            if result.returncode != 0:
                print("✗ Test prediction failed!")
                return False

            print("✓ Test prediction successful!")

        return True

    def deploy(self, create_deployment: bool = False):
        """Deploy the model to Replicate"""
        print(f"\n{'='*60}")
        print(f"Deploying {self.model_name} to Replicate...")
        print(f"{'='*60}\n")

        os.chdir(self.model_dir)

        # Login to Replicate
        print("Logging in to Replicate...")
        result = subprocess.run(["cog", "login"], capture_output=False)

        if result.returncode != 0:
            print("✗ Login failed!")
            return False

        # Push the model
        model_ref = f"r8.im/{self.replicate_username}/{self.model_name}"
        print(f"\nPushing to {model_ref}...")

        result = subprocess.run(["cog", "push", model_ref], capture_output=False)

        if result.returncode != 0:
            print("✗ Push failed!")
            return False

        print(f"✓ Successfully pushed to {model_ref}")

        if create_deployment:
            print("\n📝 To create a deployment:")
            print(f"   1. Visit https://replicate.com/{self.replicate_username}/{self.model_name}")
            print("   2. Click 'Deploy'")
            print("   3. Configure hardware and scaling settings")
            print("   4. Launch deployment")

        return True

    def run_workflow(self, test_image: Optional[str] = None, deploy_flag: bool = True):
        """Run the complete workflow"""
        if not self.check_prerequisites():
            sys.exit(1)

        if test_image:
            if not self.test_local(test_image):
                print("\n⚠ Local testing failed. Fix issues before deploying.")
                sys.exit(1)

        if deploy_flag:
            if not self.deploy():
                sys.exit(1)

        print(f"\n{'='*60}")
        print("✓ Deployment complete!")
        print(f"{'='*60}\n")
        print(f"Your model is available at:")
        print(f"  https://replicate.com/{self.replicate_username}/{self.model_name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy models to Replicate using Cog"
    )
    parser.add_argument(
        "model",
        choices=["sam3d-objects", "depth-anything-v3", "all"],
        help="Model to deploy"
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Your Replicate username"
    )
    parser.add_argument(
        "--test-image",
        help="Path to test image for local validation"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run local tests, don't deploy"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip local testing and deploy directly"
    )

    args = parser.parse_args()

    # Get the Replicator directory (parent of common)
    replicator_dir = Path(__file__).parent.parent

    models_to_deploy = []
    if args.model == "all":
        models_to_deploy = ["sam3d-objects", "depth-anything-v3"]
    else:
        models_to_deploy = [args.model]

    for model in models_to_deploy:
        model_dir = replicator_dir / model

        if not model_dir.exists():
            print(f"✗ Model directory not found: {model_dir}")
            sys.exit(1)

        deployer = ReplicateDeployer(
            model_dir=str(model_dir),
            replicate_username=args.username
        )

        test_image = args.test_image if not args.skip_test else None
        deploy_flag = not args.test_only

        deployer.run_workflow(
            test_image=test_image,
            deploy_flag=deploy_flag
        )


if __name__ == "__main__":
    main()
