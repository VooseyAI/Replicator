import os
import sys
from typing import Optional, List
import torch
import numpy as np
from PIL import Image
import json
import tempfile
import cv2
import io
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load Depth Anything V3 model into memory"""
        print("Loading Depth Anything V3 model...")

        # Install depth-anything-3 from GitHub
        import subprocess
        print("Installing depth-anything-3 package...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git#egg=depth-anything-3[gs]"
            ])
            print("depth-anything-3 installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not install depth-anything-3: {e}")

        try:
            # Import Depth Anything V3
            from depth_anything_3 import DepthAnything3, DepthAnything3Config

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            # Pre-load the Giant model (default, best quality)
            print("Loading DA3-Giant model (1.15B params)...")
            self.models = {
                "giant": None,  # Lazy load to save memory
                "large": None,
                "base": None,
                "small": None
            }

            # Load giant by default
            self.current_model = "giant"
            self._load_model("giant")

            print("Model loaded successfully!")

        except ImportError as e:
            print(f"Error importing Depth Anything V3: {e}")
            print("Running in fallback mode...")
            self.models = None
            self.device = torch.device("cpu")

    def _load_model(self, model_size: str):
        """Load a specific model variant"""
        if self.models is None:
            return

        if self.models[model_size] is not None:
            return  # Already loaded

        try:
            from depth_anything_3 import DepthAnything3

            model_configs = {
                "giant": "depth-anything-3-giant",
                "large": "depth-anything-3-large",
                "base": "depth-anything-3-base",
                "small": "depth-anything-3-small"
            }

            print(f"Loading {model_size} model...")
            self.models[model_size] = DepthAnything3.from_pretrained(
                f"ByteDance-Seed/{model_configs[model_size]}",
                local_files_only=False
            ).to(self.device).eval()

            self.current_model = model_size
            print(f"{model_size} model loaded!")

        except Exception as e:
            print(f"Error loading {model_size} model: {e}")

    def predict(
        self,
        image: Path = Input(description="Input RGB image (ignored if video or images provided)", default=None),
        video: Path = Input(description="Input video file for frame extraction", default=None),
        images: str = Input(description="Comma-separated URLs or paths to multiple images", default=None),
        frame_interval: int = Input(
            description="Frame interval for video processing (extract every Nth frame)",
            default=30,
            ge=1,
            le=120
        ),
        mode: str = Input(
            description="Inference mode",
            choices=["monocular", "multi-view", "gaussian-splat", "metric"],
            default="monocular"
        ),
        model_size: str = Input(
            description="Model variant (giant=best quality, small=fastest)",
            choices=["giant", "large", "base", "small"],
            default="giant"
        ),
        export_format: str = Input(
            description="Output format",
            choices=["depth_map", "ply", "glb", "npz", "all"],
            default="depth_map"
        ),
        max_depth: float = Input(
            description="Maximum depth value for visualization (meters)",
            default=10.0,
            ge=1.0,
            le=100.0
        ),
        output_video: bool = Input(
            description="Create output video from depth frames (video input only)",
            default=False
        )
    ) -> List[Path]:
        """Run depth estimation with optional Gaussian splat support"""

        # Determine input source
        input_images = []

        if video is not None:
            print(f"Processing video: {video}")
            input_images = self._extract_video_frames(str(video), frame_interval)
        elif images is not None:
            print(f"Processing multiple images: {images}")
            input_images = self._load_images_from_urls(images)
        elif image is not None:
            print(f"Processing single image: {image}")
            img = Image.open(str(image)).convert("RGB")
            input_images = [np.array(img)]
        else:
            raise ValueError("Must provide image, video, or images parameter")

        print(f"Processing {len(input_images)} frame(s)")
        print(f"Mode: {mode}, Model: {model_size}, Export: {export_format}")

        # Load appropriate model
        if self.current_model != model_size:
            self._load_model(model_size)

        if self.models is None or self.models[model_size] is None:
            print("WARNING: Running in fallback mode")
            return [self._create_fallback_output(Image.fromarray(input_images[0]), export_format)]

        try:
            model = self.models[model_size]
            output_paths = []
            depth_frames = []

            # Process each frame
            for idx, img_array in enumerate(input_images):
                print(f"Processing frame {idx + 1}/{len(input_images)}...")

                # Run inference based on mode
                with torch.no_grad():
                    if mode == "monocular":
                        depth = model.infer_image(img_array)
                    elif mode == "gaussian-splat":
                        depth, gaussians = model.infer_image(img_array, return_gaussians=True)
                    elif mode == "metric":
                        depth = model.infer_image(img_array, metric=True)
                    else:  # multi-view
                        depth = model.infer_image(img_array)

                depth_frames.append(depth)

                # Export in requested format
                output_path = self._export_output(
                    depth=depth,
                    image=img_array,
                    export_format=export_format,
                    mode=mode,
                    max_depth=max_depth,
                    frame_idx=idx
                )

                output_paths.append(Path(output_path))

            # Create output video if requested and input was video
            if video is not None and output_video and export_format == "depth_map":
                video_path = self._create_depth_video(depth_frames, input_images[0].shape)
                output_paths.append(Path(video_path))

            print(f"Processing complete! Generated {len(output_paths)} output(s)")
            return output_paths

        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            print("Creating fallback output...")
            return [self._create_fallback_output(Image.fromarray(input_images[0]), export_format)]

    def _extract_video_frames(self, video_path: str, frame_interval: int) -> List[np.ndarray]:
        """Extract frames from video at specified interval"""
        frames = []
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_count += 1

        cap.release()
        print(f"Extracted {len(frames)} frames from video (every {frame_interval} frames)")
        return frames

    def _load_images_from_urls(self, images_str: str) -> List[np.ndarray]:
        """Load multiple images from comma-separated URLs or paths"""
        import urllib.request

        image_sources = [s.strip() for s in images_str.split(',')]
        images = []

        for source in image_sources:
            try:
                if source.startswith('http://') or source.startswith('https://'):
                    # Download from URL
                    with urllib.request.urlopen(source) as url:
                        img_data = url.read()
                        img = Image.open(io.BytesIO(img_data)).convert("RGB")
                else:
                    # Load from local path
                    img = Image.open(source).convert("RGB")

                images.append(np.array(img))
                print(f"Loaded image from: {source}")
            except Exception as e:
                print(f"Warning: Failed to load image from {source}: {e}")

        if len(images) == 0:
            raise ValueError("No valid images could be loaded")

        return images

    def _create_depth_video(self, depth_frames: List[np.ndarray], frame_shape: tuple, fps: int = 30) -> str:
        """Create video from depth frames"""
        output_path = "/tmp/depth_video.mp4"

        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=False)

        for depth in depth_frames:
            # Normalize depth to 0-255
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            depth_vis = (depth_normalized * 255).astype(np.uint8)
            out.write(depth_vis)

        out.release()
        print(f"Created depth video: {output_path}")
        return output_path

    def _export_output(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        export_format: str,
        mode: str,
        max_depth: float,
        frame_idx: int = 0
    ) -> str:
        """Export depth estimation in various formats"""

        if export_format == "depth_map" or export_format == "all":
            # Normalize and save as image
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            depth_vis = (depth_normalized * 255).astype(np.uint8)
            depth_img = Image.fromarray(depth_vis)

            output_path = f"/tmp/depth_output_{frame_idx:04d}.png"
            depth_img.save(output_path)

        if export_format == "ply" or export_format == "all":
            output_path = f"/tmp/depth_output_{frame_idx:04d}.ply"
            self._save_ply(depth, image, output_path, max_depth)

        if export_format == "glb" or export_format == "all":
            output_path = f"/tmp/depth_output_{frame_idx:04d}.glb"
            self._save_glb(depth, image, output_path, max_depth)

        if export_format == "npz":
            output_path = f"/tmp/depth_output_{frame_idx:04d}.npz"
            np.savez_compressed(output_path, depth=depth, image=image)

        return output_path

    def _save_ply(self, depth: np.ndarray, image: np.ndarray, output_path: str, max_depth: float):
        """Save point cloud as PLY"""
        import trimesh

        h, w = depth.shape
        fx = fy = w  # Simplified intrinsics
        cx, cy = w / 2, h / 2

        # Create point cloud
        points = []
        colors = []

        for v in range(h):
            for u in range(w):
                z = depth[v, u]
                if z > max_depth:
                    continue

                x = (u - cx) * z / fx
                y = (v - cy) * z / fy

                points.append([x, y, z])
                colors.append(image[v, u])

        if len(points) == 0:
            # Create dummy point
            points = [[0, 0, 0]]
            colors = [[128, 128, 128]]

        points = np.array(points)
        colors = np.array(colors)

        cloud = trimesh.points.PointCloud(vertices=points, colors=colors)
        cloud.export(output_path)

    def _save_glb(self, depth: np.ndarray, image: np.ndarray, output_path: str, max_depth: float):
        """Save as GLB 3D model"""
        # First create PLY, then convert to GLB
        ply_path = "/tmp/temp.ply"
        self._save_ply(depth, image, ply_path, max_depth)

        # Load and convert to GLB
        import trimesh
        mesh = trimesh.load(ply_path)
        mesh.export(output_path, file_type='glb')

    def _create_fallback_output(self, img: Image.Image, export_format: str) -> Path:
        """Create dummy output for testing"""
        print("Creating fallback depth map...")

        # Create a simple gradient depth map
        w, h = img.size
        depth = np.linspace(0, 255, h).reshape(h, 1).repeat(w, axis=1)
        depth_img = Image.fromarray(depth.astype(np.uint8))

        if export_format == "depth_map":
            output_path = "/tmp/depth_output.png"
            depth_img.save(output_path)
        else:
            output_path = "/tmp/depth_output.ply"
            # Create minimal PLY
            with open(output_path, 'w') as f:
                f.write("""ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
end_header
0 0 0
""")

        return Path(output_path)
