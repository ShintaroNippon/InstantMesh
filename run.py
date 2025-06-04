import os
import argparse
import numpy as np
from PIL import Image
import torch
import rembg
from omegaconf import OmegaConf
from torchvision import transforms
import torchvision.transforms.v2 as v2
from zero123plus import Zero123PlusPipeline, EulerAncestralDiscreteScheduler
import trimesh
import xatlas
from src.models.lrm_mesh import InstantMesh
from utils.mesh_util import xatlas_uvmap

def read_image(image_path):
    print(f"Reading image: {image_path}")
    return Image.open(image_path).convert('RGBA')

def save_image(image, path):
    print(f"Saving image to: {path}")
    image.save(path, format='PNG')

def save_mesh(mesh_data, path, export_texmap=False):
    print(f"Saving mesh to: {path}")
    try:
        mesh = trimesh.Trimesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
        if export_texmap and 'uvs' in mesh_data:
            print("Exporting texture map")
            xatlas_uvmap(mesh, uvs=mesh_data['uvs'])
        mesh.export(path)
    except Exception as e:
        print(f"Failed to save mesh: {str(e)}")
        raise

def load_checkpoint(config):
    checkpoint_path = "/teamspace/studios/this_studio/InstantMesh/checkpoints/instant_mesh_large.ckpt"
    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not hasattr(config, 'model_config'):
        raise ValueError(f"Configuration is missing 'model_config' key: {config}")
    
    try:
        model = InstantMesh(config.model_config).to('cpu')
    except Exception as e:
        print(f"Failed to initialize InstantMesh: {str(e)}")
        raise
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Failed to load checkpoint state dict: {str(e)}")
        raise
    
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Run InstantMesh for 3D model generation")
    parser.add_argument('--config', type=str, default='configs/instant-mesh-large.yaml')
    parser.add_argument('input_path', type=str, help='Path to input image')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--no_rembg', action='store_true')
    parser.add_argument('--export_texmap', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Config: {args.config}, Input: {args.input_path}, Output: {args.output_path}")
    
    try:
        config = OmegaConf.load(args.config)
        print(f"Loaded config: {config}")
    except Exception as e:
        print(f"Failed to load config file {args.config}: {str(e)}")
        raise
    
    os.makedirs(args.output_path, exist_ok=True)
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    image = read_image(args.input_path)
    if not args.no_rembg:
        print("Removing background")
        image = rembg.remove(image)
    save_image(image, os.path.join(args.output_path, 'input.png'))
    
    try:
        print("Initializing Zero123PlusPipeline")
        pipe = Zero123PlusPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            torch_dtype=torch.float16
        )
        scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing='trailing'
        )
        pipe.scheduler = scheduler
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipe = pipe.to(device)
        print(f"Pipeline moved to {device}")
    except Exception as e:
        print(f"Failed to initialize pipeline: {str(e)}")
        raise
    
    print("Generating multi-view images")
    with torch.no_grad():
        result = pipe(image)
        mv_images = result.images
    
    if isinstance(mv_images, list):
        mv_images_np = np.stack([np.array(img) for img in mv_images])
    else:
        mv_images_np = np.array(mv_images)[None, ...]
    
    print(f"mv_images_np shape: {mv_images_np.shape}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    try:
        images = torch.stack([transform(img) for img in mv_images_np]).to(device, torch.float16)
        print(f"Images tensor shape: {images.shape}")
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
    except Exception as e:
        print(f"Failed to preprocess images: {str(e)}")
        raise
    
    num_views = images.shape[0]
    cameras = torch.eye(4, device=device, dtype=torch.float16).unsqueeze(0).repeat(num_views, 1, 1)
    cameras = cameras.view(num_views, -1)  # Flatten to [num_views, 16]
    render_cameras = cameras.clone()
    render_size = 512
    
    print(f"Cameras shape: {cameras.shape}, Render cameras shape: {render_cameras.shape}, Render size: {render_size}")
    
    print("Loading InstantMesh model")
    try:
        model = load_checkpoint(config)
        model = model.to(device, torch.float16 if device == 'cuda' else torch.float32)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise
    
    print("Generating mesh")
    try:
        with torch.no_grad():
            mesh_data = model(images, cameras, render_cameras, render_size)
            print(f"Mesh data keys: {mesh_data.keys()}")
    except Exception as e:
        print(f"Failed to generate mesh: {str(e)}")
        raise
    
    output_subdir = os.path.join(args.output_path, 'instant-mesh-large', 'meshes')
    os.makedirs(output_subdir, exist_ok=True)
    mesh_path = os.path.join(output_subdir, os.path.splitext(os.path.basename(args.input_path))[0] + '.obj')
    save_mesh(mesh_data, mesh_path, export_texmap=args.export_texmap)
    
    print(f"Saved mesh to {mesh_path}")

if __name__ == '__main__':
    main()