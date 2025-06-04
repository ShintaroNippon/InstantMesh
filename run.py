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
from models.lrm import InstantNeRF
from utils.mesh_util import xatlas_uvmap

def read_image(image_path):
    return Image.open(image_path).convert('RGBA')

def save_image(image, path):
    image.save(path, format='PNG')

def save_mesh(mesh_data, path, export_texmap=False):
    mesh = trimesh.Trimesh(vertices=mesh_data['vertices'], faces=mesh_data['faces'])
    if export_texmap and 'uvs' in mesh_data:
        xatlas_uvmap(mesh, uvs=mesh_data['uvs'])
    mesh.export(path)

def load_checkpoint(config):
    checkpoint_path = "/teamspace/studios/this_studio/InstantMesh/checkpoints/instant_mesh_large.ckpt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = InstantNeRF(config.model).to('cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config yaml file.')
    parser.add_argument('input_path', type=str, help='Path to input image.')
    parser.add_argument('--output_path', type=str, default='output', help='Output directory.')
    parser.add_argument('--no_rembg', action='store_true', help='Do not remove background.')
    parser.add_argument('--export_texmap', action='store_true', help='Export texture map.')
    return parser.parse_args()

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    os.makedirs(args.output_path, exist_ok=True)
    
    # Set seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load and preprocess image
    image = read_image(args.input_path)
    if not args.no_rembg:
        image = rembg.remove(image)
        save_image(image, os.path.join(args.output_path, 'input.png'))
    else:
        save_image(image, os.path.join(args.output_path, 'input.png'))
    
    # Initialize pipeline
    try:
        pipe = Zero123PlusPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            torch_dtype=torch.float16
        )
        scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing='trailing'
        )
        pipe.scheduler = scheduler
        pipe = pipe.to('cuda')
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Zero123PlusPipeline: {str(e)}")
    
    # Generate multi-view images
    with torch.no_grad():
        mv_images = pipe(image).images[0]  # PIL Image
    
    # Convert PIL Image to NumPy array and then to tensor
    mv_images_np = np.array(mv_images)  # Convert PIL Image to NumPy array
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    images = torch.from_numpy(mv_images_np).permute(0, 3, 1, 2).to('cuda', torch.float16)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
    
    # Load model
    model = load_checkpoint(config).to('cuda', torch.float16)
    model.eval()
    
    # Generate mesh
    with torch.no_grad():
        mesh_data = model(images)
    
    # Save mesh
    output_subdir = os.path.join(args.output_path, 'instant-mesh-large', 'meshes')
    os.makedirs(output_subdir, exist_ok=True)
    mesh_path = os.path.join(output_subdir, os.path.splitext(os.path.basename(args.input_path))[0] + '.obj')
    save_mesh(mesh_data, mesh_path, export_texmap=args.export_texmap)
    
    print(f"Saved mesh to {mesh_path}")

if __name__ == '__main__':
    main()