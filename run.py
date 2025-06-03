import os
import argparse
import numpy as np
from PIL import Image
import torch
import rembg
from omegaconf import OmegaConf
from torchvision import transforms
import torchvision.transforms.v2 as v2
from diffusers import DiffusionPipeline
from zero123plus import Zero123PlusPipeline, EulerAncestralDiscreteScheduler
from zero123plus import Zero123PlusPipeline, EulerAncestralDiscreteScheduler
from instant3d.utils import read_image, save_image, save_mesh, load_checkpoint

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
    pipe = Zero123PlusPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        torch_dtype=torch.float16,
        scheduler=EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing='trailing'
        )
    )
    pipe = pipe.to('cuda')
    
    # Generate multi-view images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to('cuda', torch.float16)
    with torch.no_grad():
        mv_images = pipe(image_tensor).images[0]  # Shape: [6, H, W, 3]
    
    # Resize and normalize
    images = torch.from_numpy(mv_images).permute(0, 3, 1, 2).to('cuda', torch.float16)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
    
    # Load model
    model = load_checkpoint(config.model).to('cuda', torch.float16)
    model.eval()
    
    # Generate mesh
    with torch.no_grad():
        mesh_data = model(images)
    
    # Save mesh
    mesh_path = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.input_path))[0] + '.obj')
    save_mesh(mesh_data, mesh_path, export_texmap=args.export_texmap)
    
    print(f"Saved mesh to {mesh_path}")

if __name__ == '__main__':
    main()