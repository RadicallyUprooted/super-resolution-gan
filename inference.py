import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from model import Generator

def main():
    parser = argparse.ArgumentParser(description="SRResNet/SRGAN Inference Script")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the generator checkpoint file (.pth)")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input image or directory of images")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the directory where super-resolved images will be saved")
    parser.add_argument("--upscale_factor", type=int, default=4,
                        help="Upscale factor used during training (e.g., 4 for 4x)")

    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Generator model
    generator = Generator(n_residual_blocks=16).to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    generator.eval()
    print(f"Generator loaded from {args.checkpoint}")

    # Image transformations
    preprocess = transforms.ToTensor()
    postprocess = transforms.ToPILImage()

    # Bicubic upsampling transformation
    class BicubicUpsample(object):
        def __init__(self, scale_factor):
            self.scale_factor = scale_factor

        def __call__(self, img):
            new_size = (int(img.width * self.scale_factor), int(img.height * self.scale_factor))
            return img.resize(new_size, resample=Image.BICUBIC)

    bicubic_upsample = BicubicUpsample(scale_factor=args.upscale_factor)

    os.makedirs(args.output_path, exist_ok=True)

    if os.path.isdir(args.input_path):
        image_files = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        print(f"Found {len(image_files)} images in {args.input_path}")
    else:
        image_files = [args.input_path]
        print(f"Processing single image: {args.input_path}")

    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')

            # Save bicubic upscaled image
            bicubic_output_filename = os.path.join(args.output_path, os.path.basename(img_path).replace('.', '_bicubic.'))
            bicubic_upsample(img).save(bicubic_output_filename)
            print(f"Bicubic upscaled image saved to {bicubic_output_filename}")

            # Prepare image for generator
            lr_image = preprocess(img).unsqueeze(0).to(device) # Add batch dimension

            with torch.no_grad():
                sr_image = generator(lr_image)

            # Save SRGAN upscaled image
            srgan_output_filename = os.path.join(args.output_path, os.path.basename(img_path).replace('.', '_srgan.'))
            postprocess(sr_image.squeeze(0).cpu()).save(srgan_output_filename) # Remove batch dimension
            print(f"SRGAN upscaled image saved to {srgan_output_filename}")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    main()
