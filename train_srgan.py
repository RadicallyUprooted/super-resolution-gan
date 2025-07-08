import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from data_loader import DatasetFromFolder
from model import Generator, Discriminator, VGGLoss

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    if args.test_image_path and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Load test image
    test_img_lr_tensor = None
    if args.test_image_path:
        try:
            test_img_lr = Image.open(args.test_image_path).convert('RGB')
            test_img_lr_tensor = ToTensor()(test_img_lr).unsqueeze(0).to(device)
            print(f"Loaded test image from: {args.test_image_path}")
        except FileNotFoundError:
            print(f"Warning: Test image not found at {args.test_image_path}. Skipping test image generation.")
            args.test_image_path = None

    # Data
    train_dataset = DatasetFromFolder(args.image_dir, crop_size=args.crop_size, upscale_factor=args.upscale_factor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Models
    generator = Generator(n_residual_blocks=args.n_res_blocks).to(device)
    discriminator = Discriminator().to(device)

    if args.generator_checkpoint:
        generator.load_state_dict(torch.load(args.generator_checkpoint))
        print(f"Loaded generator checkpoint: {args.generator_checkpoint}")
    if args.discriminator_checkpoint:
        discriminator.load_state_dict(torch.load(args.discriminator_checkpoint))
        print(f"Loaded discriminator checkpoint: {args.discriminator_checkpoint}")

    # Losses and Optimizers
    vgg_loss = VGGLoss(device=device)
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr_g)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr_d)

    use_amp = device.type == 'cuda'
    scaler_g = torch.amp.GradScaler(enabled=use_amp)
    scaler_d = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for lr, hr in progress_bar:
            lr, hr = lr.to(device), hr.to(device)

            # --- Train Discriminator ---
            optimizer_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                sr = generator(lr).detach()
                real_output = discriminator(hr)
                fake_output = discriminator(sr)
                loss_d_real = adversarial_loss(real_output, torch.ones_like(real_output))
                loss_d_fake = adversarial_loss(fake_output, torch.zeros_like(fake_output))
                loss_d = (loss_d_real + loss_d_fake) / 2

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # --- Train Generator ---
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                sr = generator(lr)
                fake_output = discriminator(sr)
                loss_g_adv = adversarial_loss(fake_output, torch.ones_like(fake_output))
                loss_g_vgg = args.vgg_weight * vgg_loss(sr, hr)
                loss_g = loss_g_vgg + args.adv_weight * loss_g_adv

            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            progress_bar.set_postfix(loss_g=f"{loss_g.item():.4f}", loss_d=f"{loss_d.item():.4f}")

        # --- End of Epoch ---
        print(f"Epoch {epoch} finished.")

        # Generate and save test image
        if args.test_image_path:
            generator.eval()
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    sr_img_tensor = generator(test_img_lr_tensor)
                
                sr_img_tensor = sr_img_tensor.squeeze(0).cpu().float()
                sr_img_tensor = torch.clamp(sr_img_tensor, 0, 1)
                sr_img_pil = ToPILImage()(sr_img_tensor)
                save_path = os.path.join(args.output_dir, f"srgan_epoch_{epoch}.png")
                sr_img_pil.save(save_path)
                print(f"Saved test image for epoch {epoch} to {save_path}")

        if epoch % args.save_interval == 0:
            torch.save(generator.state_dict(), f"generator_srgan_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"discriminator_srgan_epoch_{epoch}.pth")
            print(f"Saved models at epoch {epoch}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an SRGAN model.")
    # Data
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the training images directory.")
    parser.add_argument('--crop_size', type=int, default=24, help="Size of the low-resolution crop.")
    parser.add_argument('--upscale_factor', type=int, default=4, help="Upscale factor for super-resolution.")
    # Model
    parser.add_argument('--n_res_blocks', type=int, default=16, help="Number of residual blocks in the generator.")
    # Training
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for the data loader.")
    parser.add_argument('--lr_g', type=float, default=1e-4, help="Learning rate for the generator.")
    parser.add_argument('--lr_d', type=float, default=1e-4, help="Learning rate for the discriminator.")
    parser.add_argument('--vgg_weight', type=float, default=0.006, help="Weight for the VGG/content loss.")
    parser.add_argument('--adv_weight', type=float, default=1e-3, help="Weight for the adversarial loss.")
    # Checkpoints & Output
    parser.add_argument('--generator_checkpoint', type=str, default=None, help="Path to a pre-trained generator checkpoint.")
    parser.add_argument('--discriminator_checkpoint', type=str, default=None, help="Path to a pre-trained discriminator checkpoint.")
    parser.add_argument('--save_interval', type=int, default=5, help="Save the model every N epochs.")
    parser.add_argument('--test_image_path', type=str, default=None, help="Path to a low-resolution test image to evaluate at each epoch.")
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save the upscaled test images.")

    args = parser.parse_args()
    main(args)          



