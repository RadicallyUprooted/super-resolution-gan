import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from data_loader import DatasetFromFolder
from model import Generator

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

    # Model
    model = Generator(n_residual_blocks=args.n_res_blocks).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Loss and Optimizer
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for lr, hr in progress_bar:
            lr, hr = lr.to(device), hr.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                sr = model(lr)
                loss = criterion(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

        # Generate and save test image
        if args.test_image_path:
            model.eval()
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    sr_img_tensor = model(test_img_lr_tensor)
                
                sr_img_tensor = sr_img_tensor.squeeze(0).cpu().float()
                sr_img_tensor = torch.clamp(sr_img_tensor, 0, 1)
                sr_img_pil = ToPILImage()(sr_img_tensor)
                save_path = os.path.join(args.output_dir, f"epoch_{epoch}_res.png")
                sr_img_pil.save(save_path)
                print(f"Saved test image for epoch {epoch} to {save_path}")

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), f"generator_epoch_{epoch}.pth")
            print(f"Saved model at epoch {epoch}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-train the Generator for Super-Resolution")
    parser.add_argument('--image_dir', type=str, required=True, help="Path to the training images directory.")
    parser.add_argument('--crop_size', type=int, default=24, help="Size of the low-resolution crop.")
    parser.add_argument('--upscale_factor', type=int, default=4, help="Upscale factor for super-resolution.")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--n_res_blocks', type=int, default=16, help="Number of residual blocks in the generator.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of workers for the data loader.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to a checkpoint to resume training.")
    parser.add_argument('--save_interval', type=int, default=5, help="Save the model every N epochs.")
    parser.add_argument('--test_image_path', type=str, default=None, help="Path to a low-resolution test image to evaluate at each epoch.")
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save the upscaled test images.")

    args = parser.parse_args()
    main(args)
