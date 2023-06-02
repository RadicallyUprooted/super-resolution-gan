import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from torchvision.transforms import ToTensor, ToPILImage, Resize, InterpolationMode
from PIL import Image
from data_loader import DatasetFromFolder
from model import Generator, Discriminator, VGGLoss
import numpy as np
from tqdm import tqdm

checkpoint = "models\generator_17.pth"
MODEL = f"models\SRGAN_4x_60k.pth"

epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_set = DatasetFromFolder("danbooru")
train_loader = DataLoader(dataset=data_set, num_workers=0, batch_size=16, shuffle=True)

#generator = Generator().to(device)
#generator.load_state_dict(torch.load(checkpoint))
generator = torch.load(MODEL)['generator'].to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=1e-4)

#discriminator = Discriminator().to(device)
discriminator = torch.load(MODEL)['discriminator'].to(device)
optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4)

#content_loss_criterion = nn.MSELoss().to(device)
adversarial_loss_criterion = nn.BCEWithLogitsLoss().to(device)
vgg_loss = VGGLoss().to(device)

for epoch in range(1, epochs + 1):

    progress_bar = tqdm(enumerate(train_loader), total = len(train_loader))
    generator.train()
    discriminator.train()
    g_loss_avg = 0.
    d_loss_avg = 0.

    for i, batch in progress_bar:

        lr, hr = batch[0].to(device), batch[1].to(device)
        with amp.autocast():
            
            sr = generator(lr)
            sr = (sr + 1.) / 2.
            
            sr_disc = discriminator(sr)
                
            content_loss = 0.006 * vgg_loss(sr, hr)
            adversarial_loss = adversarial_loss_criterion(sr_disc, torch.ones_like(sr_disc))
            perceptual_loss = content_loss + 1e-3 * adversarial_loss

        optimizer_g.zero_grad()
        perceptual_loss.backward()
        optimizer_g.step()

        g_loss_avg += perceptual_loss.item()

        with amp.autocast():

            hr_disc = discriminator(hr)
            sr_disc = discriminator(sr.detach())

            fake_loss = adversarial_loss_criterion(sr_disc, torch.zeros_like(sr_disc))
            true_loss = adversarial_loss_criterion(hr_disc, torch.ones_like(hr_disc))
            adversarial_loss = fake_loss + true_loss

        d_loss_avg += adversarial_loss.item()
        
        optimizer_d.zero_grad()
        adversarial_loss.backward()
        optimizer_d.step()
        progress_bar.set_description(f"[{epoch}/{epochs}][{i + 1}/{len(train_loader)}] "
                                     f"Generator Loss: {perceptual_loss.item():.4f} Discriminator Loss: {adversarial_loss.item():.4f}")
    generator.eval()
    with torch.no_grad():

        img = Image.open("baboon.png")
        img = img.convert('RGB')
        img = Resize(512 // 4, InterpolationMode.BICUBIC)(img)
        img = ToTensor()(img)
        img = img.unsqueeze(0).to(device)
        output = generator(img)
        output = output.squeeze(0).detach().cpu()
        output = (output + 1.) / 2.
        output = ToPILImage()(output)
        output.save(f"epochs_srgan/epoch_{epoch}_res.png")

    print(f"Epoch {epoch}. Generator Loss: {g_loss_avg / len(train_loader):.4f}. Discriminator Loss: {d_loss_avg / len(train_loader):.4f}.")
    
    if epoch % 1 == 0:
        torch.save({'epoch': epoch,
                    'generator': generator,
                    'discriminator': discriminator,
                    'optimizer_g': optimizer_g,
                    'optimizer_d': optimizer_d},
                    f'models/checkpoint_{epoch}_srgan.pth')          



