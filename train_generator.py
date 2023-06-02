import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from data_loader import DatasetFromFolder
from model import Generator
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Resize, InterpolationMode
from tqdm import tqdm

EPOCHS = 17
checkpoint = "models\generator_5.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data_set = DatasetFromFolder(r"danbooru")
train_loader = DataLoader(dataset=data_set, num_workers=0, batch_size=16, shuffle=True)

model = Generator().to(device)
model.load_state_dict(torch.load(checkpoint))
#model.load_state_dict(torch.load(MODEL))
#model = torch.load(MODEL).to(device)
content_loss_criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(8, EPOCHS + 1):
    progress_bar = tqdm(enumerate(train_loader), total = len(train_loader))
    epoch_loss = 0
    model.train()
    for i, batch in progress_bar:

        lr, hr = batch[0].to(device), batch[1].to(device)
        with amp.autocast():

            out = model(lr)
            loss = content_loss_criterion(out, hr)

        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        progress_bar.set_description(f"[{epoch}/{EPOCHS}][{i + 1}/{len(train_loader)}] "
                                     f"MSE loss: {loss.item():.4f}")

    print(f"Epoch {epoch}. Training loss: {epoch_loss / len(train_loader)}")   

    if epoch % 1 == 0:
        torch.save(model.state_dict(), f"models\generator_{epoch}.pth")    
    
    model.eval()
    with torch.no_grad():

        img = Image.open("baboon.png")
        img = img.convert('RGB')
        img = Resize(512 // 4, InterpolationMode.BICUBIC)(img)
        img = ToTensor()(img)
        img = img.unsqueeze(0).to(device)
        output = model(img)
        output = output.squeeze(0).cpu()
        output = (output + 1.) / 2.
        output = ToPILImage()(output)
        output.save(f"epochs/epoch_{epoch}_res.png")
