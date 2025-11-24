#!/usr/bin/env python
from getpass import getpass
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch import seed_everything

from dataclasses import dataclass, asdict, field
from clearml import Task

torch.set_float32_matmul_precision('medium')

@dataclass
class CFG:
    seed: int = 42
    batch_size: int = 64
    lr: float = 0.0002
    num_epochs: int = 10
    noise_dim: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gen_dict: dict = field(default_factory=lambda: {
        'kernel_size': 4,
        'stride': 2,
        'padding': 1,
    })
    disc_dict: dict = field(default_factory=lambda: {
        'kernel_size': 4,
        'stride': 2,
        'padding': 1,
    })
    
class MNISTDataModule(LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        datasets.MNIST(root='./data', train=True, download=True)
        datasets.MNIST(root='./data', train=False, download=True)

    def setup(self, stage=None):
        self.mnist_train = datasets.MNIST(root='./data', train=True, transform=self.transform)
        self.mnist_test = datasets.MNIST(root='./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=20)


class Generator(nn.Module):
    def __init__(self, noise_dim=100, *args, **kwargs):
        super(Generator, self).__init__()
        kernel_size = kwargs.get('kernel_size', 4)
        stride = kwargs.get('stride', 2)
        padding = kwargs.get('padding', 1)
        self.main = nn.Sequential(
            # Вход: вектор шума размера noise_dim
            nn.Linear(noise_dim, 256 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 7, 7)),
            # Состояние: (256, 7, 7)
            nn.ConvTranspose2d(
                256, 128, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            ),  # -> (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            ),  # -> (1, 28, 28)
            nn.Tanh(),
        )
        
    def forward(self, x):
        return self.main(x)
        
        
class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__()
        kernel_size = kwargs.get('kernel_size', 4)
        stride = kwargs.get('stride', 2)
        padding = kwargs.get('padding', 1)
        self.main = nn.Sequential(
            # Вход: изображение (1, 28, 28)
            nn.Conv2d(
                1, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            ),  # -> (64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                64, 128, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            ),  # -> (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return self.main(x)
    
class GAN_MNIST_Model(LightningModule):
    def __init__(self, 
                 noise_dim=100, 
                 lr=0.01, 
                 gen_dict=None, 
                 disc_dict=None,
                 debug_epoch=1,
                 task_clearml = None,
                 ):
        super().__init__()
        # import pdb; pdb.set_trace()
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(noise_dim=noise_dim, **(gen_dict or {})).to(self.dev)
        self.discriminator = Discriminator( **(disc_dict or {})).to(self.dev)
        self.criterion = nn.BCELoss()
        
        self.noise_dim = noise_dim
        self.debug_epoch = debug_epoch
        self.task = task_clearml
        self.lr = lr
        self.real_label = 1.0
        self.fake_label = 0.0
        self.automatic_optimization = False  # Отключаем автоматическое управление оптимизацией

    def forward(self, input):
        return self.generator(input)

    def training_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        opt_g, opt_d = self.optimizers()
        real_images, _ = batch
        real_images = real_images.to(self.dev)
        batch_size = real_images.size(0)
        noise = torch.randn(batch_size, self.noise_dim, device=self.dev)
        label = torch.full((batch_size,), self.real_label, device=self.dev)

        # Обучение генератора
        opt_g.zero_grad()
        fake_images = self(noise)
        output = self.discriminator(fake_images).view(-1)
        errG = self.criterion(output, label)
        D_G_z2 = output.mean().item()
        #errG.backward()
        self.manual_backward(errG)
        opt_g.step()
        #optimizer_g.step()
        #return errG

        # Обучение дискриминатора
        opt_d.zero_grad()
        output = self.discriminator(real_images).view(-1)
        errD_real = self.criterion(output, label)
        self.manual_backward(errD_real)
        #errD_real.backward()
        
        D_x = output.mean().item()

        label.fill_(self.fake_label)
        fake_images = self(noise).detach() # Отключаем градиенты для фейковых изображений
        output = self.discriminator(fake_images).view(-1)
        errD_fake = self.criterion(output, label)
        #errD_fake.backward()
        self.manual_backward(errD_fake)
        opt_d.step()

        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        if batch_idx % 2 == 0:
            self.log('errD', errD.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('errG', errG.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('D_x', D_x, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('D_G_z1', D_G_z1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('D_G_z2', D_G_z2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if self.task is not None and self.current_epoch % self.debug_epoch == 0:    
            self.task.get_logger().report_scalar("errD", "value", iteration=self.global_step, value=errD.item())
            self.task.get_logger().report_scalar("errG", "value", iteration=self.global_step, value=errG.item())
        
        #return errD 
            
    def on_train_epoch_end(self):
        if self.current_epoch % self.debug_epoch == 0:
            fixed_noise = torch.randn(64, self.noise_dim, device=self.dev)
            fake_images = self(fixed_noise).detach().cpu()
            os.makedirs('output', exist_ok=True)
            torchvision.utils.save_image(fake_images, f'output/fake_images_epoch_{self.current_epoch}.png', normalize=True)
            if self.task is not None:
                self.task.get_logger().report_image(
                    "fake_images",
                    iteration=self.current_epoch,
                    series="GAN",
                    local_path=f'output/fake_images_epoch_{self.current_epoch}.png'
                )
    
    def configure_optimizers(self):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [optimizer_g, optimizer_d]

def check_clearml_env():
    os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml/'
    os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
    os.environ['CLEARML_FILES_HOST'] ='https://files.clear.ml'
    if os.getenv('CLEARML_API_ACCESS_KEY') is None:
        os.environ['CLEARML_API_ACCESS_KEY'] = getpass(prompt="Введите API Access токен: ")
    if os.getenv('CLEARML_API_SECRET_KEY') is None:
        os.environ['CLEARML_API_SECRET_KEY'] = getpass(prompt="Введите API Secret токен: ")
        
def main(epochs, debug_samples_epoch, fast_dev_run, clearml):
    cfg = CFG()
    seed_everything(cfg.seed)
    cfg.num_epochs = epochs
    
    task = None
    if clearml:
        try:
            task = Task.init(project_name="GAN_lightning_ClearMl", task_name="Pytorch lightning model")
        except:
            check_clearml_env()
            task = Task.init(project_name="GAN_lightning_ClearMl", task_name="Pytorch lightning model")    
        task.add_tags(["GAN", "lightning"])
        cfg_dict = asdict(cfg)
        task.connect(cfg_dict) # Добавление конфигурации в ClearML
    
    data = MNISTDataModule(batch_size=cfg.batch_size)
    model = GAN_MNIST_Model(lr=cfg.lr, 
                            gen_dict=cfg.gen_dict, 
                            disc_dict=cfg.disc_dict, 
                            noise_dim=cfg.noise_dim,
                            debug_epoch=debug_samples_epoch,
                            task_clearml=task,)
    
    try:
        if fast_dev_run:
            trainer = Trainer(fast_dev_run=fast_dev_run)
            trainer.fit(model, datamodule=data)
            print("!!!Тестовый прогон завершился УСПЕШНО!!!")
            return
        
        trainer = Trainer(
        accelerator='auto',
        devices=-1,
        max_epochs=epochs,
        callbacks=[
            ModelCheckpoint(
                monitor='D_G_z2',
                mode='min',
                save_top_k=1,
                save_weights_only=True,
                dirpath='models',
                filename='generator',
                enable_version_counter=True,
            )])
        trainer.fit(model, data)
        
    except Exception as exp:
        print(f"!!!EXCEPTION: {exp}")
        print("!!!Прогон завершился с ошибкой!!!")
        sys.exit(1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python Lightning script")
    parser.add_argument('--fast_dev_run', type=bool, default=False, help='Run a single batch for quick debugging')
    parser.add_argument('--clearml', type=bool, default=False, help='Use ClearML for logging')
    parser.add_argument('--epochs', type=int, default=10, help='Run number of epochs')
    parser.add_argument('--debug_samples_epoch', type=int, default=1, help='Frequency of saving debug samples')
    args = parser.parse_args()
    main(args.epochs, args.debug_samples_epoch, args.fast_dev_run, args.clearml)