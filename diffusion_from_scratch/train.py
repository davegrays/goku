import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_from_scratch.unet import UncondUNet


class DiffusionModel:
    def __init__(self, t_steps=1000, epochs=10, dataset_path=None, model=UncondUNet()):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # from original ddpm paper
        # betas and alphas deal with noise schedule
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.t_steps = t_steps
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.t_steps).to(
            self.device
        )
        self.alpha_hats = torch.cumprod(1 - self.betas, dim=0)
        # model
        self.epochs = epochs
        self.batch_size = 16
        self.model = model.to(self.device)
        self.loss_function = torch.nn.MSELoss()
        # data
        self.dataset_path = dataset_path
        self.image_size = 64

    def sample_time(self, batch_size):
        return torch.randint(self.t_steps, size=(batch_size,)).to(self.device)

    def add_noise_one_step(self, x, t):
        # from original ddpm paper (Algorithm 1)
        rescaled_x = x * torch.sqrt(self.alpha_hats[t][:, None, None, None])
        noise = torch.randn_like(x).to(self.device)
        scaled_noise = noise * torch.sqrt(1 - self.alpha_hats[t][:, None, None, None])
        return rescaled_x + scaled_noise, noise

    def denoise_one_step(self, x, timestep):
        t = (torch.ones(x.shape[0]) * timestep).long().to(self.device)
        beta = self.betas[t][:, None, None, None]
        alpha_hat = self.alpha_hats[t][:, None, None, None]

        # from original ddpm paper (Algorithm 2)
        noise_pred = self.model(x, t)
        c1 = 1 / torch.sqrt(1 - beta)
        c2 = beta / torch.sqrt(1 - alpha_hat)
        noise_scale = torch.sqrt(beta) if timestep > 1 else 0
        noise_add = torch.randn_like(x).to(self.device) * noise_scale

        return c1 * (x - c2 * noise_pred) + noise_add

    def denoise_fully(self, x, steps=1000):
        # from original ddpm paper (Algorithm 2)
        self.model.eval()
        with torch.no_grad():
            steps = min(steps, self.t_steps)
            for i in tqdm(reversed(range(1, self.t_steps, self.t_steps // steps))):
                x = self.denoise_one_step(x, i)
        self.model.train()
        return x

    def train_diffusion_model(self):
        train_images = self.get_data()
        val_images = []

        # instantiate model, loss_function, and optimizer
        optimizer = torch.optim.Adam(self.model.parameters())

        # basic training loop:
        # for each image, add noise and run forward pass on noisy image
        # backward pass on noise_pred vs noise
        for e in range(self.epochs):
            train_cumloss, val_cumloss = 0, 0
            self.model.train()

            for i, (x, _) in enumerate(train_images):
                x = x.to(self.device)
                optimizer.zero_grad()
                # next 3 lines equate to original ddpm paper Algorithm 1 steps 3-5
                t = self.sample_time(x.shape[0])
                x_noisy, noise = self.add_noise_one_step(x, t)
                noise_pred = self.model(x_noisy, t)
                loss = self.loss_function(noise, noise_pred)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    train_cumloss += loss.item()

            print("epoch", e, "iters", i)
            print("train loss:", train_cumloss)
            with torch.no_grad():
                self.model.eval()
                for x in val_images:
                    x = x.to(self.device)
                    t = self.sample_time(x.shape[0])
                    x_noisy, noise = self.add_noise_one_step(x, t)
                    loss = self.loss_function(noise, self.model(x_noisy, t))
                    val_cumloss += loss.item()
            print("val loss:", train_cumloss)

        return self.model

    def get_image_transforms(self):
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.image_size + self.image_size // 4),
                torchvision.transforms.RandomResizedCrop(
                    self.image_size, scale=(0.8, 1.0)
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def get_data(self):
        dataset = torchvision.datasets.ImageFolder(
            self.dataset_path, transform=self.get_image_transforms()
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    @staticmethod
    def get_reverse_transform():
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda x: (x + 1) / 2),
                torchvision.transforms.Lambda(
                    lambda x: x.permute(1, 2, 0)
                ),  # CHW to HWC
                torchvision.transforms.Lambda(lambda x: x * 255.0),
                torchvision.transforms.Lambda(
                    lambda x: x.cpu().numpy().astype(np.uint8)
                ),
                torchvision.transforms.ToPILImage(),
            ]
        )
