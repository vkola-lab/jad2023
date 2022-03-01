import torch.nn as nn

from image_encoder import ImageEncoder_64
from latent_decoder import LatentDecoder_64
from latent_discriminator import LatentDiscriminator


class ImageAdversarialAutoencoder_64(nn.Module):
    

    def __init__(self, n_fil=32, dim_z=512) -> None:

        super().__init__()
        self.encoder = ImageEncoder_64(n_fil, dim_z)
        self.decoder = LatentDecoder_64(n_fil, dim_z, (2, 2, 2))
        # self.decoder = LatentDecoder_64(n_fil, dim_z, (1, 1, 1))
        self.discriminator = LatentDiscriminator(dim_z)

    def forward(self, x, phase):
        pass
