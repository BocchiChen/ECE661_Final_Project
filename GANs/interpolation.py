import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from DRA_GAN import *
from torchvision.utils import save_image
from PIL import Image

# Assuming you have a trained generator model
generator = Generator()
generator.load_state_dict(torch.load('./test/generator.pth'))
generator.eval()

# Sample two random latent vectors
latent_vector1 = torch.randn(1, 100)
# latent_vector1_class = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]).view(1, 10)
# latent_vector1 = torch.cat((latent_vector1, latent_vector1_class), dim=1)
latent_vector2 = torch.randn(1, 100)
# latent_vector2_class = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 10)
# latent_vector2 = torch.cat((latent_vector2, latent_vector2_class), dim=1)

print(latent_vector1.shape)
print(latent_vector2.shape)

# Set the number of interpolation steps
num_steps = 10

# Perform linear interpolation
interpolated_latents = []
for alpha in torch.linspace(0, 1, num_steps):
    interpolated_latent = alpha * latent_vector1 + (1 - alpha) * latent_vector2
    interpolated_latents.append(interpolated_latent)

# Generate images from interpolated latent vectors
generated_images = []
i = 0
for latent_vector in interpolated_latents:
    with torch.no_grad():
        generated_image = generator(latent_vector).data[:50]
    save_image(generated_image, os.path.join('./images_interpolation', "%d.png" % i), nrow=5, normalize=True)
    i += 1
    generated_images.append(generated_image)

