from PIL import Image,ImageOps
import base64
import io
import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as T


def decode_base64(base64_string,resize = 224):

    transform_size = T.Resize(resize)
    # transform_normalize = T.Normalize(mean=0, std=1)
    transform_to_tensor = T.Compose([
    T.ToTensor()
    ])

    base64_decoded = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(base64_decoded))
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = transform_size(image)
    # image_tensor = torch.tensor(np.array(image)).float()
    image_tensor = transform_to_tensor(image)
    image_tensor = image_tensor + (image_tensor>0).float()*0.5

    image_tensor = image_tensor
    # image_tensor = transform_normalize(image_tensor)
    
    return image_tensor #( 1.0 - image_tensor / 255.0 ) * 2

def plot_image_tensor(image_tensor,f):
    image_np = image_tensor.numpy()
    plt.imshow(image_np, interpolation='nearest')
    plt.show()
    plt.savefig(f'{f}')
    
