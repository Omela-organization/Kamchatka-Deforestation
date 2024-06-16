from PIL import Image
import numpy as np
from torchvision.transforms import ToPILImage
import torch
import torch.nn.functional as F

def split_image_into_patches(image_array):
    h, w, _ = image_array.shape
    patches = []
    
    for i in range(0, h, 256):
        for j in range(0, w, 256):
            patch = image_array[i:i+256, j:j+256, :]
            patch = Image.fromarray(patch)
            patches.append(patch)
    
    return patches


def merge_patches_into_image(patches, image_shape):
    h, w, _ = image_shape
    image = np.zeros((h, w, _), dtype=np.uint8)
    
    patch_index = 0
    for i in range(0, h, 256):
        for j in range(0, w, 512):
            image[i:i+256, j:j+256, :] = patches[patch_index]
            patch_index += 1
    
    return image


def convert_to_mask(input_array):
    
    rgb_image = np.concatenate([input_array, input_array, input_array], axis=-1)

    rgb_image = np.where(rgb_image == 0, [252, 15, 192], rgb_image)


    rgb_image = rgb_image.astype(np.uint8)

    return rgb_image

def overlay_mask(original_image, mask):
    overlayed_image = np.copy(original_image)

    mask_indices = np.all(mask == [252, 15, 192], axis=2)

    overlayed_image[mask_indices] = mask[mask_indices]

    return overlayed_image



def generate_image(img, model):
    #Требуется загрузить изображение в формате в виде массива нампи размерности (H,W,3)
    H, W, _ = img.shape
    SIZE=10240

    img =np.array(Image.fromarray(img).resize((SIZE,SIZE)))
    p = split_image_into_patches(img)

    norm = transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                      std = [0.229, 0.224, 0.225])

    arr = []
    for i in range(len(p)):
        tensor = pil_to_tensor(p[i]).unsqueeze(0)/255.0
        tensor = norm(tensor).cuda().to(torch.float16)
        out = F.sigmoid(model(tensor))
        out = torch.where(out >= 0.35, torch.tensor(1), torch.tensor(0)).detach().cpu()
        out = np.array(ToPILImage()(out.squeeze(0).to(torch.uint8))).reshape(256,256,1)
        arr.append(out)

    mask = merge_patches_into_image(arr, (SIZE, SIZE, 1))
    mask = convert_to_mask(mask)

    img = overlay_mask(img, mask)
    return img




