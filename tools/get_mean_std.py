import os
from tqdm import tqdm
from PIL import Image
import numpy as np

img_folder = ""

def get_mean_std(img_folder):
    image_files = [f for f in os.listdir(img_folder) if f.endswith('.png') or f.endswith('.jpg')]

    mean_r = 0
    mean_g = 0
    mean_b = 0

    for image_file in tqdm(image_files):
        img_path = os.path.join(img_folder, image_file)
        img = Image.open(img_path)
        img = np.asarray(img)

        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(image_files)
    mean_g /= len(image_files)
    mean_r /= len(image_files)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0.0

    for image_file in tqdm(image_files):
        img_path = os.path.join(img_folder, image_file)
        img = Image.open(img_path)
        img = np.asarray(img)

        diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)


    mean = (mean_b.item() / 255.0, mean_g.item() / 255.0, mean_r.item() / 255.0)
    std = (std_b.item() / 255.0, std_g.item() / 255.0, std_r.item() / 255.0)
    return mean, std


if __name__ == "__main__":

    mean, std = get_mean_std()
    print(f"mean: {mean}, std: {std}")


