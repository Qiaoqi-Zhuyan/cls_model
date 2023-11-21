import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import albumentations as A
import cv2

num_augmented_img = 5
img_folder = ""
save_folder = ""

def get_transforms():
    transforms = {}

    transform1 = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightness(p=0.2)
    ])

    transform2 = A.Compose([
        A.MultiplicativeNoise(multiplier=1.5, p=1),
        A.RandomBrightness(p=0.3)
    ])

    transform3 = A.Compose([
        A.OpticalDistortion(),
        A.Flip(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1),
        A.RandomContrast(p=0.5)
    ])

    transform4 = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Cutout(num_holes=50, max_h_size=40, max_w_size=40, fill_value=128, p=1)
    ])

    transform5 = A.Compose([
        A.Blur(blur_limit=(15, 15), p=1),
        A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1)
    ])

    # add more ...

    transforms = {
        "transform1": transform1,
        "transform2": transform2,
        "transform3": transform3,
        "transform4": transform4,
        "transform5": transform5,
    }

    return transforms

def augment_image(image):

    transforms = get_transforms()

    img1 = transforms["transform1"](image=image)['image']
    img2 = transforms["transform2"](image=image)['image']
    img3 = transforms["transform3"](image=image)['image']
    img4 = transforms["transform4"](image=image)['image']
    img5 = transforms["transform5"](image=image)['image']

    aug_imgs = {
        "img1" : img1,
        "img2" : img2,
        "img3" : img3,
        "img4" : img4,
        "img5" : img5,
    }

    return aug_imgs

def augmented_and_save(num_augmented_img, img_folder, save_folder):
    image_files = [f for f in os.listdir(img_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for image_file in tqdm(image_files):
        img_path = os.path.join(img_folder, image_file)
        img = cv2.imread(img_path)
        np_img = np.array(img)
        for i in range(num_augmented_img):
            aug_imgs = augment_image(np_img)
            img1 = aug_imgs["img1"]
            img2 = aug_imgs["img2"]
            img3 = aug_imgs["img3"]
            img4 = aug_imgs["img4"]
            img5 = aug_imgs["img5"]

            output_filename1 = os.path.splitext(image_file)[0] + f'_trans1_{i + 1}.png'
            output_filename2 = os.path.splitext(image_file)[0] + f'_trans2_{i + 1}.png'
            output_filename3 = os.path.splitext(image_file)[0] + f'_trans3_{i + 1}.png'
            output_filename4 = os.path.splitext(image_file)[0] + f'_trans4_{i + 1}.png'
            output_filename5 = os.path.splitext(image_file)[0] + f'_trans5_{i + 1}.png'

            out_path1 = os.path.join(save_folder, output_filename1)
            cv2.imwrite(out_path1, img1)

            out_path2 = os.path.join(save_folder, output_filename2)
            cv2.imwrite(out_path2, img2)

            out_path3 = os.path.join(save_folder, output_filename3)
            cv2.imwrite(out_path3, img3)

            out_path4 = os.path.join(save_folder, output_filename4)
            cv2.imwrite(out_path4, img4)

            out_path5 = os.path.join(save_folder, output_filename5)
            cv2.imwrite(out_path5, img5)

    print(f"finish and save in {save_folder}")


if __name__ == "__main__":

    augmented_and_save(num_augmented_img=num_augmented_img, save_folder=save_folder, img_folder=img_folder)


