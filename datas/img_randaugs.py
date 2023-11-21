from timm.data.auto_augment import rand_augment_transform
from PIL import Image
import os
from tqdm import tqdm

img_folder = ""
save_folder = ""
num_augmented_img = 3

aug_config = {
    "config1": "rand-m9-n4-mstd0.5-mmax9-inc1",
    "config2": "rand-m10-n3-mstd0.6-mmax9-inc0",
    "config3": "rand-m10-n3-mstd0.5-mmax10-inc0",
    "config4": "rand-m10-n4-mstd0.4-mmax10-inc1",
}

'''
格式 - rand-m9-n3-mstd0.5 
m - 随机增强的幅度
n - 每张图像进行的随机变换数，默认为 2.
mstd - 标准偏差的噪声幅度
mmax - 设置幅度的上界，默认 10
w - 加权索引的概率（index of a set of weights to influence choice of operation）
inc - 采用随幅度增加的数据增强，默认为 0
'''

def get_transforms(aug_config):
    tfm1 = rand_augment_transform(
        config_str=aug_config["config1"],
        hparams={'translate_const': 100}
    )

    tfm2 = rand_augment_transform(
        config_str=aug_config["config2"],
        hparams={'translate_const': 100}
    )

    tfm3 = rand_augment_transform(
        config_str=aug_config["config3"],
        hparams={'translate_const': 100}
    )

    tfm4 = rand_augment_transform(
        config_str=aug_config["config4"],
        hparams={'translate_const': 100}
    )

    return tfm1, tfm2, tfm3, tfm4


def augment_image(image):

    tfm1, tfm2, tfm3, tfm4 = get_transforms(aug_config)

    aug_imgs = {
        "img1" : tfm1(image),
        "img2" : tfm2(image),
        "img3" : tfm3(image),
        "img4" : tfm4(image)
    }

    return aug_imgs


def augmented_and_save(num_augmented_img, img_folder, save_folder):
    image_files = [f for f in os.listdir(img_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for image_file in tqdm(image_files):
        img_path = os.path.join(img_folder, image_file)
        img = Image.open(img_path)
        for i in range(num_augmented_img):
            aug_imgs = augment_image(img)
            img1 = aug_imgs["img1"]
            img2 = aug_imgs["img2"]
            img3 = aug_imgs["img3"]
            img4 = aug_imgs["img4"]

            output_filename1 = os.path.splitext(image_file)[0] + f'_trans1_{i + 1}.png'
            output_filename2 = os.path.splitext(image_file)[0] + f'_trans2_{i + 1}.png'
            output_filename3 = os.path.splitext(image_file)[0] + f'_trans3_{i + 1}.png'
            output_filename4 = os.path.splitext(image_file)[0] + f'_trans4_{i + 1}.png'

            out_path1 = os.path.join(save_folder, output_filename1)
            img1.save(out_path1)

            out_path2 = os.path.join(save_folder, output_filename2)
            img2.save(out_path2)

            out_path3 = os.path.join(save_folder, output_filename3)
            img3.save(out_path3)

            out_path4 = os.path.join(save_folder, output_filename4)
            img4.save(out_path4)

    print(f"finish and save in {save_folder}")


if __name__ == "__main__":

    augmented_and_save(num_augmented_img, img_folder, save_folder)





