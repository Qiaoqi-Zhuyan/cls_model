import logging
import os

import os

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from timm.utils import accuracy

from trainer import config, model
from datas.build import CustomDataset_Origin, build_transform

# logger
logger_name = f"{config.model_name}.log"
logger_save_path = "labs/logger"
logger_save = os.path.join(logger_save_path, logger_name)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(logger_save)
    ]
)
logging.basicConfig(stream=None)
logger = logging.getLogger("test_logger")


def build_test_dataloader(config):

    transform = build_transform(config)

    test_dataset = CustomDataset_Origin(
        csv_path=config.csv_path,
        imgs_path=config.img_folder,
        img_size=config.img_size,
        transforms=transform
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return test_dataloader


def val_model(model, test_dataloader, config):
    model.eval(True)
    with torch.no_grad():
        with tqdm(test_dataloader, desc="Test") as t:
            for x, y in t:
                x = x.to('cuda', non_blocking=True)
                y = y.to('cuda', non_blocking=True)
                y_p = model(x)

                pred = torch.argmax(y_p, dim=1)

                test_timm_acc = '{:.4f}'.format(accuracy(pred, y))

                y = y.to('cpu')
                pred = pred.to('cpu')
                test_balanced_accuracy = '{:.4f}'.format(balanced_accuracy_score(y, pred))

                t.set_postfix(test_acc=test_timm_acc, test_BA=test_balanced_accuracy)

                logger.info(f'test:'
                            f'test Accuracy: {test_timm_acc},'
                            f'test Balanced Accuracy: {test_balanced_accuracy}')

if __name__ == "__main__":
    test_dataloader = build_test_dataloader(config)
    model = model.load_state_dict()
    val_model(model, test_dataloader, config)
