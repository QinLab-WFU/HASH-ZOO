import glob
import os

import torch
from loguru import logger

from DPSH import CreateModel
from DPSH_demo import get_config, build_loader_local
from _utils import prediction, run_evals


def calc_code_with_label(main_dir, proj_name, model_name, dataset, hash_bit):
    param = get_config()
    param["dataset"] = dataset
    # param["model"] = model_name

    # TODO: only create loader once while different hash_bit but same dataset
    train_loader, test_loader, database_loader = build_loader_local(param)

    pkl_dir = f"{main_dir}/{proj_name}/output/{model_name}/{dataset}/{hash_bit}"
    pkl_list = glob.glob(f"{pkl_dir}/*.pkl")

    if len(pkl_list) != 1:
        logger.error(pkl_list)
        raise Exception(f'cannot find *.pkl in {pkl_dir}')

    checkpoint = torch.load(pkl_list[0], map_location="cpu")

    model = CreateModel(model_name, hash_bit, torch.cuda.is_available())
    msg = model.load_state_dict(checkpoint, strict=False)
    logger.info(msg)

    qB, qL = prediction(model, test_loader)
    rB, rL = prediction(model, database_loader)

    return qB, qL, rB, rL


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    proj_name = "DPSH"
    model_name = "alexnet"

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    evals = ["mAP", "NDCG", "PR-curve", "TopN-precision", "P@H≤2"]

    datasets = ["cifar", "nuswide", "flickr", "coco"]

    hash_bits = [16, 32, 48, 64, 128]

    run_evals("/home/sxz/Projects/HASH-ZOO", proj_name, model_name, evals, datasets, hash_bits, calc_code_with_label)
