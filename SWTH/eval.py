import glob

import torch
from loguru import logger
from torch.backends import cudnn

from SWTH.main import parse_option, build_loader_local
from SWTH.models import build_model
from _utils import prediction, run_evals


def calc_code_with_label(main_dir, proj_name, model_name, dataset, hash_bit):
    _, config = parse_option()

    config.defrost()
    config.DATA.DATASET = dataset
    config.MODEL.hash_length = hash_bit
    config.MODEL.TYPE = model_name
    config.freeze()

    # TODO: only create loader once while different hash_bit but same dataset
    train_loader, test_loader, database_loader, _ = build_loader_local(config)

    pkl_dir = f"{main_dir}/{proj_name}/output/{model_name}/{dataset}/{hash_bit}"
    pkl_list = glob.glob(f"{pkl_dir}/*.pkl")

    if len(pkl_list) != 1:
        logger.error(pkl_list)
        raise Exception(f'cannot find *.pkl in {pkl_dir}')

    checkpoint = torch.load(pkl_list[0], map_location="cpu")

    model = build_model(config)

    msg = model.load_state_dict(checkpoint, strict=False)
    logger.info(msg)

    model.cuda()

    qB, qL = prediction(model, test_loader, 0)
    rB, rL = prediction(model, database_loader, 0)

    return qB, qL, rB, rL


if __name__ == "__main__":
    torch.cuda.device(1).__enter__()
    cudnn.benchmark = True

    proj_name = "SWTH"
    model_name = "swin"

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    evals = ["mAP", "NDCG", "PR-curve", "TopN-precision", "P@H≤2"]

    datasets = ["cifar", "nuswide", "flickr", "coco"]

    hash_bits = [16, 32, 48, 64, 128]

    run_evals("/home/sxz/Projects/HASH-ZOO", proj_name, model_name, evals, datasets, hash_bits, calc_code_with_label)
