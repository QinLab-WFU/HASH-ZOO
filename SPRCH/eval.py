import glob

import torch
import torchvision
from loguru import logger

from SPRCH.main import get_config, build_loader_local
from SPRCH.utils.evaluate import choose_gpu
from _utils import prediction, run_evals


def calc_code_with_label(main_dir, proj_name, model_name, dataset, hash_bit):
    args = get_config()
    args.data_name = dataset
    args.binary_bits = hash_bit

    # TODO: only create loader once while different hash_bit but same dataset
    train_loader, test_loader, database_loader = build_loader_local(args)

    pkl_dir = f"{main_dir}/{proj_name}/output/{model_name}/{dataset}/{hash_bit}"
    pkl_list = glob.glob(f"{pkl_dir}/*.pkl")

    if len(pkl_list) != 1:
        logger.error(pkl_list)
        raise Exception(f'cannot find *.pkl in {pkl_dir}')

    checkpoint = torch.load(pkl_list[0], map_location="cpu")

    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
    else:
        raise NotImplementedError(f"unknown model: {model_name}")

    model.fc = torch.nn.Linear(512, args.binary_bits)
    msg = model.load_state_dict(checkpoint, strict=False)
    logger.info(msg)

    model.cuda()

    qB, qL = prediction(model, test_loader)
    rB, rL = prediction(model, database_loader)

    return qB, qL, rB, rL


if __name__ == "__main__":
    choose_gpu(1)

    proj_name = "SPRCH"
    model_name = "resnet18"

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    evals = ["mAP", "NDCG", "PR-curve", "TopN-precision", "P@H≤2"]

    datasets = ["cifar", "nuswide", "flickr", "coco"]

    hash_bits = [16, 32, 48, 64, 128]

    run_evals("/home/sxz/Projects/HASH-ZOO", proj_name, model_name, evals, datasets, hash_bits, calc_code_with_label)
