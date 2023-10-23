import glob

import torch
from loguru import logger

from OrthoHash import configs
from OrthoHash.main import get_config, build_loader_local
from _utils import prediction, run_evals


def calc_code_with_label(main_dir, proj_name, model_name, dataset, hash_bit):
    config = get_config()
    config["dataset"] = dataset
    config['multiclass'] = dataset != 'cifar'
    config["arch"] = model_name
    config["arch_kwargs"]["nbit"] = hash_bit
    config["arch_kwargs"]["pretrained"] = False
    config['dataset_kwargs']['resize'] = 256 if dataset != 'cifar' else 224

    # TODO: only create loader once while different hash_bit but same dataset
    _, test_loader, database_loader = build_loader_local(config)

    pkl_dir = f"{main_dir}/{proj_name}/output/{model_name}/{dataset}/{hash_bit}"
    pkl_list = glob.glob(f"{pkl_dir}/*.pkl")

    if len(pkl_list) != 1:
        logger.error(pkl_list)
        raise Exception(f'cannot find *.pkl in {pkl_dir}')

    checkpoint = torch.load(pkl_list[0], map_location="cpu")

    # load codebook
    codebook = torch.load(f"{pkl_dir}/codebook.pth")

    model = configs.arch(config, codebook=codebook)
    msg = model.load_state_dict(checkpoint, strict=False)
    logger.info(msg)

    model.cuda()

    qB, qL = prediction(model, test_loader, 1)
    rB, rL = prediction(model, database_loader, 1)

    return qB, qL, rB, rL


if __name__ == "__main__":
    torch.cuda.device(0).__enter__()
    torch.backends.cudnn.benchmark = True

    proj_name = "OrthoHash"
    model_name = "alexnet"

    # batch_size is too small will cause "Too many open files..."
    torch.multiprocessing.set_sharing_strategy('file_system')

    evals = ["mAP", "NDCG", "PR-curve", "TopN-precision", "P@H≤2"]

    datasets = ["cifar", "nuswide", "flickr", "coco"]

    hash_bits = [16, 32, 48, 64, 128]

    run_evals("/home/sxz/Projects/HASH-ZOO", proj_name, model_name, evals, datasets, hash_bits, calc_code_with_label)
