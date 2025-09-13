import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from dlquantification.transnet import TransNet
from dlquantification.histnet import HistNet
from dlquantification.gmnet import GMNet
from dlquantification.deepsets import DeepSets
from dlquantification.utils.utils import UnlabeledBagGenerator, UnlabeledMixerBagGenerator, APPBagGenerator
from dlquantification.utils.lequabaggenerator import LeQuaBagGenerator
from dlquantification.featureextraction.fullyconnected import FCFeatureExtractionModule
from dlquantification.featureextraction.nofe import NoFeatureExtractionModule
import os
from dlquantification.utils.lossfunc import MRAE
from dlquantification.utils.lossfunc import NMD
import json
import argparse
from tqdm import tqdm
from dlquantification.featureextraction.timeseriescnn import TimeSeriesCNN
from dlquantification.featureextraction.lstm import LSTMFeatureExtractionModule
from dlquantification.utils.eeg_dataset import EEGDataset
from dlquantification.utils.uci_har_dataset import UCIHARDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_lequa(data_path, parameters_path, train_name, network, network_parameters, dataset, standarize,
                feature_extraction="rff", bag_generator="APPBagGenerator", cuda_device="cuda:0"):
    with open(os.path.join(parameters_path, f"common_parameters_{dataset}.json")) as f:
        common_params = json.load(f)
    with open(network_parameters) as f:
        network_params = json.load(f)

    if dataset == "EEG":
        n_classes = 8
        sample_size = common_params["bag_size"]
        n_channels = common_params.get("input_channels", 8)
        loss = MRAE(eps=1.0 / (2 * sample_size), n_classes=n_classes)

        path = "./dataset/EMG_Data"
        ds_full = EEGDataset(data_directory=path, sequence_length=sample_size)

        n_total = len(ds_full)
        n_train = int(0.8 * n_total)

        X_all, Y_all = zip(*[(x.to(cuda_device), y.to(cuda_device)) for x, y in ds_full])
        X_tr, Y_tr = torch.stack(X_all[:n_train]), torch.stack(Y_all[:n_train])
        X_v, Y_v = torch.stack(X_all[n_train:]), torch.stack(Y_all[n_train:])

    elif dataset == "SMARTFALL":
        n_classes = 2
        sample_size = common_params["bag_size"]
        n_channels = 3
        loss = MRAE(eps=1.0 / (2 * sample_size), n_classes=n_classes)

        from dlquantification.utils.smartfall_dataset import SmartFallDataset
        ds_full = SmartFallDataset(data_directory="./dataset/smartfallMM", sequence_length=sample_size)
        n_total = len(ds_full)
        n_train = int(0.8 * n_total)

        X_all, Y_all = zip(*[(x.to(cuda_device), y.to(cuda_device)) for x, y in ds_full])
        X_tr, Y_tr = torch.stack(X_all[:n_train]), torch.stack(Y_all[:n_train])
        X_v, Y_v = torch.stack(X_all[n_train:]), torch.stack(Y_all[n_train:])

    elif dataset == "UCIHAR":
        n_classes = 6
        sample_size = common_params["bag_size"]
        n_channels = 9  # 3-axes * 3 sensors

        loss = MRAE(eps=1.0 / (2 * sample_size), n_classes=n_classes)

        ds_train = UCIHARDataset(base_dir="./dataset/UCI_HAR", split="train", sequence_length=sample_size)
        ds_test = UCIHARDataset(base_dir="./dataset/UCI_HAR", split="test", sequence_length=sample_size)

        X_tr, Y_tr = zip(*[(x.to(cuda_device), y.to(cuda_device)) for x, y in ds_train])
        X_v, Y_v = zip(*[(x.to(cuda_device), y.to(cuda_device)) for x, y in ds_test])

        X_tr, Y_tr = torch.stack(X_tr), torch.stack(Y_tr)
        X_v, Y_v = torch.stack(X_v), torch.stack(Y_v)


    else:
        raise ValueError("Dataset not recognized")

    fe = LSTMFeatureExtractionModule(
        input_dim=n_channels,
        hidden_size=64,
        output_size=8,
        num_layers=2,
        dropout_lstm=0.3,
        dropout_linear=0.3
    )

    train_baggen = APPBagGenerator(device=cuda_device)
    val_baggen = APPBagGenerator(device=cuda_device)

    parameters = {**common_params, **network_params}
    parameters.update({
        "n_classes": n_classes,
        "feature_extraction_module": fe,
        "bag_generator": train_baggen,
        "val_bag_generator": val_baggen,
        "quant_loss": loss,
        "dataset_name": dataset,
        "device": cuda_device,
    })

    parameters["save_model_path"] = os.path.join("savedmodels", f"{train_name}.pth")

    model = GMNet(**parameters)
    model.fit(
        dataset=TensorDataset(X_tr, Y_tr),
        val_dataset=TensorDataset(X_v, Y_v),
    )

    last_model_path = parameters["save_model_path"]
    model.model.load_state_dict(torch.load(last_model_path))

    return model

def test_lequa(model, data_path, train_name, dataset, standarize):
    print("Testing the model...")

    if dataset == "EEG":
        path = "./dataset/EMG_Data"
    elif dataset == "SMARTFALL":
        path = "./dataset/smartfallMM"
    elif dataset == "UCIHAR":
        path = "./dataset/UCI HAR Dataset"
    else:
        raise ValueError("Unsupported dataset")

    with open(os.path.join("experiments/parameters", f"common_parameters_{dataset}.json")) as f:
        common_params = json.load(f)

    sample_size = common_params["bag_size"]
    bag_size = common_params["bag_size"]

    if dataset == "EEG":
        ds_full = EEGDataset(data_directory=path, sequence_length=sample_size)
        n_classes = 8
    elif dataset == "SMARTFALL":
        from dlquantification.utils.smartfall_dataset import SmartFallDataset
        ds_full = SmartFallDataset(data_directory=path, sequence_length=sample_size)
        n_classes = 2
    elif dataset == "UCIHAR":
        ds_full = UCIHARDataset(base_dir=path, split="test", sequence_length=sample_size)
        n_classes = 6
    else:
        raise ValueError("Unsupported dataset")

    if dataset == "UCIHAR":
        X, Y = zip(*[ds_full[i] for i in range(len(ds_full))])
    else:
        n_total = len(ds_full)
        n_train = int(0.8 * n_total)
        X, Y = zip(*[ds_full[i] for i in range(n_train, n_total)])

    X = torch.stack(X)
    Y = torch.stack(Y)

    n_bags = X.shape[0] // bag_size
    X = X[:n_bags * bag_size]
    Y = Y[:n_bags * bag_size]
    X = X.view(n_bags, bag_size, sample_size, -1)
    Y = Y.view(n_bags, bag_size)

    test_dataset = TensorDataset(X)
    print(f"Loaded {len(test_dataset)} {dataset} test bags.")

    print(f"Evaluating {dataset} test data...")
    preds = model.predict(test_dataset, process_in_batches=13)

    results = pd.DataFrame(preds)
    os.makedirs("results/", exist_ok=True)
    results.to_csv(os.path.join("results/", f"{train_name}.txt"), index_label="id")
    print(f"Saved {dataset} predictions.")

    p_true = []
    for i in range(n_bags):
        y_bag = Y[i]
        prevalences = torch.bincount(y_bag, minlength=n_classes).float() / len(y_bag)
        p_true.append(prevalences)
    p_true = torch.stack(p_true)

    eps = 1.0 / (2 * sample_size)
    mrae_fn = MRAE(eps=eps, n_classes=n_classes)
    preds_tensor = torch.tensor(preds)
    mrae = mrae_fn(p_true, preds_tensor).item()

    print(f"[Test] MRAE: {mrae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for EEG, SMARTFALL, or UCIHAR")
    parser.add_argument("-t", "--train_name", required=True)
    parser.add_argument("-n", "--network", required=True)
    parser.add_argument("-p", "--network_parameters", required=True)
    parser.add_argument("-f", "--feature_extraction", default="timeseriescnn")
    parser.add_argument("-b", "--bag_generator", default="APPBagGenerator")
    parser.add_argument("-s", "--standarize", action='store_true')
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-c", "--cuda_device", required=True)
    args = vars(parser.parse_args())

    print("Using following arguments:")
    print(args)

    args["cuda_device"] = torch.device(args["cuda_device"])

    data_path = "data/"
    parameters_path = "experiments/parameters"

    model = train_lequa(data_path, parameters_path, **args)
    test_lequa(model, data_path, args["train_name"], args["dataset"], args["standarize"])
