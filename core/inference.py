
"""
Script for testing CGT models
"""
import torch
import mlflow
import os
import uuid
import yaml
from tqdm import tqdm
import mlflow.pytorch
import numpy as np
import pandas as pd
import shutil
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report

from histocartography.ml import CellGraphModel, TissueGraphModel, HACTModel
from grapher import Grapher

from dataloader import make_data_loader

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NODE_DIM = 514

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--cg_path',
        type=str,
        help='path to the cell graphs.',
        default=None,
        required=False
    )
    
    parser.add_argument(
        '--tg_path',
        type=str,
        help='path to tissue graphs.',
        default='../../dataset/hact-net-data/tissue_graphs/',
        required=False
    )
    
    parser.add_argument(
        '--assign_mat_path',
        type=str,
        help='path to the assignment matrices.',
        default=None,
        required=False
    )
    parser.add_argument(
        '-conf',
        '--config_fpath',
        type=str,
        help='path to the config file.',
        default='./config/bracs_grapher_7_classes_pna.yml',
        required=False
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='path to where the model is saved.',
        default=None,
        required=False
    )
    parser.add_argument(
        '--in_ram',
        help='if the data should be stored in RAM.',
        action='store_true',
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        help='batch size.',
        default=4,
        required=False
    )

    parser.add_argument(
        '-l',
        '--learning_rate',
        type=float,
        help='learning rate.',
        default=1e-4,
        required=False
    )
    parser.add_argument(
        '--out_path',
        type=str,
        help='path to where the output data are saved (currently only for the interpretability).',
        default='../../data/graphs',
        required=False
    )
    
    parser.add_argument(
        '--pretrained',
        type=str,
        help='path to saved model',
        default="./logs/model_best_val.pt",
        required=False
    )
    

    parser = Grapher.add_model_specific_args(parser)
    return parser.parse_args()


def main(args):
    """
    Test HACTNet, CG-GNN or TG-GNN.
    Args:
        args (Namespace): parsed arguments.
    """

    assert not(args.pretrained and args.model_path is not None), "Provide a model path or set pretrained. Not both."
    assert (args.pretrained or args.model_path is not None), "Provide either a model path or set pretrained."

    
    with open(args.config_fpath, 'r') as f:
        config = yaml.safe_load(f)

    
    dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'test') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'test') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'test') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
        shuffle=False
    )

    
    if 'bracs_grapher' in args.config_fpath:
        model = Grapher(
            gnn_params=config['gnn_params'],
            classification_params=config['classification_params'],
            node_dim=NODE_DIM,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name='BRACS',
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=20,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
            num_classes=7
        ).to(DEVICE)
    else:
        raise ValueError('Model type not recognized. Options are: TG, CG or HACT.')

    
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    model.eval()

    
    all_test_logits = []
    all_test_labels = []
    for batch in tqdm(dataloader, desc='Testing', unit='batch'):
        labels = batch[-1]
        data = batch[:-1]
        with torch.no_grad():
            logits = model(*data)
        all_test_logits.append(logits)
        all_test_labels.append(labels)

    all_test_logits = torch.cat(all_test_logits).cpu()
    all_test_preds = torch.argmax(all_test_logits, dim=1)
    all_test_labels = torch.cat(all_test_labels).cpu()

    all_test_preds = all_test_preds.detach().numpy()
    all_test_labels = all_test_labels.detach().numpy()

    accuracy = accuracy_score(all_test_labels, all_test_preds)
    weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')
    report = classification_report(all_test_labels, all_test_preds, digits=2)

    print('Test weighted F1 score {}'.format(weighted_f1_score))
    print('Test accuracy {}'.format(accuracy))
    print('Test classification report {}'.format(report))


if __name__ == "__main__":
    main(args=parse_arguments())
