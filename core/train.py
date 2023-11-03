#!/usr/bin/env python3
"""
Script for training CGT
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
from train_utils import parse_device


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
DEVICE = parse_device()

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
        default='',
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
        '--epochs', type=int, help='epochs.', default=200, required=False
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
        default=None,
        required=False
    )
    parser.add_argument(
        '--logger',
        type=str,
        help='Logger type. Options are "mlflow" or "none"',
        required=False,
        default='none'
    )

    parser = Grapher.add_model_specific_args(parser)

    return parser.parse_args()


def main(args):
    """
    Train CGT.
    Args:
        args (Namespace): parsed arguments.
    """

    if args.config_fpath is not None:
        with open(args.config_fpath, 'r') as f:
            config = yaml.safe_load(f)

    if args.logger == 'mlflow':
        mlflow.log_params({
            'batch_size': args.batch_size
        })
        df = pd.io.json.json_normalize(config)
        rep = {"graph_building.": "", "model_params.": "", "gnn_params.": ""}  
        for i, j in rep.items():
            df.columns = df.columns.str.replace(i, j)
        flatten_config = df.to_dict(orient='records')[0]
        for key, val in flatten_config.items():
            mlflow.log_params({key: str(val)})

    
    model_path = os.path.join(args.model_path, "logs", str(uuid.uuid4()))
    os.makedirs(model_path, exist_ok=True)
    writer = SummaryWriter(model_path)

    
    train_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'train') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'train') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'train') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )
    val_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'test') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'test') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'test') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )
    test_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, 'test') if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, 'test') if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, 'test') if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
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


    '''
    Compute params and flops
    '''
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameter: {param}")

    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    '''define loss function'''
    loss_fn = torch.nn.CrossEntropyLoss()

    
    step = 0
    best_val_loss = 10e5
    best_val_accuracy = 0.
    best_val_weighted_f1_score = 0.

    
    all_train_loss = []
    all_train_accuracy = []
    all_train_weighted_f1_score = []

    all_val_loss = []
    all_val_accuracy = []
    all_val_weighted_f1_score = []

    for epoch in range(args.epochs):
        
        
        model.train()
        epoch_loss = 0
        all_train_logits = []
        all_train_labels = []
        for batch in tqdm(train_dataloader, desc='Epoch training {}'.format(epoch), unit='batch'):
            
            labels = batch[-1]
            data = batch[:-1]

            logits = model(*data)

            all_train_logits.append(logits)
            all_train_labels.append(labels)

            
            loss = loss_fn(logits, labels)
            

            epoch_loss += loss.cpu().detach().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            if args.logger == 'mlflow':
                mlflow.log_metric('train_loss', loss.item(), step=step)

            
            step += 1

        
        train_loss = epoch_loss/len(train_dataloader)
        all_train_loss.append(train_loss)
        
        all_train_logits = torch.cat(all_train_logits).cpu()
        all_train_preds = torch.argmax(all_train_logits, dim=1).detach().numpy()
        all_train_labels = torch.cat(all_train_labels).cpu().detach().numpy()

        epoch_train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        epoch_train_weighted_f1_score = f1_score(all_train_labels, all_train_preds, average='weighted')
        if args.logger == 'mlflow':
            mlflow.log_metric('train_loss', train_loss, step=step)
            mlflow.log_metric('train_accuracy', epoch_train_accuracy, step=step)
            mlflow.log_metric('train_weighted_f1_score', epoch_train_weighted_f1_score, step=step)
        all_train_accuracy.append(epoch_train_accuracy)
        all_train_weighted_f1_score.append(epoch_train_weighted_f1_score)
        print('train loss: {} | train weighted F1: {} | train accuracy: {}'.format(train_loss, epoch_train_weighted_f1_score, epoch_train_accuracy))

        '''Log of each Epoch'''
        writer.add_scalar('Training/Loss', train_loss, epoch)
        writer.add_scalar('Training/Weighted_F1_score', epoch_train_weighted_f1_score, epoch)
        writer.add_scalar('Training/Accuracy', epoch_train_accuracy, epoch)
        writer.add_scalar('Training/Lr', optimizer.param_groups[0]['lr'], epoch)

        if scheduler is not None:
            scheduler.step()

        
        model.eval()
        all_val_logits = []
        all_val_labels = []
        for batch in tqdm(val_dataloader, desc='Epoch validation {}'.format(epoch), unit='batch'):
            labels = batch[-1]
            data = batch[:-1]
            with torch.no_grad():
                logits = model(*data)
            all_val_logits.append(logits)
            all_val_labels.append(labels)

        all_val_logits = torch.cat(all_val_logits).cpu()
        all_val_preds = torch.argmax(all_val_logits, dim=1)
        all_val_labels = torch.cat(all_val_labels).cpu()

        
        with torch.no_grad():
            loss = loss_fn(all_val_logits, all_val_labels).item()
            all_val_loss.append(loss)
        if args.logger == 'mlflow':
            mlflow.log_metric('val_loss', loss, step=step)
        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_loss.pt'))

        
        all_val_preds = all_val_preds.detach().numpy()
        all_val_labels = all_val_labels.detach().numpy()
        accuracy = accuracy_score(all_val_labels, all_val_preds)
        if args.logger == 'mlflow':
            mlflow.log_metric('val_accuracy', accuracy, step=step)
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_accuracy.pt'))

        
        weighted_f1_score = f1_score(all_val_labels, all_val_preds, average='weighted')
        if args.logger == 'mlflow':
            mlflow.log_metric('val_weighted_f1_score', weighted_f1_score, step=step)
        if weighted_f1_score > best_val_weighted_f1_score:
            best_val_weighted_f1_score = weighted_f1_score
            torch.save(model.state_dict(), os.path.join(model_path, 'model_best_val_weighted_f1_score.pt'))

        all_val_weighted_f1_score.append(weighted_f1_score)
        all_val_accuracy.append(accuracy)

        print('val loss: {} | val weighted F1: {} | val accuracy: {}'.format(loss, weighted_f1_score, accuracy))

        '''Log of each Epoch'''
        writer.add_scalar('Val/Loss', loss, epoch)
        writer.add_scalar('Val/Weighted_F1_score', weighted_f1_score, epoch)
        writer.add_scalar('Val/Accuracy', accuracy, epoch)

    '''
    All train loops are finished.
    '''
    
    plt.figure()
    plt.title('Loss vs. epochs')
    plt.plot(range(args.epochs), all_train_loss, label='Training loss')
    plt.plot(range(args.epochs), all_val_loss, label='Val loss')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Metrics vs. epochs')
    plt.plot(range(args.epochs), all_train_weighted_f1_score, label='Training F1')
    plt.plot(range(args.epochs), all_train_accuracy, label='Training accuracy')
    plt.plot(range(args.epochs), all_val_weighted_f1_score, label='Val F1')
    plt.plot(range(args.epochs), all_val_accuracy, label='Val accuracy')
    plt.legend()
    plt.show()

    
    model.eval()

    best_F1 = 0
    for metric in ['best_val_loss', 'best_val_accuracy', 'best_val_weighted_f1_score']:
        
        print('\n*** Start testing w/ {} model ***'.format(metric))

        model_name = [f for f in os.listdir(model_path) if f.endswith(".pt") and metric in f][0]
        model.load_state_dict(torch.load(os.path.join(model_path, model_name)))

        all_test_logits = []
        all_test_labels = []
        for batch in tqdm(test_dataloader, desc='Testing: {}'.format(metric), unit='batch'):
            labels = batch[-1]
            data = batch[:-1]
            with torch.no_grad():
                logits = model(*data)
            all_test_logits.append(logits)
            all_test_labels.append(labels)

        all_test_logits = torch.cat(all_test_logits).cpu()
        all_test_preds = torch.argmax(all_test_logits, dim=1)
        all_test_labels = torch.cat(all_test_labels).cpu()

        
        with torch.no_grad():
            loss = loss_fn(all_test_logits, all_test_labels).item()
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_loss_' + metric, loss)

        
        all_test_preds = all_test_preds.detach().numpy()
        all_test_labels = all_test_labels.detach().numpy()
        accuracy = accuracy_score(all_test_labels, all_test_preds)
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_accuracy_' + metric, accuracy, step=step)

        
        weighted_f1_score = f1_score(all_test_labels, all_test_preds, average='weighted')
        if args.logger == 'mlflow':
            mlflow.log_metric('best_test_weighted_f1_score_' + metric, weighted_f1_score, step=step)

        '''
            compute and store the best classification report
        '''
        if weighted_f1_score > best_F1:
            best_F1 = weighted_f1_score
            report = classification_report(all_test_labels, all_test_preds, digits=2)
            out_path = os.path.join(model_path, 'classification_report.txt')
            with open(out_path, "w") as f:
                f.write(report)

        if args.logger == 'mlflow':
            artifact_path = 'evaluators/class_report_{}'.format(metric)
            mlflow.log_artifact(out_path, artifact_path=artifact_path)

        mlflow.pytorch.log_model(model, 'model_' + metric)

        print('Test loss: {} | Test weighted F1: {} | Test accuracy: {}'.format(loss, weighted_f1_score, accuracy))

    if args.logger == 'mlflow':
        shutil.rmtree(model_path)


if __name__ == "__main__":
    main(args=parse_arguments())
