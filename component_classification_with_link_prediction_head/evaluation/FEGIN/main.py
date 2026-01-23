# import os.pa
import os, sys
import time
from shutil import copy, rmtree
from itertools import product
import pdb
import argparse
import random
import torch
import numpy as np
from kernel.datasets import get_dataset,get_circuit_dataset,MyOwnDataset
from kernel.pin_level_dataset import PinLevelDataset
from kernel.train_eval import cross_validation_with_val_set
from kernel.train_eval import cross_validation_without_val_set
from kernel.train_eval import trainFEGIN
from kernel.gcn import *
from kernel.graph_sage import *
from kernel.gin import *
from kernel.FEGIN import *
from kernel.gat import *
from kernel.graclus import Graclus
from kernel.top_k import TopK
from kernel.diff_pool import *
from kernel.global_attention import GlobalAttentionNet
from kernel.set2set import Set2SetNet
from kernel.sort_pool import SortPool
import warnings
from fegin_experiment_tracker import FEGINExperimentTracker
from kernel.dataset_component_component import ComponentComponentDataset
from kernel.dataset_component_net import ComponentNetDataset
from kernel.dataset_component_pin import ComponentPinDataset
from kernel.dataset_component_pin_net import ComponentPinNetDataset
from kernel.multitask_FEGIN import MultiTaskFEGIN
from kernel.multitask_train_eval import train_multitask_fegin
from kernel.multitask_dataset import MultiTaskCircuitDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)


# used to traceback which code cause warnings, can delete
import traceback
import warnings
import sys
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback

# file_names = ['ltspice_demos_pin_level']
# for dataset_name in file_names:
def main():
# General settings.
    parser = argparse.ArgumentParser(description='GNN for component identification and link prediction')
    parser.add_argument('--data', type=str, default='ltspice_demos',
                        help='Dataset name (e.g., ltspice_demos)')
    parser.add_argument('--clean', action='store_true', default=False,
                        help='use a cleaned version of dataset by removing isomorphism')
    parser.add_argument('--no_val', action='store_true', default=True,
                        help='if True, do not use validation set, but directly report best\
                        test performance.')
    parser.add_argument('--representation', type=str, default='component_pin_net',
                        choices=['component_component', 'component_net', 'component_pin', 'component_pin_net'],
                        help='Circuit representation to use')

    # GNN settings.
    parser.add_argument('--model', type=str, default='FEGIN', choices=['FEGIN', 'MultiTaskFEGIN'],
                        help='Model architecture')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--hiddens', type=int, default=32)
    parser.add_argument('--h', type=int, default=2, help='the height of rooted subgraph \
                        for NGNN models')
    parser.add_argument('--node_label', type=str, default="spd", 
                        help='apply distance encoding to nodes within each subgraph, use node\
                        labels as additional node features; support "hop", "drnl", "spd", \
                        "spd5", etc. Default "spd"=="spd2".')
    parser.add_argument('--use_rd', action='store_true', default=False, 
                        help='use resistance distance as additional node labels')
    parser.add_argument('--use_rp', type=int, default=None, 
                        help='use RW return probability as additional node features,\
                        specify num of RW steps here')
    parser.add_argument('--max_nodes_per_hop', type=int, default=None)
    parser.add_argument('--emb_size', type=int, default=250)

    # Multi-task specific settings
    parser.add_argument('--lambda_node', type=float, default=1.0,
                        help='Weight for component classification loss')
    parser.add_argument('--lambda_edge', type=float, default=1.0,
                        help='Weight for link prediction loss')
    parser.add_argument('--neg_sampling_ratio', type=float, default=2.0,
                        help='Ratio of negative to positive edge samples')

    # Training settings.
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1E-2)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)

    # Other settings.
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--search', action='store_true', default=False, 
                        help='search hyperparameters (layers, hiddens)')
    parser.add_argument('--save_appendix', default='', 
                        help='what to append to save-names when saving results')
    parser.add_argument('--keep_old', action='store_true', default=False,
                        help='if True, do not overwrite old .py files in the result folder')
    parser.add_argument('--reprocess', action='store_true', default=False,
                        help='if True, reprocess data')
    parser.add_argument('--cpu', action='store_true', default=False, help='use cpu')
    args = parser.parse_args()
    dataset_name = args.data

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    file_dir = os.path.dirname(os.path.realpath('__file__'))
    if args.save_appendix == '':
        args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
    args.res_dir = os.path.join(file_dir, 'results/model_{}_dataset_{}'.format(args.model, dataset_name))
    print('Results will be saved in ' + args.res_dir)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir) 
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')

    tracker = FEGINExperimentTracker(
        experiment_name=args.save_appendix if args.save_appendix else "default",
        dataset_name=dataset_name,
        model_name=args.model
    )
    config_dict = {
        'dataset': dataset_name,
        'representation': args.representation,
        'model': args.model,
        'layers': args.layers,
        'hiddens': args.hiddens,
        'h': args.h,
        'node_label': args.node_label,
        'use_rd': args.use_rd,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lr_decay_factor': args.lr_decay_factor,
        'lr_decay_step_size': args.lr_decay_step_size,
        'emb_size': args.emb_size,
        'seed': args.seed
    }
    if args.model == 'MultiTaskFEGIN':
        config_dict.update({
            'lambda_node': args.lambda_node,
            'lambda_edge': args.lambda_edge,
            'neg_sampling_ratio': args.neg_sampling_ratio
        })
    tracker.log_config(config_dict)

    def logger(info):
        f = open(os.path.join(args.res_dir, 'log.txt'), 'a')
        print(info, file=f)
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )

    
    datasets = [args.data]
    if args.search:
        if args.h is None:
            layers = [2, 3, 4, 5]
            hiddens = [32]
            hs = [None]
        else:
            layers = [3, 4, 5, 6]
            hiddens = [32, 32, 32, 32]
            # hs = [2, 3, 4, 5]
            hs = [3]
    else:
        layers = [args.layers]
        hiddens = [args.hiddens]
        hs = [args.h]

    if args.model == 'all':
        nets = [NestedGAT, GAT]
        # nets = [NestedGCN, NestedGraphSAGE,NestedGIN,GCN, GraphSAGE, GIN]
    else:
        nets = [eval(args.model)]

    
    # device = 'cpu'### only for GAT and nestedGAT
    if args.no_val:
        cross_val_method = cross_validation_without_val_set
    else:
        cross_val_method = cross_validation_with_val_set

    results = []
    for dataset_name, Net in product(datasets, nets):
        best_result = (float('inf'), 0, 0)
        log = '-----\n{} - {}'.format(dataset_name, Net.__name__)
        print(log)
        logger(log)
        if args.h is not None:
            combinations = zip(layers, hiddens, hs)
        else:
            combinations = product(layers, hiddens, hs)
        training_log = []
        for num_layers, hidden, h in combinations:
            log = "Using {} layers, {} hidden units, h = {}".format(num_layers, hidden, h)
            print(log)
            logger(log)
            if args.model == 'MultiTaskFEGIN':
                # dictionary
                train_dataset = MultiTaskCircuitDataset(
                    root="data/",
                    name=dataset_name,
                    representation=args.representation,
                    h=args.h,
                    max_nodes_per_hop=args.max_nodes_per_hop,
                    node_label=args.node_label,
                    use_rd=args.use_rd,
                    neg_sampling_ratio=args.neg_sampling_ratio,
                    max_pins=2,
                    split='train'
                )
                test_dataset = MultiTaskCircuitDataset(
                    root="data/",
                    name=dataset_name,
                    representation=args.representation,
                    h=args.h,
                    max_nodes_per_hop=args.max_nodes_per_hop,
                    node_label=args.node_label,
                    use_rd=args.use_rd,
                    neg_sampling_ratio=args.neg_sampling_ratio,
                    max_pins=2,
                    split='test'
                )
                print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
            elif args.model == "FEGIN":
                print(f"Loading {args.representation} representation dataset")
    
                dataset_classes = {
                    'component_component': ComponentComponentDataset,
                    'component_net': ComponentNetDataset,
                    'component_pin': ComponentPinDataset,
                    'component_pin_net': ComponentPinNetDataset,
                }
                
                dataset_class = dataset_classes[args.representation]
                dataset = dataset_class(
                    root="data/",
                    name=dataset_name,
                    representation=args.representation,
                    h=args.h,
                    max_nodes_per_hop=args.max_nodes_per_hop,
                    node_label=args.node_label,
                    use_rd=args.use_rd
                )
            else:
                dataset = MyOwnDataset("data/",dataset_name,
                    h, 
                    args.node_label, 
                    args.use_rd, 
                    args.max_nodes_per_hop)
            if args.model == 'MultiTaskFEGIN':
                model = MultiTaskFEGIN(train_dataset, args.layers, args.hiddens, args.emb_size, args.node_label!='no', args.use_rd, lambda_node=args.lambda_node, lambda_edge=args.lambda_edge, max_pins=2)
                results = train_multitask_fegin(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    dataset_name=dataset_name,
                    model=model,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    lr_decay_factor=args.lr_decay_factor,
                    lr_decay_step_size=args.lr_decay_step_size,
                    weight_decay=0,
                    device=device,
                    logger=logger,
                    tracker=tracker,
                    representation=args.representation,
                    lambda_node=args.lambda_node,
                    lambda_edge=args.lambda_edge
                )
                tracker.log_test_results(
                    test_acc=results['best_acc'],
                    test_f1_weighted=results['best_node_f1_weighted'],
                    test_f1_macro=results['best_node_f1_macro'],
                    all_preds=results['predictions'],
                    all_labels=results['labels'],
                    class_names=['R', 'C', 'V', 'X', 'M']
                )
                tracker.log_best_scores(results['best_edge_auc'], results['best_combined_score'])
            elif args.model=="FEGIN":
                model = Net(dataset, num_layers, hidden, args.emb_size,args.node_label!='no', args.use_rd)
                loss, f1,f1_std, fegin_results = trainFEGIN(
                    dataset,dataset_name,
                    model,
                    folds=3,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    lr_decay_factor=args.lr_decay_factor,
                    lr_decay_step_size=args.lr_decay_step_size,
                    weight_decay=0,
                    device=device, 
                    logger=logger, 
                    tracker=tracker,
                    representation=args.representation)
                tracker.log_test_results(
                    test_acc=fegin_results['best_acc'],
                    test_f1_weighted=fegin_results['best_f1_weighted'],
                    test_f1_macro=fegin_results['best_f1_macro'],
                    all_preds=fegin_results['predictions'],
                    all_labels=fegin_results['labels'],
                    class_names=['R', 'C', 'V', 'X', 'M']
                )
                if f1 > best_result[1]:
                    best_result = (loss,f1,f1_std)
                    best_hyper = (num_layers, hidden, h)
            else:
                model = Net(dataset, num_layers, hidden, args.node_label!='no', args.use_rd)
                loss, auc, auc_std,f1,f1_std = cross_val_method(
                    dataset,
                    model,
                    folds=3,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    lr_decay_factor=args.lr_decay_factor,
                    lr_decay_step_size=args.lr_decay_step_size,
                    weight_decay=0,
                    device=device, 
                    logger=logger)
                if f1 > best_result[1]:
                    best_result = (loss,f1,f1_std)
                    best_hyper = (num_layers, hidden, h)
            
            if args.model == "FEGIN":
                print("========================FINAL RESULTS================================")
                print(f"FEGIN weighted F1: {fegin_results['best_f1_weighted']:.4f} ± {f1_std:.4f}")
            elif args.model == "MultiTaskFEGIN":
                print("========================FINAL RESULTS================================")
                print(f"Component Classification F1: {results['best_node_f1_weighted']:.4f}")
                print(f"Link Prediction AUC: {results['best_edge_auc']:.4f}")
                print(f"Combined Score: {results['best_combined_score']:.4f}")
        '''
        desc = 'f1:{:.3f} ± {:.3f}'.format(
            best_result[1], best_result[2]
        )
        log = 'Best result - {}, with {} layers and {} hidden units and h = {}'.format(
            desc, best_hyper[0], best_hyper[1], best_hyper[2]
        )
        print(log)
        logger(log)
        results += ['{} - {}: {}'.format(dataset_name, model.__class__.__name__, desc)]
        # break
        '''
        
    log = '-----\n{}'.format('\n'.join(results))
    print(cmd_input[:-1])
    print(log)
    logger(log)

if __name__ == "__main__":
    main()