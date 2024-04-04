# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


# Wandb and Logger
import wandb

# ECE
from domainbed.calibration.ece import init_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')

    parser.add_argument('--wandb_entity_name', type=str, default="hoge")
    parser.add_argument('--wandb_project_name', type=str, default="default_project")
    parser.add_argument('--wandb_exp_id', type=str, default="99999")

    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=1e-08)
    parser.add_argument('--rho', type=float, default=0.05)

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--model', type=str, default="resnet50")

    parser.add_argument('--irm_lambda', type=float, default=50000)
    parser.add_argument('--irm_penalty_anneal_iters', type=int, default=5000)

    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--optimizer_name', type=str, default="adam")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])


    parser.add_argument('--hparams_path', type=str,
        help='json file_path of hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')

    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')

    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)


    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    
    parser.add_argument('--top_n', type=int, default=1, help="Top N Eigenvalues for Hessian")
    parser.add_argument('--calc_hessian_interval', type=int, default=1000, help="Frequency of Hessian Calculation")
    parser.add_argument('--calc_hessian', action='store_true')
    parser.add_argument('--no_label_noise', action='store_true')

    parser.add_argument('--wandb_offline', action = 'store_true')
    
    # for BIRM
    parser.add_argument('--birm_type', type=str, default="bayes_fullbatch")
    parser.add_argument('--birm_sd', type=float, default=0.1)
    parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
    
    # for IBIRM
    parser.add_argument('--ib_lambda', type=float, default=1.)
    parser.add_argument('--ib_penalty_anneal_iters', type=int, default=0)
    
    # for PAIR
    parser.add_argument('--pair_pref', type=int, default=0)
    
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams_path:
        with open(args.hparams_path) as hparams_f:
            hparams.update(json.load(hparams_f))

    print('(Before Update) HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    # ======================== Wandb ================================
    print('================[Start of Wandb Logging]================') 
    run_id = f'{args.wandb_exp_id}_{args.optimizer_name}_{args.lr}_{args.seed}'

    wandb_configs = {
        'lr': args.lr,
        'beta_1': args.beta_1,
        'beta_2': args.beta_2,
        'eps': args.eps,
        'rho': args.rho,

        'momentum': args.momentum,
        'alpha': args.alpha,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,

        'irm_lambda': args.irm_lambda,
        'irm_penalty_anneal_iters': args.irm_penalty_anneal_iters,

        'dataset': args.dataset,
        'algorithm': args.algorithm,
        'optimizer_name': args.optimizer_name,
        'model': args.model,
        'seed': args.seed,
        'task': args.task,
        'test_envs': args.test_envs,
        'output_dir': args.output_dir,
        'holdout_fraction': args.holdout_fraction,

        'top_n': args.top_n,
        'calc_hessian': args.calc_hessian,
        'calc_hessian_interval': args.calc_hessian_interval,
    }
    hparams.update(wandb_configs)
    
    if args.algorithm == "BIRM":
        birm_configs = {
            'birm_type': args.birm_type,
            'birm_sd': args.birm_sd,
            'l2_regularizer_weight': args.l2_regularizer_weight,
        }
        wandb_configs.update(birm_configs)
        hparams.update(birm_configs)
    
    if args.algorithm == "IBIRM":
        ibirm_configs = {
            'ib_lambda': args.ib_lambda,
            'ib_penalty_anneal_iters': args.ib_penalty_anneal_iters,
        }
        wandb_configs.update(ibirm_configs)
        hparams.update(wandb_configs)
        
    if args.algorithm == "PAIR":
        pair_configs = {
            'pair_pref': args.pair_pref,
        }
        wandb_configs.update(pair_configs)
        hparams.update(wandb_configs)

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "dryrun"
    
    wandb.init(config=hparams, 
               project=args.wandb_project_name, 
               entity=args.wandb_entity_name, 
               name=run_id,
               )

    hparams.update(wandb.config)

    print('(After Update) HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    # ======================== Libraries and Hardware ================================
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dataset_dir = args.data_dir


    if args.dataset in vars(datasets):
        print(f'dataset_dir: {dataset_dir}')
        dataset = vars(datasets)[args.dataset](dataset_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError


    # ======================== ece config ========================
    ece_config = init_config()
    ece_config['num_reps'] = 100
    ece_config['norm'] = 1
    ece_config['ce_type'] = 'ew_ece_bin'
    ece_config['num_bins'] = 10


    # ======================== Splits ================================
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):

        uda = []
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights)) # Train ?
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")


    # ======================== Data Loaders ========================
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]

    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]


    # ======================== Model ========================
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    algorithm.to(device)

    # ======================== Iterator ========================
    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    last_results_keys = None

    # ======================== Training and Evaluation Loop ========================
    for step in range(start_step, n_steps):

        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        
        # won't use
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None

        # Training for Each Epoch 
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        # Evaluation
        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            # Log Training Results
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # Prepare Loaders for Evaluation
            evals = zip(eval_loader_names, eval_loaders, eval_weights)

            avg_val_acc = []
            avg_test_acc = 0

            # Evaluation Loop fro Each Epoch
            for name, loader, weights in evals:
                acc, ece = misc.evaluate(algorithm, loader, ece_config, device)
                results[name+'_acc'] = acc
                results[name+'_ece'] = ece

            # [ACC] ============================================================================================================================
            # https://github.com/salesforce/ensemble-of-averages/blob/main/domainbed/scripts/train.py
            agg_val_acc, nagg_val_acc = 0, 0
            for name in results.keys():
                if 'acc' in name and 'out' in name and int(name.split('env')[1].split('_')[0]) not in args.test_envs:
                    agg_val_acc += results[name]
                    nagg_val_acc += 1.
            agg_val_acc /= (nagg_val_acc + 1e-9)
            results['avg_val_acc'] = agg_val_acc

            agg_test_acc, nagg_test_acc = 0, 0
            for name in results.keys():
                if 'acc' in name and name !='avg_val_acc' and int(name.split('env')[1].split('_')[0]) in args.test_envs:
                    agg_test_acc += results[name]
                    nagg_test_acc += 1.
            agg_test_acc /= (nagg_test_acc + 1e-9)
            results['avg_test_acc'] = agg_test_acc

            # [ECE] ============================================================================================================================
            agg_val_ece, nagg_val_ece = 0, 0
            for name in results.keys():
                if 'ece' in name and 'out' in name and int(name.split('env')[1].split('_')[0]) not in args.test_envs:
                    agg_val_ece += results[name]
                    nagg_val_ece += 1.
            agg_val_ece /= (nagg_val_ece + 1e-9)
            results['avg_val_ece'] = agg_val_ece

            agg_test_ece, nagg_test_ece = 0, 0
            for name in results.keys():
                if 'ece' in name and name !='avg_val_ece' and int(name.split('env')[1].split('_')[0]) in args.test_envs:
                    agg_test_ece += results[name]
                    nagg_test_ece += 1.
            agg_test_ece /= (nagg_test_ece + 1e-9)
            results['avg_test_ece'] = agg_test_ece

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys

            misc.print_row([results[key] for key in results_keys], colwidth=12)
            wandb.log(results, step=step)
  
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])
