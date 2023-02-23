# -*- coding:utf-8 -*-

# Copyright xmuspeech (Author: Leo 2021-10-05)
# Apache 2.0
# Only support "nccl" backend multi-gpu training.
import sys
import os
import logging
import argparse
import traceback
import time
import yaml
import copy
import math
import numpy as np

import torch
sys.path.insert(0, 'subtools/pytorch')
from libs.support.logging_stdout import patch_logging_stream
import libs.support.utils as utils
import libs.support.kaldi_common as kaldi_common
import libs.training.trainer_online as trainer
import libs.training.trainer_online_sam as trainer_sam
import libs.training.lr_scheduler_online as learn_rate_scheduler
import libs.training.optim as optim
import libs.egs.egs_online as egs

"""A launcher script with python version (Snowdar's launcher to do experiments w.r.t snowdar-xvector.py).

Python version is gived (rather than Shell) to have more freedom, such as decreasing limitation of parameters that transfering 
them to python from shell.

Note, this launcher does not contain dataset preparation, augmentation, and back-end scoring etc.
    1.See subtools/recipe/voxcelebSRC/runVoxceleb_online.sh to get complete stages.
    2.An on-the-fly feature extraction mod.


How to modify this launcher:
    1.Prepare your kaldi format dataset and model.py (model blueprint);
    2.Give the path of dataset, model blueprint, etc. in main parameters field;
    3.Change the imported name of model in 'model = model_py.model_name(...)' w.r.t model.py by yourself;
    4.Modify any training parameters what you want to change (epochs, optimizer and lr_scheduler etc.);
    5.Modify parameters of extracting in stage 4 w.r.t your own training config;
    6.Run this launcher.

Conclusion: preprare -> config -> run.

How to run this launcher to train a model:
    1.For CPU-based training case. The key option is --use-gpu.
        python3 launcher.py --use-gpu=false
    2.For single-GPU training case (Default).
        python3 launcher.py
    3.For DDP-based multi-GPU training case. Note --nproc_per_node is equal to number of gpu id in --gpu-id.
        python3 -m torch.distributed.launch --nproc_per_node=2 launcher.py --gpu-id=0,1
    4.For Horovod-based multi-GPU training case. Note --np is equal to number of gpu id in --gpu-id.
        horovodrun -np 2 launcher.py --gpu-id=0,1
    5.For all of above, a runLauncher.sh script has been created to launch launcher.py conveniently.
      The key option to use single or multiple GPU is --gpu-id.
      The subtools/runPytorchLauncher.sh is a soft symbolic which is linked to subtools/pytorch/launcher/runLauncher.sh, 
      so just use it.

        [ CPU ]
            subtools/runPytorchLauncher.sh launcher.py --use-gpu=false

        [ Single-GPU ]
        (1) Auto-select GPU device
            subtools/runPytorchLauncher.sh launcher.py
        (2) Specify GPU device
            subtools/runPytorchLauncher.sh launcher.py --gpu-id=2

        [ Multi-GPU ]
        (1) Use DDP solution (Default).
            subtools/runPytorchLauncher.sh launcher.py --gpu-id=2,3 --multi-gpu-solution="ddp"
        (2) Use Horovod solution.
            subtools/runPytorchLauncher.sh launcher.py --gpu-id=2,3 --multi-gpu-solution="horovod"

If you have any other requirements, you could modify the codes in anywhere. 
For more details of multi-GPU devolopment, see subtools/README.md.
"""

# Logger
# Logger
patch_logging_stream(logging.INFO)
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [ %(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ]\n#### %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Parser: add this parser to run launcher with some frequent options (really for conveninece).
parser = argparse.ArgumentParser(
    description="""Train xvector framework with pytorch.""",
    formatter_class=argparse.RawTextHelpFormatter,
    conflict_handler='resolve')

parser.add_argument("--stage", type=int, default=3,
                    help="The stage to control the start of training epoch (default 3).\n"
                         "    stage 0: Generate raw wav kaldidir which contains utt2chunk and utt2dur. (preprocess_raw_wav_egs.sh).\n"
                         "    stage 1: remove utts (preprocess_raw_wav_egs.sh).\n"
                         "    stage 2.1: get chunk egs (preprocess_raw_wav_egs.sh).\n"
                         "    stage 2.2: Prepare speech augment csv files.\n"
                         "    stage 3: Training.\n"
                         "    stage 4: extract xvector.")

parser.add_argument("--endstage", type=int, default=4,
                    help="The endstage to control the endstart of training epoch (default 4).")

parser.add_argument("--train-stage", type=int, default=-1,
                    help="The stage to control the start of training epoch (default -1).\n"
                         "    -1 -> creating model_dir.\n"
                         "     0 -> model initialization (e.g. transfer learning).\n"
                         "    >0 -> recovering training.")

parser.add_argument("--force-clear", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="Clear the dir generated by preprocess.")

parser.add_argument("--pre-rirmusan", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help="Prepare the openrir and musan dataset for adding reverb and noises.")

parser.add_argument('--use-amp', type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help='Use automatic mixed precision training')

parser.add_argument("--skip-nan-batch", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="Whether skip optimizer stepping when the gradient has nan/inf values.")

parser.add_argument("--accum-grad", type=int, default=1,
                    help="Using accumulate grad.")

parser.add_argument("--multi-gpu-solution", type=str, default="ddp",
                    choices=["ddp"],
                    help="if number of gpu_id > 1, this option will be valid to init a multi-gpu solution.")

parser.add_argument("--use-gpu", type=str, action=kaldi_common.StrToBoolAction,
                    default=True, choices=["true", "false"],
                    help="Use GPU or not.")

parser.add_argument("--gpu-id", type=str, default="",
                    help="If NULL, then it will be auto-specified.")

parser.add_argument("--benchmark", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help="If true, save training time but require a little more gpu-memory.")

parser.add_argument("--run-lr-finder", type=str, action=kaldi_common.StrToBoolAction,
                    default=False, choices=["true", "false"],
                    help="If true, run lr finder rather than training.")

parser.add_argument("--sleep", type=int, default=0,
                    help="The waiting time to launch a launcher.")

parser.add_argument("--local_rank", type=int, default=0,
                    help="Do not delete it when using DDP-based multi-GPU training.\n"
                         "It is important for torch.distributed.launch.")

args = parser.parse_args()
##
######################################################### PARAMS ########################################################
##
##--------------------------------------------------##
# Control options
stage = max(0, args.stage)
endstage = min(4, args.endstage)
# train_stage = max(-1, args.train_stage)
##--------------------------------------------------##
# Preprocess options
force_clear = args.force_clear
preprocess_nj = 20

# The number of iters per epoch will be about 1/3 of the fixchunk mode (whole_utt=False)
# The number of epochs are supposed to be three-fold(10 -> 30)
whole_utt = True
random_segment = False # Works only when whole_utt is False
seg_dur = 3.015
# seg_dur = 6.015
# dynamic_dur = True
dynamic_dur = False
amp_th = 100
de_silence = False
vad_wav_savdir = "export/vad_wav"
# vad_wav_savdir = "export/yourpath"



min_len = 2.0
max_len = 1000.0
limit_utts = 8 # The spker whose utts < limit_utts will be removed

valid_split_type = "--total-spk"  # --total-spk or --default
valid_utts = 2048
valid_chunk_num = 2
valid_fix_chunk_num = False

##--------------------------------------------------##
# Prepare speech augmention csv files.
pre_rirmusan = args.pre_rirmusan  # whether skip this stage.
openrir_folder = "export/path"  # where contains RIRS_NOISES folder.
musan_folder = "export/path"    # where contains musan folder.
# csv_aug_folder = "exp/aug_csv"  # csv file location.
csv_aug_folder = "exp/aug_csv_6s"
savewav_folder = "export/path"  # save the noise seg into SSD.
max_noise_len = seg_dur  # The max dur of noise.
##--------------------------------------------------##
# Training options
use_amp = args.use_amp
skip_nan_batch = args.skip_nan_batch
accum_grad = args.accum_grad


use_gpu = args.use_gpu  # Default true.
# If true, save much training time but require a little more gpu-memory.
benchmark = args.benchmark
gpu_id = args.gpu_id  # If NULL, then it will be auto-specified.
run_lr_finder = args.run_lr_finder
# print(f"gpu_id:{args.gpu_id}")
use_w2v2 = True
# PTM =["w2v2", "hubert", "unispeech-sat", "wavlm"]
PTM = "wavlm"
PTM_nlayer = 10 # -1 means using layer sum
# PTM_dir = "/data/huggingface/models/wav2vec2-large-xlsr-53"
# PTM_dir = "/data/huggingface/models/wav2vec2-large-en"
PTM_dir = "/data/huggingface/models/wavlm_model"
# PTM_dir = "/data/huggingface/models/hubert_model"
# PTM_dir = "/data/huggingface/models/unispeech_sat_model"

# The ft_sec_stage shoud be manually set for PTM-based training
# When the ft_sec_stage is true, we unfreeze the feature extractor and set the max_lr to 5e-5
ft_sec_stage = False
epochs_first_stage = 20
epochs_second_stage = 20
total_epochs = epochs_first_stage + epochs_second_stage
# large margin training
ft_large_margin = False
ft_lm_epochs = 5
# for first stage
epochs = epochs_first_stage
# train_stage = max(-1, args.train_stage)
train_stage = -1
if ft_sec_stage:
    epochs = total_epochs # Total epochs to train. It is important.
    train_stage = epochs_first_stage
# for dynamic
epoch_iters = 2129 # for voxceleb2 when n_gpus is 4 and total_btz is 512
# print epoch_iters in stage 3 for detail
total_iter = total_epochs * epoch_iters 
cur_iter = int((train_stage/total_epochs) * total_iter) if train_stage != -1 else 0
if ft_large_margin:
    # Since the longer seg_dur may cause GPU OOM, total_btz: 512 -> 256
    assert ft_sec_stage == True
    epochs = total_epochs + ft_lm_epochs
    train_stage = total_epochs
    # for dynamic
    epoch_iters = 2129 * 2
    total_iter = ft_lm_epochs * epoch_iters
    cur_iter = 0

compute_batch_num_valid = 10
# About validation computation and loss reporting. If report_times_every_epoch is not None,
report_interval_iters = 100
# then compute report_interval_iters by report_times_every_epoch.

suffix = "params"  # Used in saved model file.
# Other options
exist_model = ""  # Use it in transfer learning.
##--------------------------------------------------##
# Main params
# traindata = "data/raw/voxceleb2_dev"
traindata = "data/raw/voxceleb2_dev"
# traindata_for_egs = "data/raw/voxceleb2_dev"
traindata_for_egs = "data/raw/voxceleb2_dev_egs"
# egs_dir = "exp/egs/voxceleb2_dev_vad"
egs_dir = "exp/egs/voxceleb2_dev_whole"
data_type = 'shard'
num_utts_per_shard = 2000
shard_dir = '/work/w2v2_sv/shard_whole'
# shard_dir = 'export/yourpath'
egs_conf = "subtools/conf/egs_pre_sre_ecapa.yaml"
# egs_conf="subtools/conf/egs_pre_sre.yaml"
# model_blueprint = "subtools/pytorch/model/ecapa_tdnn_xvector.py"
model_blueprint = "subtools/pytorch/model/ecapa_tdnn_xvector_ssl.py"
# model_dir = "exp/ecapa_c1024"
model_dir = "exp/ecapa_tdnn_xvector_ssl_whole_3s"

##--------------------------------------------------##
# Define model_params by model_blueprint w.r.t your model's __init__(model_params).
model_params = {
    "use_w2v2":use_w2v2,
    "PTM":PTM, 
    "PTM_dir":PTM_dir,
    "PTM_nlayer":PTM_nlayer,
    "aug_dropout": 0., "tail_dropout": 0.,
    "training": True, "extracted_embedding": "near",
    "ecapa_params": {"channels": 1024,
                     "embd_dim": 192,
                     "mfa_conv": 1536,
                     "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True}},

    # statistics, lde, attentive, multi-head, multi-resolution
    "pooling": "ecpa-attentive",
    "pooling_params": {
        "hidden_size": 128,
        "time_attention": True,
        "stddev": True,
    },
    "fc1": False,
    "fc1_params": {
        "nonlinearity": 'relu', "nonlinearity_params": {"inplace": True},
        "bn-relu": False,
        "bn": True,
        "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

    "fc2_params": {
        "nonlinearity": '', "nonlinearity_params": {"inplace": True},
        "bn-relu": False,
        "bn": True,
        "bn_params": {"momentum": 0.5, "affine": False, "track_running_stats": True}},

    "margin_loss": True,
    # "margin_loss_params": {
        # "method": "aam", "m": 0.2, "feature_normalize": True,
        # "s": 30, "mhe_loss": False, "mhe_w": 0.01},
    "margin_loss_params": {
        "method": "aam", "m": 0.5, "feature_normalize": True, "s": 30, "mhe_loss": False, "mhe_w": 0.01,"dymatic_m": False, "total_iter":total_iter, "cur_iter": cur_iter, "m_init": 0.2, "m_final": 0.5,"sqrt": False, "square": False},
    "use_step": False,
    "step_params": {
        "margin_warm":False,
        "margin_warm_conf":{"start_epoch":1,"end_epoch":1,"offset_margin":-0.0,"init_lambda":1.0},
        "T": None,
        "m": True, "lambda_0": 0, "lambda_b": 1000, "alpha": 5, "gamma": 1e-4,
        "s": False, "s_tuple": (30, 12), "s_list": None,
        "t": False, "t_tuple": (0.5, 1.2),
        "p": False, "p_tuple": (0.5, 0.1)}
}

optimizer_params = {
    "name": "adamW",
    "learn_rate": 0.00025,
    "beta1": 0.9,
    "beta2": 0.999,
    "beta3": 0.999,
    # Should be large for decouped weight decay (adamW) and small for L2 regularization (sgd, adam).
    "weight_decay": 1e-1,
    "lookahead.k": 5,
    # 0 means not using lookahead and if used, suggest to set it as 0.5.
    "lookahead.alpha": 0.,
    "gc": False,  # If true, use gradient centralization.
    "nesterov": False,  # for sgd
    "sam": False,
    "sam.rho": 2.0, # 2.0 for adaptive
    "sam.adaptive": True,
    # "custwd_dict":{
    #     "train_len":0,
    #     "bias":0
    # }
}

lr_scheduler_params = {
    "name": "cyclic",
    "cyclic.max_lr": 1e-4,
    "cyclic.base_lr": 1e-8,
    # "cyclic.step_size_up": 24000,
    "cyclic.step_size_up": 3500,
    "cyclic.mode": 'triangular2',
}

if ft_sec_stage:
    lr_scheduler_params["cyclic.max_lr"] = 5e-5
if ft_large_margin:
    lr_scheduler_params["cyclic.max_lr"] = 1e-5

##--------------------------------------------------##
##
######################################################### START #########################################################
##
# Set seed
utils.set_all_seed(1024)
##
# Init environment
# It is used for multi-gpu training if used (number of gpu-id > 1).
# And it will do nothing for single-GPU training.
utils.init_multi_gpu_training(args.gpu_id, args.multi_gpu_solution)
##
# Set sleep time for a rest
# Use it to run a launcher with a countdown function when there are no extra GPU memory
# but you really want to go to bed and know when the GPU memory will be free.
if args.sleep > 0:
    time.sleep(args.sleep)

##
# Auto-config params
# If multi-GPU used, it will auto-scale learning rate by multiplying number of processes.
optimizer_params["learn_rate"] = utils.auto_scale_lr(
    optimizer_params["learn_rate"])
# It is used for model.step() defined in model blueprint.
if lr_scheduler_params["name"] == "warmR" and model_params["use_step"]:
    model_params["step_params"]["T"]=(lr_scheduler_params["warmR.T_max"], lr_scheduler_params["warmR.T_mult"])

# Preprocess
if stage <= 2 and endstage >= 0 and utils.is_main_training():
    # Here only give limited options because it is not convenient.
    # Suggest to pre-execute this shell script to make it freedom and then continue to run this launcher.
    kaldi_common.execute_command("bash subtools/pytorch/pipeline/preprocess_wav_egs.sh "
                                 "--stage {stage} --endstage {endstage} --nj {nj} --whole-utt {whole_utt} --random-segment {random_segment} "
                                 "--seg-dur {seg_dur} --amp-th {amp_th} --de-silence {de_silence} --vad-wav-savdir {vad_wav_savdir} "
                                 "--min-len {min_len} --max-len {max_len} --limit-utts {limit_utts} "
                                 "--valid-split-type {valid_split_type} --valid-num-utts {valid_utts} --valid-chunk-num {valid_chunk_num} "
                                 "--valid-fix-chunk-num {valid_fix_chunk_num} --force-clear {force_clear} "
                                 "--pre-rirmusan {pre_rirmusan} --openrir-folder {openrir_folder} --musan-folder {musan_folder} "
                                 "--csv-aug-folder {csv_aug_folder} --savewav-folder {savewav_folder} --max-noise-len {max_noise_len} "
                                 "--data-type {data_type}  --shard-dir {shard_dir} --num-utts-per-shard {num_utts_per_shard} "
                                 "{traindata} {traindata_for_egs} {egs_dir}".format(stage=stage, endstage=endstage, nj=preprocess_nj,
                                                                                    whole_utt=str(whole_utt).lower(), random_segment=str(random_segment).lower(), seg_dur=seg_dur,
                                                                                    amp_th=amp_th, de_silence=str(de_silence).lower(), vad_wav_savdir=vad_wav_savdir,
                                                                                    min_len=min_len, max_len=max_len, limit_utts=limit_utts, valid_split_type=valid_split_type,
                                                                                    valid_utts=valid_utts, valid_chunk_num=valid_chunk_num, valid_fix_chunk_num=str(valid_fix_chunk_num).lower(),
                                                                                    force_clear=str(force_clear).lower(), pre_rirmusan=str(pre_rirmusan).lower(), openrir_folder=openrir_folder,
                                                                                    musan_folder=musan_folder, csv_aug_folder=csv_aug_folder, savewav_folder=savewav_folder, max_noise_len=max_noise_len,
                                                                                    data_type=data_type, num_utts_per_shard=num_utts_per_shard, shard_dir=shard_dir,
                                                                                    traindata=traindata, traindata_for_egs=traindata_for_egs, egs_dir=egs_dir))

# Train model
if stage <= 3 <= endstage:
    if utils.is_main_training():
        logger.info("Get model_blueprint from model directory.")
    # Save the raw model_blueprint in model_dir/config and get the copy of model_blueprint path.
    model_blueprint = utils.create_model_dir(
        model_dir, model_blueprint, stage=train_stage)

    if utils.is_main_training():
        logger.info("Load egs to bunch.")
    # The dict [info] contains feat_dim and num_targets
    with open(egs_conf, 'r') as fin:
        egs_params = yaml.load(fin, Loader=yaml.FullLoader)
        egs_params['dataset_conf']['csv_aug_folder'] = csv_aug_folder
        egs_params['dataset_conf']['use_w2v2'] = True if use_w2v2 else False
        if ft_large_margin:
            egs_params['dataset_conf']['batch_conf']['batch_size'] = 64
        egs_params['dataset_conf']['random_chunk_size'] = seg_dur
        egs_params['data_loader_conf']['prefetch_factor'] = 20
        if dynamic_dur:
            egs_params['dataset_conf']['dynamic_random_chunk'] = True
            # egs_params['data_loader_conf']['num_workers'] = 0
            batch_size = egs_params['dataset_conf']['batch_conf']['batch_size']
            egs_params['dataset_conf']['dynamic_conf'] = {"batch_size":batch_size, "total_iter":total_iter, "cur_iter":cur_iter, "chunk_len_init":2.0, "chunk_len_final":6.0, "sqrt":False, "square":True}
        
    bunch, info = egs.BaseBunch.get_bunch_from_egsdir(egs_dir, egs_params)
    feat_extraction_config = copy.deepcopy(
        egs_params['dataset_conf']['feature_extraction_conf'])
    feat_extraction_config['kaldi_featset']['dither'] = 0.0
    feat_config_path = os.path.join(model_dir, 'config', 'feat_conf.yaml')
    if utils.is_main_training():
        with open(feat_config_path, 'w') as fou:
            yaml.dump(feat_extraction_config, fou)

    if utils.is_main_training():
        logger.info("Create model from model blueprint.")
    # Another way: import the model.py in this python directly, but it is not friendly to the shell script of extracting and
    # I don't want to change anything about extracting script when the model.py is changed.
    model_py = utils.create_model_from_py(model_blueprint)

    model = model_py.ECAPA_TDNN(
        info["feat_dim"], info["num_targets"], **model_params)
    if use_w2v2:
        # Download the PTM from https://huggingface.co/models
        # All the PTM paths should match the paths used in ecapa_tdnn_xvector_ssl.py
        # We regard all the PTMs as model.wav2vec2
        if PTM == "w2v2":
            w2v2_path = PTM_dir + "/pytorch_model.bin"
            state_dict = torch.load(w2v2_path, map_location="cpu")
        elif PTM == "hubert":
            # The state_dict of hubert model starts with "hubert." which needed to be replaced with 'wav2vec2'.
            w2v2_hubert_path = PTM_dir + "/pytorch_model_rename.bin"
            state_dict = torch.load(w2v2_hubert_path, map_location="cpu")
        elif PTM == "unispeech-sat":
            # The state_dict of unispeech_sat model starts with "unispeech_sat." which needed to be replaced with 'wav2vec2'.
            w2v2_unisat_path = PTM_dir + "/pytorch_model_rename.bin"
            state_dict = torch.load(w2v2_unisat_path, map_location="cpu")
        elif PTM == "wavlm":
            # The state_dict of wavlm model starts with "wavlm." which needed to be replaced with 'wav2vec2'.
            w2v2_wavlm_path = PTM_dir + "/pytorch_model_rename.bin"
            state_dict = torch.load(w2v2_wavlm_path, map_location="cpu")
        else:
             raise ValueError("Use PTM among [w2v2, hubert, unispeech-sat, wavlm]")
        # It is recommended to set strict=True to check whether the parameters of PTM are successfully loaded from the error message, and then set it to False.
        # model.load_state_dict(state_dict,strict=True)
        model.load_state_dict(state_dict,strict=False)
        model.PTM.config.mask_time_prob = model.PTM.config.mask_feature_prob = 0.0
        model.PTM.config.layerdrop = 0.00
        model.PTM.config.attention_dropout = model.PTM.config.hidden_dropout = model.PTM.config.feat_proj_dropout = 0.00
        model.PTM.config.save_pretrained(model_dir + "/PTM_config")
        model.PTM.masked_spec_embed.requires_grad = False
        model.freeze_feature_extractor()
        # model.freeze_encoder_layers(10)
        if not ft_sec_stage:
            model.freeze_base_model()
    epoch_iters = (info['epoch_iters']//accum_grad)
    # print(f"epoch_iters:{epoch_iters}")
    if hasattr(model,'margin_warm'):
        model.margin_warm.update_step_range(epoch_iters)

    # If multi-GPU used, then batchnorm will be converted to synchronized batchnorm, which is important
    # to make peformance stable.
    # It will change nothing for single-GPU training.
    model = utils.convert_synchronized_batchnorm(model)

    # if utils.is_main_training():
        # print(model)
        # p1=sum(p.numel() for p in model.parameters())
        # script_model = copy.deepcopy(model)
        # script_model.loss=None
        # p2 = sum(p.numel() for p in script_model.parameters())
        # logger.info("model params w/o proj layer: {} / {} .".format(p1,p2))
        # script_model = torch.jit.script(script_model)
        # script_model.save(os.path.join(model_dir, 'init.zip'))
        # logger.info("The number of steps per epoch is about {}.".format(epoch_iters))           
        # logger.info("Define optimizer and lr_scheduler.")
        # del script_model
        
    optimizer = optim.get_optimizer(model, optimizer_params)
    lr_scheduler = learn_rate_scheduler.LRSchedulerWrapper(
        optimizer, lr_scheduler_params)

    # Record params to model_dir

    if utils.is_main_training():
        utils.write_list_to_file([egs_params, model_params, optimizer_params,
                                  lr_scheduler_params], model_dir+'/config/params.dict',yml=True)

    if utils.is_main_training():
        logger.info("Init a simple trainer.")
    # Package(Elements:dict, Params:dict}. It is a key parameter's package to trainer and model_dir/config/.
    package = ({"data": bunch, "model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler},
               {"model_dir": model_dir, "model_blueprint": model_blueprint, "exist_model": exist_model, "accum_grad": accum_grad,
                "start_epoch": train_stage, "epochs": epochs, "use_gpu": use_gpu, "gpu_id": gpu_id, "use_amp": use_amp,
                "skip_nan_batch": skip_nan_batch, "benchmark": benchmark, "suffix": suffix, "compute_batch_num_valid": compute_batch_num_valid,
                "report_interval_iters": report_interval_iters, "record_file": "train.csv"})
    train_exec = trainer_sam if isinstance(optimizer,optim.SAM) else trainer

    execuer = train_exec.SimpleTrainer(package)

    if run_lr_finder:
        execuer.run_lr_finder("lr_finder.csv", init_lr=1e-8,
                              final_lr=10., num_iters=2000, beta=0.98)
        endstage = 3  # Do not start extractor.
    else:
        execuer.run()


# Extract xvector
if stage <= 4 <= endstage and utils.is_main_training():
    # There are some params for xvector extracting.
    data_root = "data"  # It contains all dataset just like Kaldi recipe.
    prefix = "raw"  # For to_extracted_data.
    data_type_emb = "raw"  # shard or raw or kaldi.
    de_silence = False
    amp_th = 50

    to_extracted_positions = ["near"]  # Define this w.r.t model_blueprint.
    # All dataset should be in dataroot/prefix.
    # to_extracted_data = ["voxceleb1", "voxceleb2_dev"]
    to_extracted_data = ["voxceleb1_6s", "voxceleb2_dev_6s"]
    # It is model's name, such as 10.params or final.params (suffix is w.r.t package).
    to_extracted_epochs = ["65"]
    # to_extracted_epochs = ["aver_last5_fusion"]

    nj = 10
    force = True
    use_gpu = True
    gpu_id_extract = "3,4,5,6,7"
    sleep_time = 10
    feat_config = "feat_conf.yaml"
    # max_chunk = 10000
    max_chunk = 100 * 16000
    # Run a batch extracting process.
    try:
        for position in to_extracted_positions:
            # Generate the extracting config from nnet config where
            # which position to extract depends on the 'extracted_embedding' parameter of model_creation (by my design).
            model_blueprint, model_creation = utils.read_nnet_config(
                "{0}/config/nnet.config".format(model_dir))
            # To save memory without loading some independent components.
            model_creation = model_creation.replace(
                "training=True", "training=False")
            model_creation = model_creation.replace(
                model_params["extracted_embedding"], position)
            extract_config = "{0}.extract.config".format(position)

            utils.write_nnet_config(
                model_blueprint, model_creation, "{0}/config/{1}".format(model_dir, extract_config))

            for epoch in to_extracted_epochs:
                model_file = "{0}.{1}".format(epoch, suffix)
                point_name = "{0}_epoch_{1}".format(position, epoch)

                # If run a trainer with background thread (do not be supported now) or run this launcher extrally with stage=4
                # (it means another process), then this while-listen is useful to start extracting immediately (but require more gpu-memory).
                model_path = "{0}/{1}".format(model_dir, model_file)

                while True:
                    if os.path.exists(model_path):
                        break
                    else:
                        time.sleep(sleep_time)

                for data in to_extracted_data:
                    datadir = "{0}/{1}/{2}".format(data_root, prefix, data)
                    outdir = "{0}/{1}/{2}".format(model_dir, point_name, data)
                    # Use a well-optimized shell script (with multi-processes) to extract xvectors.
                    # Another way: use subtools/splitDataByLength.sh and subtools/pytorch/pipeline/onestep/extract_embeddings.py
                    # with python's threads to extract xvectors directly, but the shell script is more convenient.
                    kaldi_common.execute_command("bash subtools/pytorch/pipeline/extract_xvectors_for_pytorch_new.sh "
                                                 " --model {model_file} --nj {nj} --use-gpu {use_gpu} --gpu-id '{gpu_id}' "
                                                 " --data-type '{data_type}' --de-silence {de_silence}  --amp-th {amp_th} --max-chunk {max_chunk} "
                                                 " --force {force} --nnet-config config/{extract_config} --feat-config config/{feat_config} --use_w2v2 {use_w2v2} "
                                                 "{model_dir} {datadir} {outdir} ".format(model_file=model_file, nj=nj,
                                                                                         use_gpu=str(use_gpu).lower(), gpu_id=gpu_id_extract, force=str(force).lower(), extract_config=extract_config,
                                                                                      feat_config=feat_config, data_type=data_type_emb, de_silence=str(de_silence).lower(), amp_th=amp_th,use_w2v2=str(use_w2v2).lower(),
                                                                                          max_chunk=max_chunk, model_dir=model_dir, datadir=datadir, outdir=outdir))
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)
