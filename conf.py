import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from yacs.config import CfgNode as CfgNode

_C = CfgNode()
cfg = _C

# ----------- Base Options ----------------- #
_C.BASE = CfgNode()
_C.BASE.SEED = 10
_C.BASE.NUM_WORKERS = 4
_C.BASE.GPU_ID = 0
_C.BASE.ATTACK = "u_dia"

# --------------- DATASET Options ---------------------#
_C.DATA = CfgNode()
_C.DATA.DATASET = "radioml"
_C.DATA.NUM_CLASSES = 24
_C.DATA.BATCH_SIZE = 64
_C.DATA.SIZE = (1024,)
# ------------------ Batch Norm Options ------------------#

_C.BN = CfgNode()
_C.BN.EPSILON = 1e-5
_C.BN.MOMENTUM = 0.1

# ----------------- Optimizer Options -----------------#

_C.OPTIM = CfgNode()
_C.OPTIM.STEPS = 1
_C.OPTIM.LR = 1e-3
_C.OPTIM.METHOD = 'Adam'
_C.OPTIM.BETA = 0.9
_C.OPTIM.MOMENTUM = 0.9

_C.OPTIM.WD = 0.0
_C.OPTIM.TEMP = 1.0

_C.OPTIM.ADAPT = "ent"
_C.OPTIM.ADAPTIVE = False
_C.OPTIM.TBN = True
_C.OPTIM.UPDATE = True

# ----- Path Options ------------#
_C.PATH = CfgNode()
# _C.PATH.DATA_PATH = 'corrupted_data/severity_5'
# _C.PATH.NEW_NOISE_PATH = 'corrupted_data/cifar10c_bar'
# _C.PATH.SOURCE_DATA = 'data'
# _C.PATH.NOISE_ENC = 'saved_model/noise_encoder/cifar10c.pt'
# _C.PATH.MODEL_NAME = 'resenet50cifar10'
# _C.PATH.SAVED_MODEL= 'saved_model_corrupted' + '/' + _C.PATH.MODEL_NAME + '_' +_C.DATA.CORRUPTION +'.pt'
_C.PATH.SOURCE_MODEL = 'resnet18_2.pth'


# ------------------------- TTA Options --------------------
_C.TTA = CfgNode()

# RoTTA
_C.TTA.NAME = "tent"

_C.TTA.ROTTA = CfgNode()
_C.TTA.ROTTA.MEMORY_SIZE = 64
_C.TTA.ROTTA.UPDATE_FREQUENCY = 64
_C.TTA.ROTTA.NU = 0.001
_C.TTA.ROTTA.ALPHA = 0.05
_C.TTA.ROTTA.LAMBDA_T = 1.0
_C.TTA.ROTTA.LAMBDA_U = 1.0

_C.TTA.NOTE = CfgNode()
_C.TTA.NOTE.IABN_K = 3.0
_C.TTA.NOTE.TEMP = 1.0
_C.TTA.NOTE.MEMORY_SIZE = 64
_C.TTA.NOTE.UPDATE_FREQUENCY = 64
_C.TTA.NOTE.SKIP_THRESH = 0.2
# -------------- Attacking Options ----------------------#

_C.DIA = CfgNode()
_C.DIA.METHOD = "PGD"
_C.DIA.MAL_NUM = 13
_C.DIA.PSEUDO = True
_C.DIA.EPS = 1.0
_C.DIA.ALPHA = 0.2
_C.DIA.STEPS = 200
_C.DIA.WHITE = True
_C.DIA.ADAPTIVE = False
_C.DIA.ADAPTIVE = False
_C.DIA.TARGETED = False
_C.DIA.PAR = 0.0
_C.DIA.WEIGHT_P = 0.0
_C.DIA.DEPRIOR = 0.0
_C.DIA.DFTESTPRIOR = 0.0
_C.DIA.LAYER = 0
_C.DIA.PSEUDO = True


