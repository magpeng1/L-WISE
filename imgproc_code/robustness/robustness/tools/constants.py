import torch as ch
import cox
from cox import store


IMAGENET_16_RANGES = {
    "dog": (151, 268), 
    "house_cat": (281, 285), 
    "frog": (30, 32), 
    "turtle": (33, 37), 
    "bird": list(range(80, 100+1)) + list(range(127, 146+1)), # Originally (80, 100) in RestrictedImageNet, now also includes 127-146
    "monkey": (369, 382), # Originally (365, 382) in RestrictedImageNet (which included apes)
    "fish": [0, 1, 389, 391, 392, 393, 394, 395, 396, 397], # Originally (389, 397) in RestrictedImageNet. Removed eel and added tench and goldfish. 
    "crab": (118, 121), # Could also include "hermit crab" (125)
    "insect": (300, 320), # Could also include 321:326 (butterflies). Originally (300, 319) in RestrictedImageNet
    "lizard": (38, 48),
    "snake": (52, 68), 
    "spider": (72, 77),
    "big_cat": (286, 293), 
    "bear": (294, 297),
    "rodent": [330, 331, 332, 333, 335, 336, 338], # Excluding porcupine, hedgehog, beaver, including rabbits even though they are no longer technically considered rodents
    "antelope": (351, 353),
}

RESTRICTED_IMAGNET_RANGES = [(151, 268), (281, 285), 
        (30, 32), (33, 37), (80, 100), (365, 382),
          (389, 397), (118, 121), (300, 319)]

CKPT_NAME = 'checkpoint.pt'
BEST_APPEND = '.best'
CKPT_NAME_LATEST = CKPT_NAME + '.latest'
CKPT_NAME_BEST = CKPT_NAME + BEST_APPEND

ATTACK_KWARG_KEYS = [
        'criterion',
        'constraint',
        'eps',
        'step_size',
        'iterations',
        'random_start',
        'random_restarts']

LOGS_SCHEMA = {
    'epoch':int,
    'nat_prec1':float,
    'adv_prec1':float,
    'nat_loss':float,
    'adv_loss':float,
    'train_prec1':float,
    'train_loss':float,
    'time':float
}

LOGS_TABLE = 'logs'

