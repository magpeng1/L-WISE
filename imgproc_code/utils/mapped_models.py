from collections import OrderedDict
import numpy as np
import torch as ch
from robustness.datasets import *
from robustness.model_utils import make_and_restore_model
from robustness.tools.folder import ImageFolder

def invert_dict(d):  # from https://github.com/AllenDowney/ThinkPython2/blob/master/code/invert_dict.py
    """Inverts a dictionary, returning a map from val to a list of keys.
    If the mapping key->val appears in d, then in the new dictionary val maps to a list that includes key.
    """
    inverse = {}
    for key in d:
        val = d[key]
        inverse.setdefault(val, []).append(key)
    return inverse

def imagenet_mapped_model(arch, superclassed_imagenet_ds, pytorch_pretrained, gpu_id=0, resume_path=None):
  imagenet_ds = DATASETS['imagenet']('')
  folder_ds = ImageFolder(root=f"{superclassed_imagenet_ds.data_path}/val", label_mapping=superclassed_imagenet_ds.label_mapping)
  class_to_idx_imagenet = ImageFolder(root=f"{superclassed_imagenet_ds.data_path}/val", label_mapping=imagenet_ds.label_mapping).class_to_idx
  class_mapper = invert_dict({class_to_idx_imagenet[k]: v for k, v in folder_ds.class_to_idx.items()})
  class_mapper = {k: np.array(v) for k, v in class_mapper.items()}
  class_mapper = OrderedDict(sorted(class_mapper.items()))

  def forward_wrapper(forward):
    def forwarded(*args, **kwargs):
      x = forward(*args, **kwargs)
      if 'with_latent' in kwargs and kwargs['with_latent']:
          return x
      x = ch.stack([x[:, v].max(1).values for v in class_mapper.values()], 1)
      return x
    return forwarded

  if isinstance(arch, str):
    net = imagenet_ds.get_model(arch, pytorch_pretrained)
  else:
    net = arch
  net.forward = forward_wrapper(net.forward)
  return make_and_restore_model(arch=net, dataset=imagenet_ds, pytorch_pretrained=pytorch_pretrained, gpu_id=gpu_id, resume_path=resume_path)