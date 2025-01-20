from torchvision.utils import save_image
import torch.nn.functional as F
from torch.distributions import Categorical
import torch


def logit_loss(model, inp, y):
  y_h = model(inp)

  # Convert y to one-hot encodings
  y_one_hot = F.one_hot(y, num_classes=y_h.size(1)).to(torch.float32)  # This will be of size [32, num_classes]

  # Compute the dot product (element-wise multiplication followed by a sum over the class dimension.)
  loss = -(y_one_hot * y_h).sum(dim=1)

  return loss, None

class DivergentLogitLoss():
  def __init__(self, div_class_dict):
    # div_class_dict should have one key for each class for which images should diverge from specific other classes. 
    # The key is a class integer label, and the value is a list of class integer labels. 
    # E.g., if we want class 4 to diverge from classes 7 and 9, we would have: {4: [7, 9]}
    self.div_class_dict = div_class_dict

  def __call__(self, model, inp, y):
    y_h = model(inp)

    # Convert y to one-hot encodings
    y_one_hot = F.one_hot(y, num_classes=y_h.size(1)).to(torch.float32)  # This will be of size [32, num_classes]

    # y is a tensor containing integer labels for classes. Check in div_class_dict. 
    # For each class integer label that is a key in self.div_class_dict, set the "logits" in y_one_hot at positions corresponding to 
    # the integers in the value of that dictionary item to -1. 

    # Iterate over each class in y to modify y_one_hot based on div_class_dict
    for i, label in enumerate(y):
      if label.item() in self.div_class_dict:
        # Get the classes that this label should diverge from
        diverge_classes = self.div_class_dict[label.item()]
        # Set logits for these classes to -1
        y_one_hot[i, diverge_classes] = -1 / len(diverge_classes)

    # Compute the dot product (element-wise multiplication followed by a sum over the class dimension.)
    loss = -(y_one_hot * y_h).sum(dim=1)

    return loss, None
