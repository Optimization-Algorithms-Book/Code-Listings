import math
from pprint import pprint
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely import geometry

def info_value_of_dtype(dtype: torch.dtype):
  """
  Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.

  Adapted from allennlp by allenai:
    https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
  """

  if dtype == torch.bool:
    raise TypeError("Does not support torch.bool")
  elif dtype.is_floating_point:
    return torch.finfo(dtype)
  else:
    return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
  """
  Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.

  Adapted from allennlp by allenai:
    https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
  """

  return info_value_of_dtype(dtype).min

def masked_log_softmax(
  x: torch.Tensor,
  mask: torch.Tensor,
  dim: int = -1,
  eps: float = 1e-45
) -> torch.Tensor:
  """
  Apply softmax to x with masking.

  Adapted from allennlp by allenai:
    https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

  Args:
    x - Tensor of arbitrary shape to apply softmax over.
    mask - Binary mask of same shape as x where "False" indicates elements
      to disregard from operation.
    dim - Dimension over which to apply operation.
    eps - Stability constant for log operation. Added to mask to avoid NaN
      values in log.
  Outputs:
    Tensor with same dimensions as x.
  """

  x = x + (mask.float() + eps).log()
  return torch.nn.functional.log_softmax(x, dim=dim)

def masked_max(
  x: torch.Tensor,
	mask: torch.Tensor,
	dim: int,
	keepdim: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
  """
  Apply max to x with masking.

  Adapted from allennlp by allenai:
    https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py

  Args:
    x - Tensor of arbitrary shape to apply max over.
    mask - Binary mask of same shape as x where "False" indicates elements
      to disregard from operation.
    dim - Dimension over which to apply operation.
    keepdim - If True, keeps dimension dim after operation.
  Outputs:
    A ``torch.Tensor`` of including the maximum values.
  """

  x_replaced = x.masked_fill(~mask, min_value_of_dtype(x.dtype))
  max_value, max_index = x_replaced.max(dim=dim, keepdim=keepdim)
  return max_value, max_index

def convert_binary_mask_to_infinity_mask(mask: torch.Tensor) -> torch.Tensor:
  """
  Convert the 0 and 1 elements in a binary mask to -inf and 0 for the
    transformer.

  Args:
    mask: Binary mask tensor.
  Outputs:
    Infinity mask tensor with same size as mask.
  """

  return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


class PointerNetwork(nn.Module):
  """
  From "Pointer Networks" by Vinyals et al. (2017)

  Adapted from pointer-networks-pytorch by ast0414:
    https://github.com/ast0414/pointer-networks-pytorch

  Args:
    n_hidden: The number of features to expect in the inputs.
  """

  def __init__(
    self,
    n_hidden: int
  ):
    super().__init__()
    self.n_hidden = n_hidden
    self.w1 = nn.Linear(n_hidden, n_hidden, bias=False)
    self.w2 = nn.Linear(n_hidden, n_hidden, bias=False)
    self.v = nn.Linear(n_hidden, 1, bias=False)

  def forward(
    self,
    x_decoder: torch.Tensor,
    x_encoder: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-16
  ) -> torch.Tensor:
    """
    Args:
      x_decoder: Encoding over the output tokens.
      x_encoder: Encoding over the input tokens.
      mask: Binary mask over the softmax input.
    Shape:
      x_decoder: (B, Ne, C)
      x_encoder: (B, Nd, C)
      mask: (B, Nd, Ne)
    """

    # (B, Nd, Ne, C) <- (B, Ne, C)
    encoder_transform = self.w1(x_encoder).unsqueeze(1).expand(
      -1, x_decoder.shape[1], -1, -1)
    # (B, Nd, 1, C) <- (B, Nd, C)
    decoder_transform = self.w2(x_decoder).unsqueeze(2)
    # (B, Nd, Ne) <- (B, Nd, Ne, C), (B, Nd, 1, C)
    prod = self.v(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
    # (B, Nd, Ne) <- (B, Nd, Ne)
    log_score = masked_log_softmax(prod, mask, dim=-1, eps=eps)
    return log_score


class ConvexNet(nn.Module):
  def __init__(
    self,
    c_inputs: int = 5,
    c_embed: int = 8,
    n_heads: int = 2,
    n_layers: int = 1,
    dropout: float = 0.1,
    c_hidden: int = 2
  ):
    super().__init__()
    self.c_hidden = c_hidden
    self.c_inputs = c_inputs
    self.c_embed = c_embed
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.dropout = dropout
    #self.TOKENS = {'<eos>':0}
    self.embedding = nn.Linear(c_inputs, c_embed, bias=False)
    encoder_layers = nn.TransformerEncoderLayer(c_embed, n_heads, c_hidden, dropout)
    self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
    decoder_layers = nn.TransformerDecoderLayer(c_embed, n_heads, c_hidden, dropout)
    self.decoder = nn.TransformerDecoder(decoder_layers, n_layers)
    self.pointer = PointerNetwork(n_hidden=c_embed)
    
  def forward(
    self,
    batch_data: torch.Tensor,
    batch_lengths: torch.Tensor,
    batch_labels: Optional[torch.Tensor] = None
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # assumes batch-first inputs
    batch_size = batch_data.shape[0]
    max_seq_len = batch_data.shape[1]
    c_embed = self.c_embed
    n_heads = self.n_heads
    TOKENS = {'<eos>':0}
    x_embed = self.embedding(batch_data)
    encoder_outputs = self.encoder(x_embed.permute(1, 0, 2))

    # make mask
    range_tensor = torch.arange(max_seq_len, device=batch_lengths.device,
      dtype=batch_lengths.dtype).expand(batch_size, max_seq_len - len(TOKENS), max_seq_len)
    each_len_tensor = batch_lengths.view(-1, 1, 1).expand(-1, max_seq_len - len(TOKENS), max_seq_len)
    mask_tensor = (range_tensor < each_len_tensor)

    if batch_labels is not None:
      # teacher forcing
      # pass through decoder
      # here memory_mask is (batch_size * n_heads, len_decoder_seq, len_encoder_seq)
      # https://discuss.pytorch.org/t/memory-mask-in-nn-transformer/55230/5
      _bl = torch.cat((torch.zeros_like(batch_labels[:, :1]), batch_labels[:, :-1]), dim=1).permute(1, 0).unsqueeze(-1)
      _bl = _bl.expand(-1, batch_size, c_embed)
      decoder_input = torch.gather(encoder_outputs, dim=0, index=_bl)
      decoder_mask = mask_tensor.repeat((n_heads, 1, 1))
      dm = convert_binary_mask_to_infinity_mask(decoder_mask)
      tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(decoder_input)).to(dm.device)
      decoder_outputs = self.decoder(decoder_input, encoder_outputs,
        tgt_mask=tgt_mask, memory_mask=dm)

      # pass through pointer network
      log_pointer_scores = self.pointer(
        decoder_outputs.permute(1, 0, 2),
        encoder_outputs.permute(1, 0, 2),
        mask_tensor)
      _, masked_argmaxs = masked_max(log_pointer_scores, mask_tensor, dim=-1)
      return log_pointer_scores, masked_argmaxs
    else:
      #
      log_pointer_scores = []
      masked_argmaxs = []
      decoder_input = encoder_outputs[:1]
      for _ in range(max_seq_len - len(TOKENS)):
        # pass through decoder network
        decoder_mask = mask_tensor[:, :len(decoder_input)].repeat((n_heads, 1, 1))
        dm = convert_binary_mask_to_infinity_mask(decoder_mask)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(decoder_input)).to(dm.device)
        decoder_outputs = self.decoder(decoder_input, encoder_outputs,
          tgt_mask=tgt_mask, memory_mask=dm)

        # pass through pointer network
        mask_subset = mask_tensor[:, :len(decoder_outputs)]
        log_pointer_score = self.pointer(
          decoder_outputs.permute(1, 0, 2),
          encoder_outputs.permute(1, 0, 2),
          mask_subset)
        _, masked_argmax = masked_max(log_pointer_score, mask_subset, dim=-1)

        # append new predictions
        log_pointer_scores.append(log_pointer_score[:, -1, :])
        new_maxes = masked_argmax[:, -1]
        masked_argmaxs.append(new_maxes)

        # mask out predicted inputs
        new_max_mask = torch.zeros((mask_tensor.shape[0], mask_tensor.shape[2]),
          dtype=torch.bool, device=mask_tensor.device)
        new_max_mask = new_max_mask.scatter(1, new_maxes.unsqueeze(1), True)
        new_max_mask[:, :2] = False
        new_max_mask = new_max_mask.unsqueeze(1).expand(-1, mask_tensor.shape[1], -1)
        mask_tensor[new_max_mask] = False

        # prepare inputs for next iteration
        next_indices = torch.stack(masked_argmaxs, dim=0).unsqueeze(-1).expand(-1, batch_size, c_embed)
        decoder_input = torch.cat((encoder_outputs[:1],
          torch.gather(encoder_outputs, dim=0, index=next_indices)), dim=0)
      log_pointer_scores = torch.stack(log_pointer_scores, dim=1)
      masked_argmaxs = torch.stack(masked_argmaxs, dim=1)
      return log_pointer_scores, masked_argmaxs
class AverageMeter(object):
  """
  Computes and stores the average and current value

  Adapted from pointer-networks-pytorch by ast0414:
    https://github.com/ast0414/pointer-networks-pytorch
  """

  def __init__(self):
    self.history = []
    self.reset(record=False)

  def reset(
    self,
    record: bool = True
  ):
    if record:
      self.history.append(self.avg)
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(
    self,
    val: Union[float, int],
    n: int = 1
  ):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def masked_accuracy(
  output: torch.Tensor,
  target: torch.Tensor,
  mask: torch.Tensor
) -> float:
  """
  Compute accuracy of softmax output with mask applied over values.

  Adapted from pointer-networks-pytorch by ast0414:
    https://github.com/ast0414/pointer-networks-pytorch
  """

  with torch.no_grad():
    masked_output = torch.masked_select(output, mask)
    masked_target = torch.masked_select(target, mask)
    accuracy = masked_output.eq(masked_target).float().mean()
    return accuracy

def calculate_hull_overlap(data, length, pointer_argmaxs):
  """
  Compute the percent overlap between the predicted and true convex hulls.
  """

  points = data[2:length, :2]
  pred_hull_idxs = pointer_argmaxs[pointer_argmaxs > 1] - 2
  true_hull_idxs = ConvexHull(points).vertices.tolist()
  if len(pred_hull_idxs) >= 3 and len(true_hull_idxs) >= 3:
    shape_pred = geometry.Polygon(points[pred_hull_idxs].tolist())
    shape_true = geometry.Polygon(points[true_hull_idxs].tolist())
    if shape_pred.is_valid and shape_true.is_valid:
      area = shape_true.intersection(shape_pred).area
      percent_area = area / max(shape_pred.area, shape_true.area)
    else:
      percent_area = 0.0
  else:
    percent_area = 0.0
  return percent_area