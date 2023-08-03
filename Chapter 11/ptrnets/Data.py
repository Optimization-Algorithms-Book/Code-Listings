from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def generate_random_points(
  n: int,
  sort_method: str = 'lex'
) -> np.ndarray:
  """
  Randomly sample n sorted uniformly distributed 2D points from [0.0, 1.0).

  Args:
    n: Number of x,y points to generate.
    sort_method: Method to sort points. The following methods are supported:
      lex: Sort in ascending lexicographic order starting from y.
      mag: Sort from least to greatest magnitude (or distance from origin).
  Outputs:
    Shape (n, 2) sorted numpy array of x,y coordinates.
  """

  points = np.random.random(size=[n, 2])
  if sort_method == 'lex':
    points = points[np.lexsort(([points[..., ax] for ax in range(points.shape[-1])]))]
  elif sort_method == 'mag':
    points = points[np.argsort(np.mean(points, axis=-1))]
  else:
    raise ValueError(f'{sort_method} is not a valid option for sort_method.')
  return points

def display_points(points: np.ndarray) -> None:
  """
  Display a set of 2D points on a scatterplot.

  Args:
    points: x,y coordinate points.
  """

  y_offset = 0.025
  plt.scatter(points[:, 0], points[:, 1])
  for i, point in enumerate(points):
    plt.text(point[0], point[1] + y_offset, str(i))
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim(0., 1.)
  plt.ylim(0., 1.)
  plt.title(f'N: {len(points)}')
  plt.grid(True)

def display_points_with_hull(
  points: np.ndarray,
  hull: list
) -> None:
  """
  Display a set of 2D points with its convex hull.

  Args:
    points: x,y coordinate points.
    hull: List of indices indicating the convex hull of points.
  """

  display_points(points)
  for i in range(len(hull) - 1):
    p0 = hull[i]
    p1 = hull[i + 1]
    x = points[[p0, p1], 0]
    y = points[[p0, p1], 1]
    plt.plot(x, y, 'g')
    plt.arrow(x[0], y[0], (x[1] - x[0]) / 2., (y[1] - y[0]) / 2.,
              shape='full', lw=0, length_includes_head=True, head_width=.025,
              color='green')
  x = points[[p1, hull[0]], 0]
  y = points[[p1, hull[0]], 1]
  plt.arrow(x[0], y[0], (x[1] - x[0]) / 2., (y[1] - y[0]) / 2.,
            shape='full', lw=0, length_includes_head=True, head_width=.025,
            color='green')
  plt.plot(x, y, 'g')
  plt.grid(True)

def cyclic_permute(
  l: list,
  idx: int
) -> list:
  """
  Permute a list such that l[idx] becomes l[0] while preserving order.

  Args:
    l: List to permute.
    idx: Index to the element in l that should appear at position 0.
  Outputs:
    Cyclically permuted list of length len(l).
  """

  return l[idx:] + l[:idx]

def Disp_results(train_loss,train_accuracy,val_loss,val_accuracy,n_epochs):
    idx_best_train_loss = np.argmin(train_loss.history)
    best_train_loss = train_loss.history[idx_best_train_loss]
    idx_best_train_accuracy = np.argmax(train_accuracy.history)
    best_train_accuracy = train_accuracy.history[idx_best_train_accuracy]
    idx_best_val_loss = np.argmin(val_loss.history)
    best_val_loss = val_loss.history[idx_best_val_loss]
    idx_best_val_accuracy = np.argmax(val_accuracy.history)
    best_val_accuracy = val_accuracy.history[idx_best_val_accuracy]
    print('Best Scores:')
    print(f'train_loss: {best_train_loss:.4f} (ep: {idx_best_train_loss})')
    print(f'train_accuracy {best_train_accuracy:3.2%} (ep: {idx_best_train_accuracy})')
    print(f'val_loss: {best_val_loss:.4f} (ep: {idx_best_val_loss})')
    print(f'val_accuracy: {best_val_accuracy:3.2%} (ep: {idx_best_val_accuracy})')

    x_epochs = list(range(n_epochs))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(x_epochs, train_loss.history, 'b')
    ax[0].plot(x_epochs, val_loss.history, 'r')
    _ = ax[0].set_title('Train vs. Val Loss')
    ax[1].plot(x_epochs, train_accuracy.history, 'b', label='Train')
    ax[1].plot(x_epochs, val_accuracy.history, 'r', label='Val')
    _ = ax[1].set_title('Train vs. Val Accuracy')
    ax[1].legend()

TOKENS = {
'<eos>': 0
}

class Scatter2DDataset(Dataset):
  def __init__(
    self,
    n_rows: int,
    min_samples: int,
    max_samples: int
  ):
    self.min_samples = min_samples
    self.max_samples = max_samples
    self.points = []
    self.targets = []
    self.TOKENS = {'<eos>':0}
    n_points = np.random.randint(low=min_samples, high=max_samples + 1, size=n_rows)
    for c in n_points:
      points = generate_random_points(c)
      targets = ConvexHull(points).vertices.tolist()
      targets = cyclic_permute(targets, np.argmin(targets))
      self.points.append(points)
      self.targets.append(targets)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    points, targets, length = self.pad_item(self.points[idx], self.targets[idx])
    return points, targets, length

  def __len__(self) -> int:
    return len(self.points)

  def pad_item(
    self,
    points: list,
    targets: list
  ) -> Tuple[torch.tensor, torch.Tensor]:
    n_tokens = len(self.TOKENS)

    # points_padded = np.zeros((self.max_samples + n_tokens, 3 + n_tokens),
    points_padded = np.zeros((self.max_samples + n_tokens, 2 + n_tokens),
      dtype=np.float32)
    targets_padded = np.ones((self.max_samples), dtype=np.int64) \
      * self.TOKENS['<eos>']

    # points_padded[TOKENS['<sos>'], 2] = 1.0
    # points_padded[TOKENS['<eos>'], 3] = 1.0
    points_padded[self.TOKENS['<eos>'], 2] = 1.0
    points_padded[n_tokens:n_tokens + len(points), :2] = points
    # points_padded[n_tokens + len(points):, 4] = 1.0
    targets_padded[:len(targets)] = np.array([t + n_tokens for t in targets])

    points_padded = torch.tensor(points_padded, dtype=torch.float32)
    targets_padded = torch.tensor(targets_padded, dtype=torch.int64)
    length = torch.tensor(len(points) + 2, dtype=torch.int64)
    return points_padded, targets_padded, length