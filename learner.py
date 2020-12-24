import torch
import torch.nn as nn
import torch.optim as optim
from rent_dms import Strategy
from rentgym import RentGym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

