import numpy as np
import math
from config import *
from network import nmr_inversion_model_cnn
from makedir import makedir
from train import predict_inverse, training_inverse


makedir()
model_path = training_inverse(nmr_inversion_model_cnn, random_seed)
predict_inverse(model_path, random_seed)

