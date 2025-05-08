import random
import numpy as np
from dataclasses import dataclass
import torch

@dataclass
class datacfg:
    P: int
    num: int
    data_seed: int
    train_frac: float

def gen_train_test(train_cfg):#frac_train, num, seed=0
    train_frac=train_cfg.train_frac
    P=train_cfg.P
    seed=train_cfg.data_seed
    # Generate train and test split
    pairs = [(i, j, P) for i in range(P) for j in range(P)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(train_frac*len(pairs))
    train=pairs[:div]
    test=pairs[div:]
    train_labels = torch.tensor([(i+j)%P for i, j, _ in train])
    test_labels = torch.tensor([(i+j)%P for i, j, _ in test])
    return train, test, train_labels, test_labels


# Creates an array of Boolean indices according to whether each data point is in
# train or test
# Used to index into the big batch of all possible data
def get_is_train_test(train, train_cfg):
    P=train_cfg.P
    is_train = []
    is_test = []
    for x in range(P):
        for y in range(P):
            if (x, y, P) in train:
                is_train.append(True)
                is_test.append(False)
            else:
                is_train.append(False)
                is_test.append(True)
    return np.array(is_train), np.array(is_test)



