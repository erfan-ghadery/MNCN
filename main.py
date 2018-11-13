from dataset_loader import *
from model import Net
import itertools
import torch
import numpy as np
import random
from torch.backends import cudnn

torch.manual_seed(0)
np.random.seed(0)
if torch.has_cudnn:
    torch.cuda.manual_seed(0)
cudnn.benchmark = True
random.seed(0)


def train_and_test(learning_rate, batch_size, train_languages, test_languages, entity_coeff,
                                            attrib_coeff, early_stopping_mode,
                                            early_stopping_min_delta, early_stopping_patience):
    train_loader, validation_loader, _ = multidataset_loader(0.1, languages=train_languages)
    _, __, test_loader = multidataset_loader(0.0, languages=test_languages)
    net = Net(epochs, train_loader, test_loader, validation_loader, learning_rate, entity_coeff, attrib_coeff,
                              early_stopping_mode, early_stopping_min_delta, early_stopping_patience)
    net.train(batch_size=batch_size, plot_validate=False)
    f1, precision, recall, val_f1 = net.test()
    return f1, precision, recall, val_f1

epochs = 100
learning_rate = 0.002
batch_size = 128
drop_out_prob = 0.4
early_stopping_mode = 'min'
early_stopping_min_delta = 0.0
early_stopping_patience = 10
alpha = 0.5 #entity_coefficient
betha = 0.8 #attribute_coefficient

#train_languages = ['english', 'french', 'dutch', 'spanish']
#test_languages = ['english', 'french', 'dutch', 'spanish']
train_languages = ['english']
test_languages = ['english']
f1_list = []
prec_list = []
recall_list = []
combinations = list(itertools.permutations(train_languages))
for comb in combinations:
    f1, precision, recall, _ = train_and_test(learning_rate, batch_size, comb,
                                                      test_languages, alpha, betha,
                                                      early_stopping_mode,
                                                      early_stopping_min_delta,
                                                      early_stopping_patience)
    f1_list.append(f1)
    prec_list.append(precision)
    recall_list.append(recall)



precision = np.mean(prec_list)
recall = np.mean(recall_list)
f1 = 2*(precision*recall)/(precision+recall)
print("F1:" + str(f1))
print("Precision: " + str(precision))
print("Recall: " + str(recall))
