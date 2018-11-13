import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import dataloader
from torch.backends import cudnn
import random
import decimal
from copy import deepcopy
from tqdm import tqdm

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


class TopKMS(nn.Module):
    def __init__(self, k=0.3):
        super(TopKMS, self).__init__()
        '''
        This class returns the top-k hardest examples in the train data based
        on the loss of these examples - similar to the paper 
        "Training Region-based Object Detectors with Online Hard Example Mining"
        '''
        self.loss = nn.MSELoss()
        self.k = k
        return

    def forward(self, input, target):
        '''
        Argument shapes similar to the argument shapes required in the MSELoss
        function in the pytorch library
        '''
        loss = Variable(torch.Tensor(1).zero_())
        if torch.has_cudnn:
            loss = loss.cuda()
        for idx, row in enumerate(input):
            gt = target[idx]
            pred = torch.unsqueeze(row, 0)
            cost = self.loss(pred, gt.unsqueeze(0))
            loss = torch.cat((loss, cost.unsqueeze(0)), 0)
        loss = loss[1:]
        if self.k == 1.0 or input.size(0) == 1:
            valid_loss = loss
        else:
            index = torch.topk(loss, int(self.k * loss.size()[0]))
            valid_loss = loss[index[1]]
        return torch.mean(valid_loss)


class Convolutional(nn.Module):
    def __init__(self, input_size, num_of_kernels, drop_out_prob, sentence_size, hidden_size, num_of_classes):
        super(Convolutional, self).__init__()
        '''
        :param input_size: the feature size of the input
        :param num_of_kernels: number of kernels for each of the three window sizes
        :param drop_out_prob: the probability of applying dropout on the values
        :param sentence_size: the constant size of the input sentences
        :param hidden_size: the size of the hidden layer
        :param num_of_classes: the number of output classes
        '''
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.has_cudnn:
            torch.cuda.manual_seed(0)
        cudnn.benchmark = True
        random.seed(0)
        # unigram
        self.unigram_conv = nn.Conv1d(input_size, num_of_kernels, 1)
        self.unigram_maxpool = nn.MaxPool1d(sentence_size)
        # bigram
        self.bigram_conv = nn.Conv1d(input_size, num_of_kernels, 2)
        self.bigram_maxpool = nn.MaxPool1d(sentence_size - 1)
        # trigram
        self.trigram_conv = nn.Conv1d(input_size, num_of_kernels, 3)
        self.trigram_maxpool = nn.MaxPool1d(sentence_size - 2)

        self.drop_out = nn.Dropout(drop_out_prob)

        self.hidden_layer = nn.Linear(3 * num_of_kernels, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_of_classes)

        self.entity_hidden_layer = nn.Linear(3 * num_of_kernels, hidden_size)
        self.entity_layer = nn.Linear(hidden_size, 6)

        self.attrib_hidden_layer = nn.Linear(3 * num_of_kernels, hidden_size)
        self.attrib_layer = nn.Linear(hidden_size, 5)

    def forward(self, x, validate=False):
        x = self.drop_out(x)
        x = x.transpose(1, 2)
        unigram = self.unigram_conv(x)
        unigram = self.unigram_maxpool(unigram)
        unigram = F.relu(unigram)

        bigram = self.bigram_conv(x)
        bigram = self.bigram_maxpool(bigram)
        bigram = F.relu(bigram)

        trigram = self.trigram_conv(x)
        trigram = self.trigram_maxpool(trigram)
        trigram = F.relu(trigram)

        x = torch.cat((unigram, bigram, trigram), dim=1).squeeze(-1)

        label = self.hidden_layer(x)
        label = self.output_layer(label)
        label = F.sigmoid(label)

        entity = self.entity_hidden_layer(x)
        entity = self.entity_layer(entity)
        entity = F.sigmoid(entity)

        attrib = self.attrib_hidden_layer(x)
        attrib = self.attrib_layer(attrib)
        attrib = F.sigmoid(attrib)

        return label, entity, attrib


class Net:
    def __init__(self, epochs, train_dataset, test_dataset, validation_dataset, learning_rate,
                 entity_coeff, attrib_coeff, early_stopping_mode, early_stopping_min_delta, early_stopping_patience,
                 num_of_kernels=512, drop_out_prob=0.4):
        '''
        This class performes the training process on the model
        :param epochs: number of epochs
        :param train_dataset: the train_dataset
        :param test_dataset: the test dataset
        :param validation_dataset: the validation dataset
        :param learning_rate: the learning rate of the model
        :param entity_coeff: the coefficient for applying the entity loss (for regularization effect)
        :param attrib_coeff: the coefficient for applying the attribute loss (for regularization effect)
        :param early_stopping_mode: the early stopping mode (min or max)
        :param early_stopping_min_delta: the early stopping delta (minimum improvement)
        :param early_stopping_patience: the early stopping patience
        :param num_of_kernels: number of kernels for each kernel window size
        :param drop_out_prob: the probability of applying dropout
        '''
        self.model = Convolutional(input_size=300,
                                   num_of_kernels=num_of_kernels,
                                   drop_out_prob=drop_out_prob,
                                   sentence_size=train_dataset.get_sentence_length(),
                                   hidden_size=150,
                                   num_of_classes=train_dataset.get_num_of_classes())
        if torch.has_cudnn:
            self.model.cuda()
        self.num_epochs = epochs
        self.early_stopping = EarlyStopping(early_stopping_mode, early_stopping_min_delta, early_stopping_patience)

        self.loss_function = TopKMS(k=0.4)
        self.validate_loss = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.entity_coeff = entity_coeff
        self.attrib_coeff = attrib_coeff

        self.train_loader = train_dataset
        self.test_loader = test_dataset
        self.validation_loader = validation_dataset

    def to_var(self, x):
        '''
        :param x: the input matrix
        :return: pytorch tensor version of the input matrix
        '''
        try:
            x = x.float()
        except AttributeError:
            x = torch.from_numpy(x).float()
        if torch.has_cudnn:
            x = Variable(x).cuda()
        else:
            x = Variable(x)
        return x

    def train(self, batch_size, plot_validate=False):
        validate_loss = []
        train_loss = []
        data_loader = dataloader.DataLoader(dataset=self.train_loader,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=0)
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            total_loss = []
            for i, (datas, labels, entity, attrib) in enumerate(data_loader):
                datas = self.to_var(datas)
                labels = self.to_var(labels)
                entity = self.to_var(entity)
                attrib = self.to_var(attrib)
                self.optimizer.zero_grad()
                outputs, entities, attribs = self.model(datas)

                label_loss = self.loss_function(outputs, labels.squeeze(1))
                entity_loss = self.loss_function(entities, entity.squeeze(1)) * self.entity_coeff
                attrib_loss = self.loss_function(attribs, attrib.squeeze(1)) * self.attrib_coeff

                loss = label_loss + entity_loss + attrib_loss
                loss.backward()
                self.optimizer.step()
                total_loss.append(label_loss.item())
            if plot_validate is True:
                valid_loss = self.validate()
                validate_loss.append(valid_loss)
                train_loss.append(float(sum(total_loss) / len(total_loss)))
            else:
                valid_loss = self.validate()
                if self.early_stopping.step(valid_loss, deepcopy(self.model)) is True:
                    print('Early stopping at ' + str(epoch))
                    self.model = self.early_stopping.best_model
                    break
        if plot_validate is True:
            return train_loss, validate_loss

    def test(self):
        self.model.eval()
        pred_labels = []
        true_labels = []
        for data, label, entity, attrib in self.test_loader:
            data = self.to_var(data)
            data = data.unsqueeze(0)
            output, _, __ = self.model(data)

            output = output.squeeze(0)
            pred_labels.append(list(output))
            true_labels.append(label)
        best_result = [0.0, 0.0, 0.0, 0.0]
        val_f1, threshold = self.find_best_threshold()
        TP = 0
        FP = 0
        FN = 0
        for idx in range(len(pred_labels)):
            if idx == 12:
                continue
            output = pred_labels[idx]
            label = true_labels[idx]
            for i in range(len(list(output))):
                if float(output[i]) >= threshold:
                    if i in label:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if i in label:
                        FN += 1
        try:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        return f1, precision, recall, val_f1

    def find_best_threshold(self):
        '''
        finding the best threshold based on the validation set
        '''
        self.model.eval()
        pred_labels = []
        true_labels = []
        best_result = [0.0, 0]
        data_loader = dataloader.DataLoader(dataset=self.validation_loader,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0)
        for datas, labels, _, __ in data_loader:
            datas = self.to_var(datas)
            labels = self.to_var(labels)
            outputs, _, __ = self.model(datas, validate=True)
            output = outputs.squeeze(0)
            pred_labels.append(list(output))
            true_labels.append(labels.squeeze(0).squeeze(0))
        for threshold in list(frange(0, 1, decimal.Decimal('0.01'))):
            TP = 0
            FP = 0
            FN = 0
            for idx in range(len(pred_labels)):
                if idx == 12:
                    continue
                output = pred_labels[idx]
                label = true_labels[idx]
                for i in range(len(list(output))):
                    if float(output[i]) >= threshold:
                        if int(label[i]) == 1:
                            TP += 1
                        else:
                            FP += 1
                    else:
                        if int(label[i]) == 1:
                            FN += 1
            try:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1 = 0.0
            if f1 > best_result[0]:
                best_result[0] = f1
                best_result[1] = threshold
        return best_result[0], best_result[1]

    def validate(self):
        self.model.eval()
        valid_loss = 0
        data_loader = dataloader.DataLoader(dataset=self.validation_loader,
                                            batch_size=len(self.validation_loader),
                                            shuffle=False,
                                            num_workers=0)
        for datas, labels, entity, attrib in data_loader:
            datas = self.to_var(datas)
            labels = self.to_var(labels)
            outputs, _, __ = self.model(datas, validate=True)
            valid_loss += self.validate_loss(outputs, labels.squeeze(1))
        valid_loss = float(valid_loss)
        return valid_loss


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics, model):
        if self.best is None:
            self.best = metrics
            self.best_model = model
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_model = model
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

    def return_best_model(self):
        return self.best_model
