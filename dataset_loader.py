from PreProcessing import PreProcessing
import numpy as np
import xml.etree.ElementTree as ET
from gensim import models
from torch.utils.data import dataset



''' Load multilingual word embeddings '''
word_em_nl = models.KeyedVectors.load_word2vec_format('embeddings/wiki.multi.nl.bin', binary=True)
word_em_en = models.KeyedVectors.load_word2vec_format('embeddings/wiki.multi.en.bin', binary=True)
word_em_fr = models.KeyedVectors.load_word2vec_format('embeddings/wiki.multi.fr.bin', binary=True)
word_em_es = models.KeyedVectors.load_word2vec_format('embeddings/wiki.multi.es.bin', binary=True)



def pair_wise_add(x, y):
    assert len(x) == len(y)
    return [int(x[i] + y[i]) for i in range(len(x))]


def pair_wise_se(x, y):
    assert len(x) == len(y)
    for i in range(len(x)):
        if int(x[i]) > int(y[i]):
            return False
    return True


class SimpleDataset:
    def __init__(self, train_file, test_file, validation_percentage, language, redundant=False):
        self.category_label_num = {
            'RESTAURANT#GENERAL': 0,
            'SERVICE#GENERAL': 1,
            'FOOD#QUALITY': 2,
            'FOOD#STYLE_OPTIONS': 3,
            'DRINKS#STYLE_OPTIONS': 4,
            'DRINKS#PRICES': 5,
            'RESTAURANT#PRICES': 6,
            'RESTAURANT#MISCELLANEOUS': 7,
            'AMBIENCE#GENERAL': 8,
            'FOOD#PRICES': 9,
            'LOCATION#GENERAL': 10,
            'DRINKS#QUALITY': 11,
        }
        self.category_num_label = {
            0: 'RESTAURANT#GENERAL',
            1: 'SERVICE#GENERAL',
            2: 'FOOD#QUALITY',
            3: 'FOOD#STYLE_OPTIONS',
            4: 'DRINKS#STYLE_OPTIONS',
            5: 'DRINKS#PRICES',
            6: 'RESTAURANT#PRICES',
            7: 'RESTAURANT#MISCELLANEOUS',
            8: 'AMBIENCE#GENERAL',
            9: 'FOOD#PRICES',
            10: 'LOCATION#GENERAL',
            11: 'DRINKS#QUALITY',
        }
        self.entity = [
            'RESTAURANT',
            'SERVICE',
            'FOOD',
            'DRINKS',
            'AMBIENCE',
            'LOCATION',
        ]
        self.attrib = [
            'GENERAL',
            'QUALITY',
            'STYLE_OPTIONS',
            'MISCELLANEOUS',
            'PRICES',
        ]
        self.entity_label_num = {
            'RESTAURANT': 0,
            'FOOD': 1,
            'DRINKS': 2,
            'AMBIENCE': 3,
            'SERVICE': 4,
            'LOCATION': 5,
        }
        self.attrib_label_num = {
            'GENERAL': 0,
            'QUALITY': 1,
            'STYLE_OPTIONS': 2,
            'PRICES': 3,
            'MISCELLANEOUS': 4,
        }
        self.category_num_entity = {
            0: 'RESTAURANT',
            1: 'SERVICE',
            2: 'FOOD',
            3: 'FOOD',
            4: 'DRINKS',
            5: 'DRINKS',
            6: 'RESTAURANT',
            7: 'RESTAURANT',
            8: 'AMBIENCE',
            9: 'FOOD',
            10: 'LOCATION',
            11: 'DRINKS',
        }
        self.category_num_attrib = {
            0: 'GENERAL',
            1: 'GENERAL',
            2: 'QUALITY',
            3: 'STYLE_OPTIONS',
            4: 'STYLE_OPTIONS',
            5: 'PRICES',
            6: 'PRICES',
            7: 'MISCELLANEOUS',
            8: 'GENERAL',
            9: 'PRICES',
            10: 'GENERAL',
            11: 'QUALITY',
        }
        self.language = language
        self.redundant = redundant
        self.validation_percentage = validation_percentage
        self.extract_data(train_file, test_file)

    def extract_data(self, train_file, test_file):
        tree = ET.parse(train_file)
        root = tree.getroot()
        train_sentences = root.findall('**/sentence')
        tree = ET.parse(test_file)
        root = tree.getroot()
        test_sentences = root.findall('**/sentence')
        self.train_sentence_with_all_labels = {}
        self.processed_train_sentences = self.process_data(train_sentences)
        self.processed_test_sentences = self.process_data(test_sentences)
        self.train_data, self.categories, self.valid_data = self.get_inputs(self.processed_train_sentences,
                                                                            train_sentences,
                                                                            is_train=True)
        self.test_data = self.get_inputs(self.processed_test_sentences,
                                         test_sentences)
        self.number_of_categories = len(self.categories)

    def process_data(self, unprocessed_data):
        unprocessed_sentences = []
        for sentence in unprocessed_data:
            text = sentence[0].text
            if text == None:
                text = ''
            if '$' in text:
                text = text.replace('$', ' price ')
            text = text.lower()
            unprocessed_sentences.append(text)

        preprocessor = PreProcessing(unprocessed_sentences, language=self.language)
        preprocessor.Remove_Punctuation()
        processed_sentences = preprocessor.Remove_StopWords()

        return processed_sentences

    def get_inputs(self, processed_sentences, unprocessed_data, is_train=False):
        num_of_train_sentences = len(processed_sentences)
        processed_data = []
        categories = []
        for i in range(len(processed_sentences)):
            processed_sentences[i] = processed_sentences[i].split()
        num_of_data_per_cat = [0] * len(self.category_label_num.keys())
        valid_data = []
        valid_size = 0
        train_data = []
        for i in range(len(unprocessed_data)):
            sentence = processed_sentences[i]
            sentence_categories = []
            sentence_attrib = unprocessed_data[i].attrib
            try:
                if sentence_attrib['OutOfScope'] == 'TRUE':
                    continue
            except KeyError:
                pass
            if len(unprocessed_data[i]) > 1:
                if is_train:
                    if len(unprocessed_data[i][1]) == 0:
                        continue
                    if valid_size < self.validation_percentage * num_of_train_sentences:
                        add_valid_data = True
                        valid_size += 1
                    else:
                        add_valid_data = False
                    if not self.redundant:
                        labels = len(self.category_label_num.keys()) * [0]
                    for opinions in unprocessed_data[i][1]:
                        dict = opinions.attrib
                        if self.redundant:
                            labels = len(self.category_label_num.keys()) * [0]
                        if str(dict['category']) not in categories:
                            categories.append(str(dict['category']))

                        labels[self.category_label_num[str(dict['category'])]] = 1
                        sentence_categories.append(dict['category'])
                        num_of_data_per_cat[self.category_label_num[dict['category']]] += 1
                        if self.redundant is True:
                            if [sentence, labels] not in processed_data:
                                processed_data.append([sentence, labels])
                    if self.redundant is not True:
                        if [sentence, labels] not in processed_data:
                            processed_data.append([sentence, labels])
                else:
                    test_sentence_categories = []
                    for opinions in unprocessed_data[i][1]:
                        dict = opinions.attrib
                        if self.category_label_num[dict['category']] not in test_sentence_categories:
                            test_sentence_categories.append(self.category_label_num[dict['category']])
                    processed_data.append([sentence, test_sentence_categories])

        if is_train:
            num_of_valid_data_per_cat = [int(num_of_data_per_cat[i] * self.validation_percentage) for i in
                                         range(len(num_of_data_per_cat))]
            current_num_of_valid_data_per_cat = [0] * len(self.category_label_num.keys())
            for item in processed_data:
                sentence = item[0]
                label = item[1]
                temp = pair_wise_add(label, current_num_of_valid_data_per_cat)
                if pair_wise_se(temp, num_of_valid_data_per_cat) is True:
                    valid_data.append([sentence, label])
                    current_num_of_valid_data_per_cat = temp
                else:
                    train_data.append([sentence, label])
            return train_data, categories, valid_data
        else:
            return processed_data



class Concat(dataset.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.sentence_length = datasets[0].get_sentence_length()
        self.num_of_classes = datasets[0].get_num_of_classes()
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length

    def get_sentence_length(self):
        return self.sentence_length

    def get_num_of_classes(self):
        return self.num_of_classes


class DataLoader(dataset.Dataset):
    def __init__(self, data='train', simple_dataset=None, language='english'):
        assert simple_dataset is not None
        self.simple_dataset = simple_dataset
        if data == 'train':
            self.data = self.simple_dataset.train_data
        elif data == 'test':
            self.data = self.simple_dataset.test_data
        else:
            self.data = self.simple_dataset.valid_data
        self.sentence_length = 70
        self.data_type = data
        self.language = language

    def __len__(self):
        return len(self.data)

    def get_num_of_classes(self):
        return self.simple_dataset.number_of_categories

    def get_sentence_length(self):
        return self.sentence_length

    def __getitem__(self, item):
        sentence = []
        for word in self.data[item][0]:
            word_rep = []
            try:
                if self.language == 'english':
                    word_emb = word_em_en[word]
                elif self.language == 'french':
                    word_emb = word_em_fr[word]
                elif self.language == 'dutch':
                    word_emb = word_em_nl[word]
                elif self.language == 'spanish':
                    word_emb = word_em_es[word]
            except KeyError:
                continue
            word_rep += list(word_emb)
            sentence.append(np.array(word_rep))
        while len(sentence) < self.sentence_length:
            sentence.append(np.array(np.zeros(300), dtype='float32'))
        sentence = np.array(sentence, dtype='float32')
        label = np.array(self.data[item][1])
        entity_label = [0.0] * len(self.simple_dataset.entity)
        attrib_label = [0.0] * len(self.simple_dataset.attrib)
        for i, item in enumerate(label):
            if item == 1:
                entity_label[self.simple_dataset.entity_label_num[self.simple_dataset.category_num_entity[i]]] = 1.0
                attrib_label[self.simple_dataset.attrib_label_num[self.simple_dataset.category_num_attrib[i]]] = 1.0
        entity_label = np.array(entity_label)
        attrib_label = np.array(attrib_label)
        return sentence, np.array([label]), np.array([entity_label]), np.array([attrib_label])


def singledataset_loader(validation_percentage, language, redundant=False):
    if language == 'english':
        dataset = SimpleDataset('data/ABSA16_Restaurants_Train_SB1_v2.xml',
                                'data/EN_REST_SB1_TEST.xml.gold', validation_percentage=validation_percentage,
                                language=language, redundant=redundant)
    elif language == 'french':
        dataset = SimpleDataset('data/restaurants_french_train.xml',
                                'data/FR_REST_SB1_TEST.xml.gold', validation_percentage=validation_percentage,
                                language=language, redundant=redundant)
    elif language == 'spanish':
        dataset = SimpleDataset('data/SemEval-2016ABSA Restaurants-Spanish_Train_Subtask1.xml',
                                'data/SP_REST_SB1_TEST.xml.gold', validation_percentage=validation_percentage,
                                language=language, redundant=redundant)
    elif language == 'dutch':
        dataset = SimpleDataset('data/restaurants_dutch_train.xml',
                                'data/DU_REST_SB1_TEST.xml.gold', validation_percentage=validation_percentage,
                                language=language, redundant=redundant)
    train_loader = DataLoader(data='train', simple_dataset=dataset, language=language)
    validation_loader = DataLoader(data='valid', simple_dataset=dataset, language=language)
    test_loader = DataLoader(data='test', simple_dataset=dataset, language=language)
    return train_loader, validation_loader, test_loader


def multidataset_loader(validation_percentange, languages, redundant=False):
    train_loaders, validation_loaders, test_loaders = [], [], []
    for language in languages:
        train_loader, validation_loader, test_loader = singledataset_loader(validation_percentange, language, redundant=redundant)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        validation_loaders.append(validation_loader)
    return Concat(train_loaders), Concat(validation_loaders), Concat(test_loaders)

