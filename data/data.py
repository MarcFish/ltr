import numpy as np
from itertools import permutations


class Data:
    def __init__(self, filepath="./min.txt"):
        self.q_dict = dict()
        with open(filepath, 'r', encoding='utf-8') as f:
            row_list = f.readlines()
            for l in row_list:
                l = l.strip().split()
                qid = int(l[1].split(':')[1])
                relevance = int(l[0])
                feature = list()
                for i in l[2:48]:
                    feature.append(float(i.split(':')[1]))
                feature = np.asarray(feature)
                self.q_dict.setdefault(qid, dict())
                self.q_dict[qid].setdefault('rel', list())
                self.q_dict[qid].setdefault('feature', list())
                self.q_dict[qid]['rel'].append(relevance)
                self.q_dict[qid]['feature'].append(feature)
