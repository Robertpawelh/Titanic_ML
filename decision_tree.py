import math
import pandas as pd
from collections import Counter, defaultdict
import numpy as np


class Leaf:
    def __init__(self, value):
        self.value = value


class Split:
    def __init__(self, attribute, subtrees, default_value):
        self.attribute = attribute
        self.subtrees = subtrees
        self.default_value = default_value


class DecisionTree:
    def entropy(self, class_probabilities):
        sum = 0
        for p in class_probabilities:
            if p>0:
                sum += -p * math.log(p, 2)
        return sum

    def class_probabilities(self, labels):
        return [count / len(labels) for count in Counter(labels).values()]

    def partition_entropy(self, subsets):
        total_count = sum(len(subset) for subset in subsets)
        return sum(self.entropy(self.class_probabilities(subset)) * len(subset) / total_count
                   for subset in subsets)

    def make_partitions(self, inputs, attribute: str):
        partitions = defaultdict(list)
        for id, row in inputs.iterrows():
            key = row[attribute]
            partitions[key].append(row)
        return partitions

    def partition_entropy_by(self, x_train, attribute: str, label_attribute):
        partitions = self.make_partitions(x_train, attribute)
        labels = [[x[label_attribute] for x in partition]
                  for partition in partitions.values()]
        return self.partition_entropy(labels)

    def build_tree_id3(self, inputs, split_attributes, target_attribute):
        label_counts = Counter(row[target_attribute]
                               for id, row in inputs.iterrows())
        most_common_label = label_counts.most_common(1)[0][0]

        if len(label_counts) == 1 or not split_attributes:
            return Leaf(most_common_label)

        def split_entropy(attribute):
            return self.partition_entropy_by(inputs, attribute, target_attribute)

        best_attribute = min(split_attributes, key=split_entropy)

        partitions = self.make_partitions(inputs, best_attribute)
        new_attributes = [a for a in split_attributes if a != best_attribute]

        subtrees = {attribute_value: self.build_tree_id3(pd.DataFrame(subset),
                                                         new_attributes,
                                                         target_attribute)
                    for attribute_value, subset in partitions.items()}

        return Split(best_attribute, subtrees, most_common_label)

    def fit(self, x_train, label_name):
        columns = []
        for col in x_train.columns:
            if col != label_name:
                columns.append(col)
        tree = self.build_tree_id3(x_train, columns, label_name)
        self.tree = tree

    def classify(self, tree, input):
        if isinstance(tree, Leaf):
            return tree.value
        subtree_key = input[tree.attribute]

        if subtree_key not in tree.subtrees:
            return tree.default_value

        subtree = tree.subtrees[subtree_key]
        return self.classify(subtree, input)

    def predict(self, x):
        return [self.classify(self.tree, row) for id, row in x.iterrows()]
