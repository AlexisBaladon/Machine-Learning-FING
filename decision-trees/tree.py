import pandas as pd
class Tree:
    def __init__(self, label = None, value = None, filter_predicate = None, description = None, children = []):
        self.label = label
        self.value = value
        self.filter_predicate = filter_predicate
        self.description = description
        self.children = children

    def add_child(self, tree):
        self.children.append(tree)
    
    def is_leaf(self):
        return self.children == []

    def predict(self, row):
        for child in self.children:
            if (child.filter_predicate(row[self.label])):
                return child.predict(row)
        return self.value

class TreeWrapper:
    def __init__(self, tree, medians, most_common):
        self.tree = tree
        self.fill = medians | most_common
        self.value = 0

    def predict(self, row):
        for att, val in row.iteritems():
            if pd.isna(val):
                self.value += 1
                row[att] = self.fill[att]
        return self.tree.predict(row)