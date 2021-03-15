"""
Builds a character-level text-prediction tree based on a corpus that is passed to it.
Each node except for the root is a
"""
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import deque
from chukchi.Singleton.Singleton import Singleton


class CharTree:
    def __init__(self, data, children=None):
        if children is None:
            children = dict()
        self.data = data
        self.children = children

    # def __sliding_window__(self, word: str, size=2):
    #     """
    #     takes a string of length S, size N and returns S - (N - 1) tuples.
    #     These tuples are all N-long character sequences.
    #     Function returns the whole word if S<=N.
    #     For our thing we need a window of size 2.
    #     """
    #     if len(word) > size:
    #         return tuple(word)
    #     it = iter(word)
    #     win = deque([next(it) for _ in range(size)], maxlen=size)
    #     yield tuple(win)
    #     for e in it:
    #         win.append(e)
    #         yield tuple(win)

    def __hash__(self):
        return self.data.__hash__()

    def __build_branch(self, word):
        self.children[word[0]] = self.children.get(word[0], CharTree(data=word[0]))
        child = self.children[word[0]]
        if len(word) > 1:
            child.__build_branch(word[1:])
        elif len(word) == 1:
            child.children[None] = TreeLeaf()

    @staticmethod
    def build_tree(corpus: str):
        tree = CharTree(data="")
        for word in word_tokenize(corpus):
            tree.__build_branch(word)
            # windows = self.__sliding_window__(word)
            # for w in windows:
            #     char_1 = w[0]
            #     char_2 = w[0]
            #     current[char_1] = current.get(char_1, dict())
            #     current = current[char_1]
            #     current[char_2] = current.get(char_1, dict())
        return tree

    def __repr_children(self):
        accumulate: str = ""
        for key in self.children.keys():
            child: CharTree = self.children[key]
            child_value = child.data
            if child_value is not None:
                accumulate += f"'{child_value}': CharTree(data='{child_value}', children={{{child.__repr_children()}}})"
            else:
                accumulate += f"'None': TreeLeaf()"
            accumulate += ","
        return accumulate[:-1]  # дропаем последнюю запятую - она лишняя

    def __repr__(self):
        root = f"CharTree(data='{self.data}', children={{"
        children = self.__repr_children()
        return root + children + "})"


class TreeLeaf(CharTree, metaclass=Singleton):

    def __init__(self):
        super(TreeLeaf, self).__init__(data=None)
