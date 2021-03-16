"""
Builds a character-level text-prediction tree based on a corpus that is passed to it.
Each node except for the root is a
"""
from nltk.tokenize import word_tokenize
from typing import Dict, Any, Optional, Tuple


class CharTree:
    def __init__(self, data, children=None, count=0):
        if children is None:
            children = dict()
        self.data: Any = data
        self.children: Dict[Optional[str], CharTree] = children
        self.count: int = count

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
        child: CharTree = self.children[word[0]]
        child.count += 1
        if len(word) > 1:
            child.__build_branch(word[1:])
        elif len(word) == 1:
            child.children[None] = self.children.get(None, TreeLeaf())
            child.children[None].count += 1

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
            if child.data is not None:
                accumulate += f"'{child.data}': CharTree(data='{child.data}', count={child.count} children={{{child.__repr_children()}}}) "
            else:
                accumulate += f"'None': TreeLeaf()"
            accumulate += ","
        return accumulate[:-1]  # дропаем последнюю запятую - она лишняя

    def __repr__(self):
        root = f"CharTree(data='{self.data}', children={{"
        children = self.__repr_children()
        return root + children + "})"

    def printout(self, level=0):
        for key in self.children.keys():
            child = self.children[key]
            print('\t' * level, f"{key}-{child.count}")
            child.printout(level + 1)

    def __get_matching_subtree(self, typed_text) -> Optional['CharTree']:
        try:
            child = self.children[typed_text[0]]
            if len(typed_text) > 1:
                match = child.__get_matching_subtree(typed_text[1:])
            else:
                match = child
            return match
        except KeyError:
            return None

    def __get_most_probable_continuation(self) -> str:
        (most_probable_letter, child) = max(self.children.items(), key=lambda kv: kv[1].count)
        if most_probable_letter is None:
            return ''
        else:
            return most_probable_letter + child.__get_most_probable_continuation()

    def predict(self, typed_text):
        match = self.__get_matching_subtree(typed_text)
        if match is None:
            return None
        continuation: str = match.__get_most_probable_continuation()
        return typed_text + continuation


class TreeLeaf(CharTree):

    def __init__(self):
        super(TreeLeaf, self).__init__(data=None)
