"""
Builds a character-level text prediction tree based on a corpus that is passed to it.
HOW TO:
1. import tree
2. run `build_tree` method against a corpus (as a big string). It returns a prediction tree.
3. predict continuation for an already typed text via `predict` method. Prediction is word-level, i.e. only
   parts of words should be passed to `predict` method, not phrases.
4. OPTIONAL: save model by simply printing the tree. You can then simply pass this chunk of text to python interpreter
   and it will create the tree object you need, implying you have imported tree and TreeLeaf classes, of course.
"""
import string
from typing import Dict, Any, Optional
import re

from nltk.tokenize import word_tokenize


class CharTree:
    def __init__(self, data, children=None, count=0, step=1):
        if children is None:
            children = dict()
        self.data: Any = data
        self.children: Dict[Optional[str], CharTree] = children
        self.count: int = count
        self.step: int = step

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
        """
        not used, keep it just in case
        """
        return self.data.__hash__()

    def __repr__(self):
        """
        you can just copy the output of your typical `print` function and then copypaste it to recreate a tree.
        But a big tree should be pickled with pickle.dump(). This func is more for lulz and testing
        """
        root = f"{self.__class__.__name__}(data='{self.data}', children={{"
        children = self.__repr_children()
        return root + children + "})"

    @staticmethod
    def _format_input_data(data):
        _data = data.lower()
        _data = re.sub(r"['`ʼ‘’\"“]", 'ʼ', _data)
        _data = re.sub(r'[åòêíóýîüûã]', '', _data)  # TODO: fix corpus
        _data = re.sub(r'ль', 'ԓь', _data)
        _data = re.sub(r'Ль', 'Ԓь', _data)
        _data = re.sub(r"кʼ", 'ӄ', _data)
        _data = re.sub(r"Kʼ", 'Ӄ', _data)
        _data = re.sub(r"нʼ", 'ӈ', _data)
        _data = re.sub(r"Нʼ", 'Ӈ', _data)
        # replace latin with cyrillic AND drop punctuation
        _data = _data.translate(str.maketrans('eyopac', 'еуорас', string.punctuation))
        _data = re.sub(r"\d", '', _data)
        return _data

    @staticmethod
    def build_tree(corpus: str, step=1):
        """
        The main thing to build a CharTree. Pass a corpus as a string and get a chartree as a return value
        """
        tree = CharTree(data="", step=step)
        _corpus = corpus
        _corpus = CharTree._format_input_data(_corpus)
        for word in word_tokenize(_corpus):
            if word == "гымнин":
                a = 1
            tree.__build_branch(word)
        return tree

    def __build_branch(self, word: str):
        """
        Function used for recursive creation of chartree branches
        """
        length_of_word = len(word)
        if length_of_word >= self.step:
            seq = word[:self.step]
        else:
            seq = word[:length_of_word]
        self.children[seq] = self.children.get(seq, CharTree(data=seq, step=self.step))
        child: CharTree = self.children[seq]
        child.count += 1
        if length_of_word > self.step:
            child.__build_branch(word[self.step:])
        elif length_of_word <= self.step:
            child.children[None] = child.children.get(None, TreeLeaf())
            child.children[None].count += 1

    def __repr_children(self):
        """
        We need a separate function that creates `print` representation of children in recursive manner,
        because unlike the root object these are wrapped in curly braces and are values in a dict
        """
        accumulate: str = ""
        for key in self.children.keys():
            child: CharTree = self.children[key]
            if child.data is not None:
                accumulate += f"'{child.data}': {self.__class__.__name__}(data='{child.data}', count={child.count} children={{{child.__repr_children()}}}) "
            else:
                accumulate += f"'None': TreeLeaf()"
            accumulate += ","
        return accumulate[:-1]  # dropping the last comma - it's redundant and even harmful

    def _get_matching_subtree(self, typed_text) -> Optional['CharTree']:
        """
        self-explainatory name. Given typed text, we try to find the already determined node in the tree.
        """
        try:
            length = len(typed_text)
            if length >= self.step:
                child = self.children[typed_text[:self.step]]
                match = child._get_matching_subtree(typed_text[self.step:])
            else:
                match = self.children[typed_text[:length]]
            return match
        except (KeyError, IndexError):
            return self

    def __get_most_probable_continuation(self) -> str:
        """
        Go down the tree through the nodes with higher count (most probablee ones), collecting the chars they represent
        along the way
        """
        (most_probable_letter, child) = max(self.children.items(), key=lambda kv: kv[1].count)
        if most_probable_letter is None:
            return ''
        else:
            return most_probable_letter + child.__get_most_probable_continuation()

    def printout(self, level=0):
        """
        A simple visualisation of any CharTree object.
        Left number represents the level (from 0 to N)
        The number next to a letter represents how often it was encountered in this position
        Example:
        0| l-5
        1|   a-5
        2|     m-5
        3|       p-3
        3|       a-2
        This makes "lama" LESS probable than "lamp", given input "lam"
        """
        for key in self.children.keys():
            child = self.children[key]
            print(level, '|', '\t' * level, f"{key}-{child.count}", sep='')
            child.printout(level + 1)

    def predict(self, typed_text: str):
        __lowered_text = CharTree._format_input_data(typed_text.lower())
        match = self._get_matching_subtree(__lowered_text)
        if match is None:
            return None
        continuation: str = match.__get_most_probable_continuation()
        return typed_text + continuation


class TreeLeaf(CharTree):

    def __init__(self):
        super(TreeLeaf, self).__init__(data=None)
