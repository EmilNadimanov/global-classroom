from typing import Optional

from nltk import word_tokenize

from chukchi.tree.CharTree import CharTree


class MorphTree(CharTree):

    @staticmethod
    def build_tree(corpus: str):
        """
        The main thing to build a MorphTree. Pass a corpus as a string and get a chartree as a return value
        """
        tree = MorphTree(data="")
        _corpus = CharTree._format_input_data(corpus)
        for word in word_tokenize(_corpus):
            tree._build_branch(word)
        return tree

    def _build_branch(self, word: str):
        if len(word) > 1:
            seq = word[:2]
        else:
            seq = word[:len(word)]
        self.children[seq] = self.children.get(seq, CharTree(data=seq))
        child: CharTree = self.children[seq]
        child.count += 1
        if len(word) > 1:
            child.__build_branch(word[1:])
        elif len(word) == 1:
            child.children[None] = self.children.get(None, TreeLeaf())
            child.children[None].count += 1

    def _get_matching_subtree(self, typed_text) -> Optional['MorphTree']:
        pass


