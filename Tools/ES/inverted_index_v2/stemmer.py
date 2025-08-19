# -*- coding: utf-8 -*-
"""Pure-Python Porter Stemmer (dependency-free)."""

from __future__ import annotations

class PorterStemmer:
    """English Porter Stemmer (Porter 1980)."""

    def __init__(self) -> None:
        self.vowels = set("aeiou")

    def _is_consonant(self, word: str, i: int) -> bool:
        ch = word[i]
        if ch in self.vowels:
            return False
        if ch == "y":
            if i == 0:
                return True
            return not self._is_consonant(word, i - 1)
        return True

    def _m(self, stem: str) -> int:
        if not stem:
            return 0
        m = 0
        prev_is_c = self._is_consonant(stem, 0)
        for i in range(1, len(stem)):
            is_c = self._is_consonant(stem, i)
            if prev_is_c and not is_c:
                m += 1
            prev_is_c = is_c
        return m

    def _vowel_in_stem(self, stem: str) -> bool:
        for i in range(len(stem)):
            if not self._is_consonant(stem, i):
                return True
        return False

    def _doublec(self, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and self._is_consonant(word, len(word) - 1)

    def _cvc(self, word: str) -> bool:
        if len(word) < 3:
            return False
        c1 = self._is_consonant(word, -1)
        v = not self._is_consonant(word, -2)
        c0 = self._is_consonant(word, -3)
        if c0 and v and c1:
            ch = word[-1]
            return ch not in ("w", "x", "y")
        return False

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word
        w = word
        if w.endswith("sses"):
            w = w[:-2]
        elif w.endswith("ies"):
            w = w[:-2]
        elif w.endswith("ss"):
            pass
        elif w.endswith("s"):
            w = w[:-1]
        flag = False
        if w.endswith("eed"):
            stem = w[:-3]
            if self._m(stem) > 0:
                w = w[:-1]
        elif w.endswith("ed"):
            stem = w[:-2]
            if self._vowel_in_stem(stem):
                w = stem
                flag = True
        elif w.endswith("ing"):
            stem = w[:-4] if w.endswith("ying") else w[:-3]
            if self._vowel_in_stem(stem):
                w = stem
                flag = True
        if flag:
            if w.endswith(("at", "bl", "iz")):
                w = w + "e"
            elif self._doublec(w):
                if w[-1] not in ("l", "s", "z"):
                    w = w[:-1]
            elif self._m(w) == 1 and self._cvc(w):
                w = w + "e"
        if w.endswith("y") and self._vowel_in_stem(w[:-1]):
            w = w[:-1] + "i"
        step2_map = {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance", "izer": "ize",
            "abli": "able", "alli": "al", "entli": "ent", "eli": "e", "ousli": "ous",
            "ization": "ize", "ation": "ate", "ator": "ate", "alism": "al", "iveness": "ive",
            "fulness": "ful", "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
        }
        for k, v in step2_map.items():
            if w.endswith(k):
                stem = w[: -len(k)]
                if self._m(stem) > 0:
                    w = stem + v
                break
        step3_map = {
            "icate": "ic", "ative": "", "alize": "al", "iciti": "ic", "ical": "ic", "ful": "", "ness": "",
        }
        for k, v in step3_map.items():
            if w.endswith(k):
                stem = w[: -len(k)]
                if self._m(stem) > 0:
                    w = stem + v
                break
        step4_suffixes = (
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
            "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        )
        for suf in step4_suffixes:
            if w.endswith(suf):
                stem = w[: -len(suf)]
                if self._m(stem) > 1:
                    if suf == "ion":
                        if stem.endswith(("s", "t")):
                            w = stem
                    else:
                        w = stem
                break
        if self._m(w[:-1]) > 1 and w.endswith("e"):
            w = w[:-1]
        elif self._m(w[:-1]) == 1 and not self._cvc(w[:-1]) and w.endswith("e"):
            w = w[:-1]
        if self._m(w) > 1 and self._doublec(w) and w.endswith("l"):
            w = w[:-1]
        return w

