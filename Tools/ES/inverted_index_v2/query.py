# -*- coding: utf-8 -*-
"""Boolean query parser with phrase support (shunting-yard)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

_PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}
_ASSOC = {"NOT": "right", "AND": "left", "OR": "left"}

@dataclass
class TermToken:
    term: str

@dataclass
class PhraseToken:
    terms: List[str]

@dataclass
class OpToken:
    op: str  # AND, OR, NOT

Token = Union[TermToken, PhraseToken, OpToken, str]


def tokenize_raw(query: str) -> List[str]:
    parts: List[str] = []
    i = 0
    while i < len(query):
        if query[i] == '"':
            j = i + 1
            while j < len(query) and query[j] != '"':
                j += 1
            if j >= len(query):
                phrase = query[i + 1 :]
                i = len(query)
            else:
                phrase = query[i + 1 : j]
                i = j + 1
            parts.append(f'"{phrase}"')
        elif query[i] in ("(", ")"):
            parts.append(query[i])
            i += 1
        elif query[i].isspace():
            i += 1
        else:
            j = i
            while j < len(query) and (not query[j].isspace()) and query[j] not in ("(", ")"):
                j += 1
            parts.append(query[i:j])
            i = j
    return parts


def parse_query(query: str, analyze_fn) -> List[Token]:
    raw_tokens = tokenize_raw(query)
    output: List[Token] = []
    stack: List[OpToken | str] = []

    def push_term_terms(terms: List[str]) -> None:
        for t in terms:
            if t:
                output.append(TermToken(term=t))

    for tok in raw_tokens:
        if tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())  # type: ignore[arg-type]
            if not stack:
                raise ValueError("Unbalanced parentheses.")
            stack.pop()
        elif tok.upper() in _PRECEDENCE:
            op = OpToken(op=tok.upper())
            while stack and isinstance(stack[-1], OpToken):
                top: OpToken = stack[-1]  # type: ignore[assignment]
                if (
                    (_ASSOC[op.op] == "left" and _PRECEDENCE[op.op] <= _PRECEDENCE[top.op])
                    or (_ASSOC[op.op] == "right" and _PRECEDENCE[op.op] < _PRECEDENCE[top.op])
                ):
                    output.append(stack.pop())  # type: ignore[arg-type]
                else:
                    break
            stack.append(op)
        elif tok.startswith('"') and tok.endswith('"'):
            phrase_text = tok[1:-1]
            terms = analyze_fn(phrase_text)
            if not terms:
                continue
            output.append(PhraseToken(terms=terms))
        else:
            terms = analyze_fn(tok)
            if not terms:
                continue
            push_term_terms(terms)

    while stack:
        top = stack.pop()
        if top in ("(", ")"):
            raise ValueError("Unbalanced parentheses at end.")
        output.append(top)  # type: ignore[arg-type]

    return output

