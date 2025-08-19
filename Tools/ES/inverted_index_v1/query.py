# -*- coding: utf-8 -*-
"""Boolean query parser with phrase support and shunting-yard algorithm.

This module parses queries including:
- Terms (processed by Analyzer).
- Phrase queries in double quotes (e.g., "deep learning").
- Boolean operators: AND, OR, NOT.
- Parentheses for precedence.

Output is an RPN (Reverse Polish Notation) list of tokens for easy evaluation.

Author: Your Name
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Union

# 操作符优先级：NOT > AND > OR
_PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}
_ASSOC = {"NOT": "right", "AND": "left", "OR": "left"}


@dataclass
class TermToken:
    """A token representing a processed term."""
    term: str


@dataclass
class PhraseToken:
    """A token representing a phrase (list of processed terms)."""
    terms: List[str]


@dataclass
class OpToken:
    """Boolean operator token."""
    op: str  # AND, OR, NOT


Token = Union[TermToken, PhraseToken, OpToken, str]  # parentheses as '(' or ')'


_PHRASE_RE = re.compile(r'"([^"]+)"')


def tokenize_raw(query: str) -> List[str]:
    """Tokenize raw query string into rough tokens (keep quotes for phrases)."""
    # 中文注释：先提取带引号短语，再按空白和括号拆分
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
    """Parse a textual boolean/phrase query into RPN tokens.

    Args:
        query: Input query string.
        analyze_fn: A callable(text) -> List[str] which returns processed terms.

    Returns:
        RPN token list.
    """
    # 中文注释：将原始词和短语通过 analyzer 处理成标准 term
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
            stack.pop()  # pop '('
        elif tok.upper() in _PRECEDENCE:
            op = OpToken(op=tok.upper())
            # 处理一元 NOT 的结合性（右结合）
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

