"""Docstring removal shared by the data pipeline and the CLI.

Both training data (data/standardize_code.py) and inference (aegis/predictor.py)
must see code in the same form, so this lives in the installed package.
"""

import ast
from typing import Optional

_DOCSTRING_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def strip_docstrings(code: str) -> Optional[str]:
    """Return `code` re-rendered with every module, class and function docstring removed.

    Comments are dropped as a side effect of the AST round-trip. Bare strings that are
    not docstrings (mid-body, or inside if/for/try blocks) are preserved. Returns None
    when the code cannot be parsed.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None

    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        # Loop: a bare string that follows a docstring would otherwise become the new docstring.
        while node.body and _is_docstring(node.body[0]):
            node.body.pop(0)
        if not node.body:
            node.body.append(ast.Pass())

    ast.fix_missing_locations(tree)
    return ast.unparse(tree)
