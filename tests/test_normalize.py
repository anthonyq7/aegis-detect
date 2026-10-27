import ast

from aegis.normalize import strip_docstrings


def _has_docstring(code: str) -> bool:
    tree = ast.parse(code)
    nodes = [tree] + [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    return any(ast.get_docstring(n) is not None for n in nodes)


def test_module_function_and_class_docstrings_removed():
    code = (
        '"""Module doc."""\n'
        "import os\n\n"
        "class A:\n"
        '    """Class doc."""\n'
        "    def m(self):\n"
        '        """Method doc."""\n'
        "        return os.sep\n\n"
        "async def f():\n"
        "    '''Async doc.'''\n"
        "    return 1\n"
    )
    out = strip_docstrings(code)
    assert out is not None
    assert not _has_docstring(out)
    assert "doc" not in out.lower()
    assert "return os.sep" in out


def test_docstring_only_body_becomes_pass():
    out = strip_docstrings('def f():\n    """Only a docstring."""\n')
    assert out == "def f():\n    pass"
    ast.parse(out)


def test_consecutive_leading_strings_all_removed():
    code = '"""doc"""\n"second bare string"\n\ndef f():\n    """a"""\n    "b"\n    return 1\n'
    out = strip_docstrings(code)
    assert not _has_docstring(out)
    assert "second" not in out and '"b"' not in out


def test_comments_removed():
    out = strip_docstrings("x = 1  # trailing\n# full line\ny = 2\n")
    assert "#" not in out
    assert out == "x = 1\ny = 2"


def test_non_docstring_strings_preserved():
    code = (
        "def f(flag):\n"
        "    x = 1\n"
        '    "mid-body string is not a docstring"\n'
        "    if flag:\n"
        '        "string inside if is not a docstring"\n'
        "    return x\n"
    )
    out = strip_docstrings(code)
    assert "mid-body string" in out
    assert "inside if" in out


def test_unparsable_returns_none():
    assert strip_docstrings("print 'python 2'") is None
    assert strip_docstrings("```python\ndef f(): pass\n```") is None


def test_idempotent():
    code = '"""doc"""\ndef f(a, b):\n    """add"""\n    return a + b  # sum\n'
    once = strip_docstrings(code)
    assert strip_docstrings(once) == once
