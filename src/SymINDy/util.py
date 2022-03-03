from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, AnyStr, Optional, Union

SPHINX_BUILD = os.environ.get("SPHINX_BUILD")

__all__ = (
    "camel_to_kebab_case",
    "camel_to_snake_case",
    "fixup_module_metadata",
    "temporary_filename",
)


def fixup_module_metadata(module_name: str, namespace: dict[str, Any]) -> None:
    """Update module name for objects in `namespace` to `module_name`."""
    # This function is modified from the Trio project and retains its original
    # license:
    #
    # Copyright © 2017 Nathaniel J. Smith
    # SPDX-License-Identifier: (Apache-2.0 OR MIT)
    seen_ids = set()

    # When we descend into types, we add the information to this list to fix
    # after we've fixed all the top-level items. This helps the case where one
    # class is aliased inside another. For example, to guarantee that
    # Schema.OPTIONS_CLASS doesn't get used instead of Options:
    #
    # class Options:
    #     pass
    #
    # class Schema:
    #     OPTIONS_CLASS = Options
    lazy_fix = []

    def fix_one(qualname: str, name: str, obj: Any) -> None:
        # avoid infinite recursion (relevant when using
        # typing.Generic, for example)
        if id(obj) in seen_ids:
            return
        seen_ids.add(id(obj))

        mod = getattr(obj, "__module__", None)
        if mod is not None and mod.startswith("colafw."):

            # If we're running under sphinx, we can add a property for the
            # documentation to use when linking to the source on GitLab.
            if SPHINX_BUILD:
                try:
                    # fixup_module_metadata might be called on something that
                    # is re-exported at a higher level, so take the original
                    # __real_module__ value.
                    real_module = obj.__real_module__
                except AttributeError:
                    real_module = obj.__module__
                obj.__real_module__ = real_module

            obj.__module__ = module_name
            # Modules, unlike everything else in Python, put fully-qualitied
            # names into their __name__ attribute. We check for '.' to avoid
            # rewriting these.
            if hasattr(obj, "__name__") and "." not in obj.__name__:
                obj.__name__ = name
                obj.__qualname__ = qualname
            if isinstance(obj, type):
                for attr_name, attr_value in obj.__dict__.items():
                    lazy_fix.append((objname, attr_name, attr_value))

    for objname, obj in namespace.items():
        if not objname.startswith("_"):  # ignore private attributes
            fix_one(objname, objname, obj)

    for objname, attr_name, attr_value in lazy_fix:
        fix_one(objname + "." + attr_name, attr_name, attr_value)


_camelcase_re = re.compile(r"([A-Z]+)(?=[a-z0-9])")


def camel_to_snake_case(name: str) -> str:
    """Convert camel case or pascal case to snake case.

    .. code-block:: python

        >>> camel_to_snake_case('ExampleWorkflow')
        'example_workflow'
    """
    # Taken from Flask-SQLAlchemy and retains its original license:
    #
    # Copyright 2010 Pallets
    # SPDX-License-Identifier: BSD-3-Clause
    def _join(match: re.Match) -> str:
        word = match.group()

        if len(word) > 1:
            return f"_{word[:-1]}_{word[-1]}".lower()

        return "_" + word.lower()

    return _camelcase_re.sub(_join, name).lstrip("_")


def camel_to_kebab_case(name: str) -> str:
    """Convert camel case or pascal case to kebab case.

    .. code-block:: python

        >>> camel_to_kebab_case('ExampleWorkflow')
        'example-workflow'
    """
    return camel_to_snake_case(name).replace("_", "-")


@contextmanager
def temporary_filename(
    suffix: Optional[AnyStr] = None,
    prefix: Optional[AnyStr] = None,
    dir: Optional[Union[AnyStr, os.PathLike[AnyStr]]] = None,
) -> Iterator[str]:
    """Like :func:`~tempfile.NamedTemporaryFile` but returns a filename instead
    of a file object. Unlike :func:`~tempfile.NamedTemporaryFile`, this function
    may only be used as a context manager.
    """
    # We use this with delete=False to get a unique name. By closing it, we
    # ensure that the file may be opened on Windows.
    with tempfile.NamedTemporaryFile(
        suffix=suffix, prefix=prefix, dir=dir, delete=False
    ) as f:
        name = f.name

    try:
        yield name
    finally:
        os.unlink(name)
