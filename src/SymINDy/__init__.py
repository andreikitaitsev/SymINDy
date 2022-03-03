from .util import fixup_module_metadata

__version__ = "0.1"
__author__ = "Matteo Manzi, Andrei Kitaitsev"

fixup_module_metadata(__name__, globals())
del fixup_module_metadata
