"""Shared utility subpackage.

The scheduler classes below are re-exported lazily. Importing them eagerly meant that
touching *anything* in this package -- including pure-python helpers such as
``prompt_parser`` or ``filename_formatter``, which need nothing but the standard
library -- first pulled in ``fm_solvers``, and with it torch and diffusers.

Nothing in the application actually imports these names from the package (every caller
reaches for ``shared.utils.fm_solvers`` and friends directly), so the cost bought
nothing. PEP 562 module ``__getattr__`` keeps the names importable for any out-of-tree
plugin that does rely on them, while deferring the heavy import until one is used.
"""

import importlib

# Exported name -> the submodule that defines it.
_LAZY_EXPORTS = {
    "FlowDPMSolverMultistepScheduler": ".fm_solvers",
    "get_sampling_sigmas": ".fm_solvers",
    "retrieve_timesteps": ".fm_solvers",
    "FlowUniPCMultistepScheduler": ".fm_solvers_unipc",
}

# 'HuggingfaceTokenizer' used to be listed here but was never bound -- it lives in
# models.wan.modules.tokenizers and importing it here would invert the layering. Its
# presence made `from shared.utils import *` raise AttributeError, so it is left out.
__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value  # cache, so later lookups skip __getattr__ entirely
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
