"""Guards that the pure-python parts of the project stay importable on their own.

``shared/utils/__init__.py`` used to eagerly re-export the scheduler classes from
``fm_solvers``, which imports torch and diffusers. That made every module in the
package -- including ones needing nothing but ``re`` -- unreachable without the full
CUDA stack, and forced the test suite to load submodules through a stand-in parent
package.

The package now re-exports those names lazily (PEP 562 ``__getattr__``). These tests
pin that down: it is easy to reintroduce a top-level ``from .fm_solvers import ...``
and not notice, because on a developer machine with torch installed everything keeps
working. It would only show up as a slow start-up and a broken CI run.

Each check runs in a subprocess with the heavy modules poisoned, so it fails loudly
rather than passing by accident on a machine where torch happens to be installed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Importing any of these means the lazy re-export has regressed.
HEAVY_MODULES = ("torch", "diffusers", "numpy", "gradio", "transformers")

# Modules that must stay reachable without the runtime stack. Each is stdlib-only.
PURE_MODULES = [
    "shared.utils.prompt_parser",
    "shared.utils.loras_mutipliers",
    "shared.utils.frame_scheduler",
    "shared.utils.filename_formatter",
    "shared.utils.audio_metadata",
    "shared.utils.gguf_mapping",
    "shared.resolutions",
    "shared.match_archi",
    "shared.lora_mapper",
    "shared.tools.sha256_verify",
]


def _run_isolated(body: str, repo_root) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter where the heavy modules cannot be imported."""

    program = textwrap.dedent(
        f"""
        import sys

        class _Blocked(Exception):
            pass

        class _Blocker:
            def find_module(self, name, path=None):
                return self.find_spec(name, path)

            def find_spec(self, name, path=None, target=None):
                root = name.split(".")[0]
                if root in {HEAVY_MODULES!r}:
                    raise _Blocked(
                        "importing " + name + " -- the lazy re-export has regressed"
                    )
                return None

        sys.meta_path.insert(0, _Blocker())
        """
    ) + textwrap.dedent(body)

    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=120,
    )


@pytest.mark.parametrize("module_name", PURE_MODULES)
def test_pure_module_imports_without_the_runtime_stack(module_name, repo_root):
    result = _run_isolated(f"import {module_name}\n", repo_root)
    assert result.returncode == 0, (
        f"{module_name} could not be imported without the heavy runtime stack:\n"
        f"{result.stderr}"
    )


def test_shared_utils_package_itself_stays_light(repo_root):
    """`import shared.utils` must not drag in torch via the scheduler re-exports."""

    result = _run_isolated("import shared.utils\n", repo_root)
    assert result.returncode == 0, (
        "shared/utils/__init__.py pulled in a heavy dependency at import time -- the "
        f"lazy re-export has regressed:\n{result.stderr}"
    )


def test_scheduler_names_are_still_publicly_advertised(repo_root):
    """Laziness must not silently drop the package's public API."""

    result = _run_isolated(
        """
        import shared.utils as su

        expected = {
            "FlowDPMSolverMultistepScheduler",
            "FlowUniPCMultistepScheduler",
            "get_sampling_sigmas",
            "retrieve_timesteps",
        }
        assert expected <= set(su.__all__), su.__all__
        assert expected <= set(dir(su)), "dir() must advertise the lazy names too"
        """,
        repo_root,
    )
    assert result.returncode == 0, result.stderr


def test_unknown_attribute_still_raises_attribute_error(repo_root):
    """__getattr__ must not turn typos into ImportError or hangs."""

    result = _run_isolated(
        """
        import shared.utils as su

        try:
            su.definitely_not_a_real_name
        except AttributeError as exc:
            assert "definitely_not_a_real_name" in str(exc), exc
        else:
            raise AssertionError("expected AttributeError")
        """,
        repo_root,
    )
    assert result.returncode == 0, result.stderr


def test_lazy_access_is_what_pulls_the_heavy_dependency_in(repo_root):
    """The deferred import should happen on attribute access, not before.

    Touching a scheduler name is expected to fail here precisely because the blocker
    is installed -- that is the evidence the import was deferred to this point rather
    than done at package import time.
    """

    result = _run_isolated(
        """
        import shared.utils as su  # must succeed

        try:
            su.FlowUniPCMultistepScheduler
        except Exception as exc:
            assert type(exc).__name__ in ("_Blocked", "ModuleNotFoundError", "ImportError"), exc
        else:
            raise AssertionError("expected the deferred import to be attempted here")
        """,
        repo_root,
    )
    assert result.returncode == 0, result.stderr
