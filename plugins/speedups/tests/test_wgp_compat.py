# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""What the Speedups hook expects of this WanGP checkout, read from its source (no import, no GPU): the names it replaces in
wgp.py, the lines and locals of generate_media it looks for, and the mmgp functions it calls. A WanGP update that moves them fails
here; the plugin itself then falls back to WanGP's own behavior.

    python -m pytest plugins/speedups/tests -q
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "plugins"))

from speedups import hooks  # noqa: E402

WGP = ROOT / "wgp.py"


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def _generate_media():
    source = WGP.read_text(encoding="utf-8")
    function = _function(ast.parse(source), "generate_media")
    lines = [line.strip() for line in source.splitlines()[function.lineno - 1:function.end_lineno]]
    return function, lines


def test_wgp_has_the_names_the_hooks_replace():
    tree = _tree(WGP)
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    assert set(hooks._WRAPPED) | {"offload", "gc", "get_gen_info"} <= names


def test_generate_media_has_the_locals_the_hook_reads():
    function, _ = _generate_media()
    assigned = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
    assert {"state", "model_type"} <= assigned  # what the hook reads of a call
    assert {"window_no", "total_windows", "repeat_no", "total_generation"} <= assigned  # whether the model runs again in the task


def test_generate_media_has_the_lines_the_hook_looks_for():
    _, lines = _generate_media()
    parks = [index for index, line in enumerate(lines) if "offloadobj.park()" in line]
    assert any(lines[index + 1] == "offloadobj.unload_all()" and lines[index + 2] == "gc.collect()" for index in parks)  # the unload after a generation
    assert sum(line == "gc.collect()" for line in lines) >= 2  # and the collection before a generation


def test_mmgp_has_what_the_hook_calls():
    offload = _tree(ROOT / "mmgp" / "offload.py")
    functions = {node.name: node for node in offload.body if isinstance(node, ast.FunctionDef)}
    assert {"profile", "all", "load_loras_into_model", "sync_models_loras"} <= set(functions)
    assert "check_only" in {arg.arg for arg in functions["load_loras_into_model"].args.args}
    assert functions["profile"].args.kwarg is not None and "coTenantsMap" in {arg.arg for arg in functions["all"].args.args}  # profile hands it to all()
    methods = {node.name for node in ast.walk(offload) if isinstance(node, ast.FunctionDef)}
    assert {"unload_all", "refresh_loras", "park"} <= methods
