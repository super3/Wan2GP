# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""What the Speedups hooks expect of this WanGP checkout, read from its source (no import, no GPU): the names they replace in
wgp.py, the lines and locals of generate_media they look for, and the mmgp and ContinuedVideo functions they call. A WanGP update
that moves them fails here; the plugin itself then falls back to WanGP's own behavior.

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


def test_generate_media_has_the_locals_the_save_reads():
    function, _ = _generate_media()
    assigned = {arg.arg for arg in function.args.args + function.args.kwonlyargs}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
    optional = {"total_windows", "video_file", "is_image", "audio_only", "any_soundtrack_processor", "audio_source", "replace_voice_method",
                "replace_voice_sample", "return_flashvsr_continue_cache"}
    assert set(hooks._REQUIRED_SAVE_LOCALS) | optional <= assigned
    assert {"state", "model_type", "activated_loras", "loras_multipliers"} <= assigned  # what the hooks read of a call and of queued tasks


def test_generate_media_has_the_lines_the_hooks_look_for():
    _, lines = _generate_media()
    starts = lambda text: [index for index, line in enumerate(lines) if line.startswith(text)]  # noqa: E731
    with_sound, without = starts("save_path_tmp = continued_video.append("), starts("continued_video_path = continued_video.append(")
    assert len(with_sound) == 1 and len(without) == 1
    assert any(line.startswith("shutil.copyfile(continued_video_path, video_path)") for line in lines[without[0]:without[0] + 5])
    assert any(line.startswith("combine_and_concatenate_video_with_audio_tracks(") for line in lines[with_sound[0]:])
    saves = [line for line in lines if line.startswith("save_video(")]
    assert any("save_file=save_path_tmp" in line for line in saves) and any("save_file=video_path" in line for line in saves)
    parks = [index for index, line in enumerate(lines) if "offloadobj.park()" in line]
    assert any(lines[index + 1] == "offloadobj.unload_all()" for index in parks)  # the unload after a generation
    record = starts("record_file_metadata(video_path")
    assert record and 'send_cmd("output")' in lines[record[0]:]  # the event that submits a background save follows its gallery entry


def test_the_mux_takes_the_video_second_and_the_new_tracks():
    function = _function(_tree(ROOT / "shared" / "utils" / "audio_video.py"), "combine_and_concatenate_video_with_audio_tracks")
    names = [arg.arg for arg in function.args.args]
    assert names[1] == "video_path" and "new_audio_tracks" in names


def test_continued_video_and_mmgp_have_what_the_hooks_call():
    tree = _tree(ROOT / "shared" / "utils" / "video_continuation.py")
    continued = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "ContinuedVideo")
    source = ast.unparse(continued)
    assert "def append(self, frames)" in source and all(f"self.{name}" in source for name in ("video_source", "file", "encode"))
    offload = _tree(ROOT / "mmgp" / "offload.py")
    functions = {node.name: node for node in offload.body if isinstance(node, ast.FunctionDef)}
    assert {"profile", "load_loras_into_model", "activate_loras", "unload_loras_from_model", "sync_models_loras"} <= set(functions)
    assert {"check_only", "pinnedLora", "maxReservedLoras"} <= {arg.arg for arg in functions["load_loras_into_model"].args.args}
    methods = {node.name for node in ast.walk(offload) if isinstance(node, ast.FunctionDef)}
    assert {"unload_all", "refresh_loras", "park"} <= methods
