# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""A stand-in for wgp.py, for the CPU tests of the Speedups hooks: generate_media keeps, in the same order, the lines and locals of
WanGP v20.10's generate_media that the hooks look for (LoRA load, the unload after a generation, the save of a plain video, the
"output" event, the LoRA unload at the end of the task). Files are text files: a "video" holds its frames' name, a mux joins its
inputs with "+"."""
# ruff: noqa  (its save lines keep wgp.py's own layout)

import gc
import os
import shutil
import time

offload = None  # a stand-in mmgp.offload, set by the tests
offloadobj = None
LOG = []
MODEL = None
SETTINGS = {"encode_s": 0.0, "fail_encode": False}


def get_gen_info(state):
    return state.setdefault("gen", {})


def write(path, text):
    with open(path, "w") as file:
        file.write(text)


def save_video(tensor=None, save_file=None, fps=None, nrow=1, normalize=True, value_range=(-1, 1), codec_type=None, container="mp4"):
    time.sleep(SETTINGS["encode_s"])
    if SETTINGS["fail_encode"]:
        raise RuntimeError("encoder failed")
    write(save_file, f"video:{tensor}")
    LOG.append(("encode", os.path.basename(save_file)))
    return save_file


class ContinuedVideo:
    """video_continuation.ContinuedVideo, fresh: its first window opens its own work file."""

    def __init__(self, work_dir, container="mp4"):
        self.video_source, self.file, self.work_dir, self.container = None, None, work_dir, container
        self.encode = lambda frames, path: save_video(tensor=frames, save_file=path, container=container)

    def append(self, frames):
        segment = self.encode(frames, os.path.join(self.work_dir, f"tmp_continue1.{self.container}"))
        self.file = segment
        return segment


def combine_and_concatenate_video_with_audio_tracks(save_path_tmp, video_path, source_audio_tracks, new_audio_tracks, source_audio_duration, audio_sampling_rate, **kwargs):
    with open(video_path) as video:
        joined = "+".join([video.read(), *(open(track).read() for track in new_audio_tracks)])
    write(save_path_tmp, joined)
    LOG.append(("mux", os.path.basename(save_path_tmp)))


def cleanup_temp_audio_files(paths):
    for path in paths:
        os.remove(path)


def record_file_metadata(video_path, configs, is_image, audio_only, gen, **kwargs):
    with open(video_path) as video:
        assert video.read().startswith("video:"), f"{video_path} is not written"
    gen.setdefault("file_list", []).append(video_path)
    LOG.append(("record", os.path.basename(video_path)))


def store_api_output_artifact(gen, client_id, video_path, *args, **kwargs):
    LOG.append(("artifact", os.path.basename(video_path)))


def generate_media(task, send_cmd, state=None, model_type=None, activated_loras=(), loras_multipliers="", out_dir=None, sound=True,
                   continued=True, hdr=False, windows=1, crash=False):
    gen = get_gen_info(state)
    model_def = {}
    trans_lora = MODEL
    loras_selected = list(activated_loras)
    if len(loras_selected) > 0:
        offload.load_loras_into_model(trans_lora, loras_selected, [1.0] * len(loras_selected), activate_all_loras=True, pinnedLora=True, maxReservedLoras=-1)
    gc.collect()
    repeat_no, total_generation, window_no, total_windows = 1, 1, windows, windows  # the last window of the only sample
    LOG.append(("generate", model_type, getattr(trans_lora, "_loras_active_adapters", None)))
    if crash:
        try:
            raise RuntimeError("generation failed")
        except RuntimeError:
            offloadobj.unload_all()
            offload.unload_loras_from_model(trans_lora)
            send_cmd("error", "generation failed")
            return False
    if not (model_def.get("no_model_swap", False) and offloadobj.park()):
        offloadobj.unload_all()
    gc.collect()

    container = "mp4"
    sample_is_hdr, sliding_window = hdr, windows > 1
    control_audio_tracks, source_audio_tracks, side_files = [], [], {}
    video_file, is_image, audio_only = None, False, False
    any_soundtrack_processor, audio_source, replace_voice_method, replace_voice_sample, return_flashvsr_continue_cache = False, None, "", None, False
    continued_video = ContinuedVideo(out_dir, container) if continued else None
    output_video_frames = f"{task['id']}"
    file_name = f"clip_{task['id']}.{container}"
    video_path = os.path.join(out_dir, file_name)
    if sound:
        video_path = os.path.join(out_dir, file_name)
        save_path_tmp = video_path.rsplit('.', 1)[0] + f"_tmp.{container}"
        if continued_video is not None:  # the new frames are encoded alone and follow the video made so far, kept in a file of its own
            save_path_tmp = continued_video.append(output_video_frames)
        else:
            save_video( tensor=output_video_frames, save_file=save_path_tmp, fps=24, nrow=1, normalize=True, value_range=(-1, 1), codec_type = None, container=container)
        output_new_audio_temp_filepath = os.path.join(out_dir, f"tmp{task['id']}.wav")
        write(output_new_audio_temp_filepath, f"sound:{task['id']}")
        combine_and_concatenate_video_with_audio_tracks(video_path, save_path_tmp, source_audio_tracks, [output_new_audio_temp_filepath], 0, 48000)
        if continued_video is None: os.remove(save_path_tmp)
        if output_new_audio_temp_filepath is not None: os.remove(output_new_audio_temp_filepath)
        cleanup_temp_audio_files([])
    else:
        if continued_video is not None:
            continued_video_path = continued_video.append(output_video_frames)
            shutil.copyfile(continued_video_path, video_path)
        else:
            save_video( tensor=output_video_frames, save_file=video_path, fps=24, nrow=1, normalize=True, value_range=(-1, 1),  codec_type= None, container= container)
    record_file_metadata(video_path, {}, False, False, gen)
    store_api_output_artifact(gen, None, video_path)
    send_cmd("output")
    if continued_video is not None and continued_video.file is not None:
        os.remove(continued_video.file)  # remove_temp_filenames: the continued video's work file
    offload.unload_loras_from_model(trans_lora)
    return True
