# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU tests of embeddings.py and embed_server.py (no WanGP, no GPU):

    python -m pytest tests -q
"""

import base64
import hashlib
import importlib.util
import io
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


emb = load("h3_live_embeddings", "embeddings.py")  # the name live.py loads it under when it is on its own
server = load("h3_live_embed_server_under_test", "embed_server.py")
live = load("h3_live_under_test", "live.py")
HELPERS = server.simple_helpers()


def png(width=100, height=60, color=(200, 40, 10)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, "PNG")
    return buffer.getvalue()


def embedding(prompt="a prompt", tokens=12, kinds=("image", "image"), digest="ab12"):
    generator = torch.Generator().manual_seed(len(prompt))
    tags = torch.ones(tokens, dtype=torch.int64)
    tags[2:5] = 0
    return emb.PromptEmbedding(prompt, torch.randn(1, tokens, 5120, generator=generator).to(torch.bfloat16), tags, tuple(kinds), digest, "test-model", 0.25, 832, 480)


class EncPipe:
    """A pipeline's text encoding: 3 tokens per word, 5 per picture, values from the prompt."""

    def __init__(self):
        self.calls, self.device, self.dtype, self.fixed_prompt, self._interrupt = [], torch.device("cpu"), torch.bfloat16, None, False

    def _prompt_cache_key(self, prompt, presentation):
        digest = hashlib.sha256()
        for item in presentation:
            digest.update(item["type"].encode())
            digest.update(item["frames"].numpy().tobytes())
        return "minimax_h3", "bf16", prompt, digest.digest()

    def _encode_prompt(self, prompt, presentation):
        self.calls.append((prompt, len(presentation)))
        tokens = 3 * len(prompt.split()) + 5 * len(presentation)
        generator = torch.Generator().manual_seed(sum(prompt.encode()))
        return torch.randn(1, tokens, 5120, generator=generator).to(torch.bfloat16), torch.ones(tokens, dtype=torch.int64)


def test_safetensors_round_trip_is_exact():
    original = embedding('Say (S1) <d>[English] "Hi" — ça va?</d>\nnon_diegetic_music: N/A')
    data = emb.to_bytes(original)
    back = emb.from_bytes(data)
    assert torch.equal(back.prompt_embeds.view(torch.int16), original.prompt_embeds.view(torch.int16))
    assert torch.equal(back.text_token_tags, original.text_token_tags) and back.text_token_tags.dtype == torch.int64
    assert (back.prompt, back.presentation, back.presentation_digest, back.model, back.width, back.height) == (
        original.prompt, ("image", "image"), "ab12", "test-model", 832, 480)
    assert back.tokens == 12 and back.dim == 5120 and back.nbytes == 12 * 5120 * 2 + 12 * 8
    # the names of WanGP's fixed prompts: a saved embedding loads where those do
    from safetensors.torch import load as load_tensors
    assert set(load_tensors(data)) == {"prompt_embeds", "text_token_tags"}


def test_json_round_trip_is_exact():
    original = embedding("json prompt", tokens=7, kinds=())
    data = json.loads(json.dumps(emb.to_json(original)))
    assert data["tokens"] == 7 and data["dim"] == 5120 and data["dtype"] == "bfloat16" and data["presentation"] == []
    back = emb.from_json(data)
    assert torch.equal(back.prompt_embeds.view(torch.int16), original.prompt_embeds.view(torch.int16)) and back.prompt == "json prompt"


def test_malformed_embeddings_are_refused():
    good = embedding()
    bad = emb.PromptEmbedding("x", good.prompt_embeds, good.text_token_tags[:5])
    with pytest.raises(ValueError):
        emb.from_bytes(emb.to_bytes(bad))
    data = emb.to_json(good)
    data["format"] = "something-else/1"
    with pytest.raises(ValueError):
        emb.from_json(data)


def test_pictures_are_shown_as_the_pipeline_shows_a_start_image():
    presentation = emb.presentation_for([png(100, 60), png(832, 480, (0, 0, 0))], 832, 480, HELPERS)
    assert [item["type"] for item in presentation] == ["image", "image"]
    frames = presentation[0]["frames"]
    assert tuple(frames.shape) == (1, 480, 832, 3) and float(frames.min()) >= 0 and float(frames.max()) <= 1
    assert abs(float(frames[0, 240, 416, 0]) - 200 / 255) < 0.01  # the colour survives the resize
    assert float(presentation[1]["frames"].max()) == 0.0  # already the generation size: not resized


def test_the_real_pipeline_helpers_give_the_same_pictures():
    try:
        helpers = emb._pipeline_helpers()
    except Exception as error:  # noqa: BLE001 - WanGP's H3 pipeline needs its dependencies
        pytest.skip(f"models.minimax_h3.pipeline does not import here: {type(error).__name__}")
    ours = emb.presentation_for([png(100, 60)], 832, 480, HELPERS)[0]["frames"]
    theirs = emb.presentation_for([png(100, 60)], 832, 480, helpers)[0]["frames"]
    assert torch.equal(ours, theirs)


def test_encode_runs_the_pipeline_text_encoding_on_the_cpu():
    pipe = EncPipe()
    result = emb.encode(pipe, "one two three", [png(), png()], 832, 480, model="m", helpers=HELPERS)
    assert pipe.calls == [("one two three", 2)]
    assert result.tokens == 9 + 10 and result.presentation == ("image", "image") and result.prompt_embeds.device.type == "cpu"
    assert result.presentation_digest == emb.digest(pipe, "one two three", emb.presentation_for([png(), png()], 832, 480, HELPERS))
    assert (result.width, result.height, result.model) == (832, 480, "m")
    text_only = emb.encode(pipe, "one two three", helpers=HELPERS)
    assert text_only.presentation == () and text_only.tokens == 9 and (text_only.width, text_only.height) == (0, 0)


def test_supplied_embeddings_replace_the_text_encoder():
    class Pipe(EncPipe):
        pass

    emb.clear()
    emb.install(Pipe)
    try:
        pipe = Pipe()
        pictures = emb.presentation_for([png(), png()], 832, 480, HELPERS)
        made = emb.encode(pipe, "a b", [png(), png()], helpers=HELPERS)  # made by the encoder, not answered from what is supplied
        assert pipe.calls == [("a b", 2)]
        supplied = embedding("a b", tokens=4, digest=made.presentation_digest)
        assert emb.supply(supplied) == 1
        context, tags = pipe._encode_prompt("a b", pictures)
        assert torch.equal(context.view(torch.int16), supplied.prompt_embeds.view(torch.int16)) and torch.equal(tags, supplied.text_token_tags)
        assert len(pipe.calls) == 1 and emb.STATS["hits"] == 1
        pipe._encode_prompt("a b", [])  # text only: not the embedding made with two pictures
        pipe._encode_prompt("other", pictures)
        assert len(pipe.calls) == 3 and emb.STATS["misses"] == 2
        emb.encode(pipe, "a b", [png(), png()], helpers=HELPERS)  # encode() always runs the encoder
        assert len(pipe.calls) == 4
        # strict: the pictures must be the ones the embedding was made with
        emb.supply(embedding("a b", tokens=4, digest="not-these-pictures"), strict=True)
        pipe._encode_prompt("a b", pictures)
        assert len(pipe.calls) == 5 and emb.STATS["digest_mismatches"] == 1
        emb.supply(supplied)
        pipe._encode_prompt("a b", pictures)
        assert len(pipe.calls) == 5 and emb.STATS["hits"] == 2
        pipe.fixed_prompt = object()  # a fixed-prompt model keeps its own
        pipe._encode_prompt("a b", pictures)
        assert len(pipe.calls) == 6
    finally:
        emb.OPTIONS["strict"] = False
        emb.uninstall()
        emb.clear()
    assert not getattr(Pipe._encode_prompt, "_h3_live_embeddings", False)


def test_supply_dir_loads_every_file(tmp_path):
    emb.clear()
    try:
        for index in range(3):
            emb.save(embedding(f"prompt {index}"), tmp_path / f"{index}.safetensors")
        (tmp_path / "notes.txt").write_text("not an embedding")
        assert emb.supply_dir(tmp_path) == 3 and {key[0] for key in emb.SUPPLIED} == {"prompt 0", "prompt 1", "prompt 2"}
    finally:
        emb.clear()


def serve(service, **kwargs):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.make_handler(service, emb, **kwargs))
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def running(service, pipe):
    worker = threading.Thread(target=service.run, args=(pipe,), daemon=True)
    worker.start()
    assert service.ready.wait(5)
    return worker


def test_the_service_answers_requests_until_stopped():
    service = emb.EmbeddingService("m", helpers=HELPERS)
    with pytest.raises(emb.ServiceNotReady):
        service.request("too early")
    pipe = EncPipe()
    worker = running(service, pipe)
    result = service.request("hello world", [png()], 832, 480)
    assert result.tokens == 6 + 5 and result.model == "m" and service.stats()["served"] == 1
    pipe._encode_prompt = lambda prompt, presentation: (_ for _ in ()).throw(RuntimeError("out of memory"))
    with pytest.raises(RuntimeError, match="out of memory"):
        service.request("fails")
    assert service.stats()["failed"] == 1
    service.stop()
    worker.join(5)
    assert not worker.is_alive() and not service.running
    with pytest.raises(emb.ServiceNotReady):
        service.request("too late")


def test_requests_left_when_the_service_stops_are_answered():
    service = emb.EmbeddingService(helpers=HELPERS)
    service.ready.set()  # as if running: the request is queued
    errors = []
    waiter = threading.Thread(target=lambda: errors.append(pytest.raises(emb.ServiceNotReady, service.request, "queued")), daemon=True)
    waiter.start()
    while service.requests.empty():
        pass
    pipe = EncPipe()
    pipe._interrupt = True  # WanGP stopped the generation
    service.run(pipe)
    waiter.join(5)
    assert errors and not pipe.calls


def test_the_http_endpoint():
    service = emb.EmbeddingService("m", helpers=HELPERS)
    worker = running(service, EncPipe())
    httpd, url = serve(service, token="secret")
    auth = {"Authorization": "Bearer secret", "Content-Type": "application/json"}

    def call(method, path, body=None, headers=auth):
        request = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(), method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    try:
        assert call("GET", "/v1/health", headers={})[0] == 401
        status, _, body = call("GET", "/v1/health")
        assert status == 200 and json.loads(body)["ready"] and json.loads(body)["dim"] == 5120
        image = base64.b64encode(png()).decode()
        status, headers, body = call("POST", "/v1/embeddings", {"prompt": "a b c", "images": [image, image]})
        assert status == 200 and headers["Content-Type"] == "application/octet-stream" and headers["X-H3-Tokens"] == "19"
        result = emb.from_bytes(body)
        assert result.prompt == "a b c" and result.presentation == ("image", "image") and result.tokens == 19
        status, _, body = call("POST", "/v1/embeddings", {"prompt": "a b c", "format": "json"})
        assert status == 200 and emb.from_json(json.loads(body)).tokens == 9
        for bad in ({}, {"prompt": " "}, {"prompt": "x", "images": ["not base64!"]}, {"prompt": "x", "width": 8}, {"prompt": "x", "format": "npy"},
                    {"prompt": "x", "images": [image] * 5}, {"prompt": "x", "images": [3]}):
            assert call("POST", "/v1/embeddings", bad)[0] == 400, bad
        assert call("POST", "/v1/other", {"prompt": "x"})[0] == 404
        wrapped = "\n".join(image[i:i + 76] for i in range(0, len(image), 76))  # base64 with line breaks, as `base64` writes it
        assert call("POST", "/v1/embeddings", {"prompt": "x", "images": [wrapped]})[0] == 200
        assert emb.fetch(url, "fetched prompt", [png()], token="secret").presentation == ("image",)
        service.stop()
        worker.join(5)
        status, headers, _ = call("POST", "/v1/embeddings", {"prompt": "x"})
        assert status == 503 and headers["Retry-After"] == "10"
    finally:
        httpd.shutdown()
        service.stop()


def test_the_dry_run_pipeline_has_h3s_shape():
    context, tags = server.DryRunPipeline()._encode_prompt("five words in this prompt", emb.presentation_for([png(), png()], 832, 480, HELPERS))
    assert tuple(context.shape) == (1, 30 + 5 + 2 * 397, 5120) and context.dtype == torch.bfloat16 and int((tags == 0).sum()) == 2 * 397


def test_the_generate_hook_serves_embeddings_while_a_service_waits():
    class Pipeline(EncPipe):
        def generate(self, input_prompt="", seed=0, frame_num=17, fps=24, **kwargs):
            return {"x": torch.zeros(3, 1, 2, 2), "audio": None, "audio_sampling_rate": 1}

    assert live.install(Pipeline) == []
    service = emb.EmbeddingService(helpers=HELPERS)
    live.STATE["embed_service"] = service
    try:
        pipe, returned = Pipeline(), []
        worker = threading.Thread(target=lambda: returned.append(pipe.generate(input_prompt="service")), daemon=True)
        worker.start()
        assert service.ready.wait(5)
        assert service.request("through the hook").tokens == 9 and pipe.calls == [("through the hook", 0)]
        service.stop()
        worker.join(5)
        assert returned == [None]  # the service generation makes no video
        live.STATE["embed_service"] = None
        assert pipe.generate(input_prompt="video")["x"] is not None  # without a service: as usual
    finally:
        live.STATE["embed_service"] = None
        live.uninstall()


def test_a_session_supplies_the_embeddings_of_its_folder(tmp_path, monkeypatch):
    emb.clear()
    folder = tmp_path / "embeddings"
    folder.mkdir()
    emb.save(embedding("clip prompt", kinds=()), folder / "clip.safetensors")
    monkeypatch.setattr(live, "find_ffmpeg", lambda: None)
    try:
        with pytest.raises(RuntimeError, match="ffmpeg"):  # stops right after reading its settings
            live.run_session(lambda *a, **k: None, EncPipe(), {"input_prompt": "x", "frame_num": 17, "fps": 24},
                             live.LiveSettings(enabled=True, embeddings_dir=str(folder)), root=tmp_path)
        assert ("clip prompt", ()) in emb.SUPPLIED
        assert live.LiveSettings.from_dict({"embeddings_dir": str(folder)}).embeddings_dir == str(folder)
    finally:
        emb.clear()
