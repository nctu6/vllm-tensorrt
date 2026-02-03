# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import asyncio
import itertools
import os
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
import yaml
from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.inputs import PromptType
from vllm.inputs.data import is_tokens_prompt
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.utils import get_prompt_text

logger = init_logger(__name__)

try:
    import _turbomind as _tm
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    _TM_IMPORT_ERROR = exc
    _tm = None  # type: ignore[assignment]


def _require_turbomind() -> None:
    if _tm is None:
        raise ModuleNotFoundError(
            "TurboMind backend requires the `_turbomind` extension module. "
            "Build/install TurboMind and ensure it is on PYTHONPATH."
        ) from _TM_IMPORT_ERROR


def _construct_stop_or_bad_words(words: list[int] | None = None):
    if words is None or len(words) == 0:
        return None
    offsets = list(range(1, len(words) + 1))
    return [words, offsets]


class StreamingSemaphore:
    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.fut: asyncio.Future | None = None
        self.val = 0

    async def acquire(self) -> None:
        if self.val:
            self.val = 0
            return
        self.fut = self.loop.create_future()
        await self.fut
        self.fut = None
        self.val = 0

    def release(self) -> None:
        if not self.val:
            self.val = 1
            if self.fut and not self.fut.done():
                self.fut.set_result(None)

    def release_threadsafe(self) -> None:
        self.loop.call_soon_threadsafe(self.release)


@dataclass
class TMOutput:
    token_ids: list[int]
    status: int
    logprobs: list[dict[int, float]] | None = None


def _get_logprobs_impl(
    logprob_vals: torch.Tensor,
    logprob_idxs: torch.Tensor,
    logprob_nums: torch.Tensor,
    output_ids: list[int],
    logprobs: int,
    offset: int,
) -> list[dict[int, float]]:
    out_logprobs: list[dict[int, float]] = []
    length = len(output_ids) + offset
    for pos, idx, val, n in zip(
        range(len(output_ids)),
        logprob_idxs[offset:length],
        logprob_vals[offset:length],
        logprob_nums[offset:length],
    ):
        topn = min(n.item(), logprobs)
        tok_res = {idx[i].item(): val[i].item() for i in range(topn)}
        token_id = output_ids[pos]
        if token_id not in tok_res:
            valid_n = n.item()
            tok_res[token_id] = val[:valid_n][idx[:valid_n] == token_id].item()
        ids = list(tok_res.keys())
        for k in ids:
            if tok_res[k] == float("-inf"):
                tok_res.pop(k)
        out_logprobs.append(tok_res)
    return out_logprobs


def _get_logprobs(outputs: dict, output_logprobs: int):
    logprob_vals = outputs["logprob_vals"]
    logprob_idxs = outputs["logprob_indexes"]
    logprob_nums = outputs["logprob_nums"]
    offset = 0

    def _func(out: TMOutput) -> None:
        nonlocal offset
        out.logprobs = _get_logprobs_impl(
            logprob_vals, logprob_idxs, logprob_nums, out.token_ids, output_logprobs, offset
        )
        offset += len(out.token_ids)

    return _func


@dataclass
class TurbomindEngineConfig:
    tp: int = 1
    dp: int = 1
    cp: int = 1
    session_len: int | None = None
    max_batch_size: int | None = None
    cache_max_entry_count: float = 0.8
    cache_chunk_size: int = -1
    cache_block_seq_len: int = 64
    enable_prefix_caching: bool = False
    quant_policy: int = 0
    max_prefill_token_num: int = 8192
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1
    async_: int = 1
    devices: list[int] | None = None
    nnodes: int = 1
    node_rank: int = 0
    communicator: str = "nccl"
    enable_metrics: bool = False
    outer_dp_size: int = 1
    attn_dp_size: int = 1
    attn_tp_size: int = 1
    attn_cp_size: int = 1
    mlp_tp_size: int = 1

    def finalize(self) -> None:
        if self.devices is None:
            self.devices = list(range(self.tp))
        self.attn_dp_size = 1
        self.attn_tp_size = self.tp
        self.attn_cp_size = self.cp
        self.outer_dp_size = 1
        self.mlp_tp_size = self.tp

    def to_engine_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "max_batch_size": self.max_batch_size or 0,
            "max_prefill_token_num": self.max_prefill_token_num,
            "max_context_token_num": 0,
            "cache_max_entry_count": self.cache_max_entry_count,
            "cache_chunk_size": self.cache_chunk_size,
            "enable_prefix_caching": self.enable_prefix_caching,
            "quant_policy": self.quant_policy,
            "num_tokens_per_iter": self.num_tokens_per_iter,
            "max_prefill_iters": self.max_prefill_iters,
            "async_": self.async_,
            "outer_dp_size": self.outer_dp_size,
            "attn_dp_size": self.attn_dp_size,
            "attn_tp_size": self.attn_tp_size,
            "attn_cp_size": self.attn_cp_size,
            "mlp_tp_size": self.mlp_tp_size,
            "devices": self.devices,
            "nnodes": self.nnodes,
            "node_rank": self.node_rank,
            "communicator": self.communicator,
            "enable_metrics": self.enable_metrics,
        }


class TurboMindRuntime:
    def __init__(
        self,
        model_dir: str,
        engine_config: TurbomindEngineConfig,
        vllm_config: VllmConfig,
    ) -> None:
        _require_turbomind()
        self.model_dir = model_dir
        self.engine_config = engine_config
        self.vllm_config = vllm_config
        self._use_in_memory_weights = False
        self._tm_model = None
        self.config_dict = self._load_config()
        self.model_comm = self._create_model_comm()
        self._create_weights()
        if self._use_in_memory_weights:
            self._load_weights_from_hf()
        else:
            self._load_weights_from_dir()
        self._process_weights()
        self._create_engine()

    def _load_config(self) -> dict[str, Any]:
        config_path = os.path.join(self.model_dir, "config.yaml")
        if not os.path.exists(config_path):
            config_dict = self._build_config_from_hf()
            self._use_in_memory_weights = True
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
            if "model_config" not in config_dict:
                raise ValueError(f"Invalid TurboMind config: {config_path}")

        model_cfg = config_dict.setdefault("model_config", {})
        attn_cfg = config_dict.setdefault("attention_config", {})

        if self.engine_config.session_len:
            model_cfg["session_len"] = self.engine_config.session_len
        if self.engine_config.cache_block_seq_len:
            attn_cfg["cache_block_seq_len"] = self.engine_config.cache_block_seq_len

        config_dict["engine_config"] = self.engine_config.to_engine_dict()
        return config_dict

    def _sanitize_model_id(self, model_path: str) -> str:
        model_id = os.path.basename(model_path.rstrip("/")) or "model"
        safe = model_id.replace("/", "_").replace(":", "_")
        return safe[:80]

    def _resolve_lmdeploy_dtype(self) -> str:
        dtype = self.vllm_config.model_config.dtype
        if isinstance(dtype, torch.dtype):
            if dtype == torch.bfloat16:
                return "bfloat16"
            if dtype == torch.float16:
                return "float16"
            return "auto"
        if isinstance(dtype, str):
            lowered = dtype.lower()
            if lowered in ("auto", "float16", "bfloat16"):
                return lowered
        return "auto"

    def _build_lmdeploy_engine_config(self):
        from lmdeploy.messages import TurbomindEngineConfig as LmdeployEngineConfig

        lm_cfg = LmdeployEngineConfig(
            dtype=self._resolve_lmdeploy_dtype(),
            model_format=None,
            tp=self.engine_config.tp,
            dp=self.engine_config.dp,
            cp=self.engine_config.cp,
            session_len=self.engine_config.session_len,
            max_batch_size=self.engine_config.max_batch_size,
            cache_max_entry_count=self.engine_config.cache_max_entry_count,
            cache_chunk_size=self.engine_config.cache_chunk_size,
            cache_block_seq_len=self.engine_config.cache_block_seq_len,
            enable_prefix_caching=self.engine_config.enable_prefix_caching,
            quant_policy=self.engine_config.quant_policy,
            max_prefill_token_num=self.engine_config.max_prefill_token_num,
            num_tokens_per_iter=self.engine_config.num_tokens_per_iter,
            max_prefill_iters=self.engine_config.max_prefill_iters,
            async_=self.engine_config.async_,
            devices=self.engine_config.devices,
            nnodes=self.engine_config.nnodes,
            node_rank=self.engine_config.node_rank,
            communicator=self.engine_config.communicator,
            enable_metrics=self.engine_config.enable_metrics,
            download_dir=self.vllm_config.load_config.download_dir,
            revision=self.vllm_config.model_config.revision,
        )
        lm_cfg.attn_tp_size = self.engine_config.tp
        lm_cfg.attn_cp_size = self.engine_config.cp
        lm_cfg.mlp_tp_size = self.engine_config.tp
        return lm_cfg

    def _build_config_from_hf(self) -> dict[str, Any]:
        try:
            from lmdeploy.turbomind.deploy.converter import get_tm_model
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TurboMind conversion requires the vendored lmdeploy pipeline "
                "and its Python dependencies (e.g., mmengine)."
            ) from exc

        engine_cfg = self._build_lmdeploy_engine_config()
        model_name = self._sanitize_model_id(self.model_dir)
        tm_model = get_tm_model(
            self.model_dir,
            model_name,
            "",
            engine_cfg,
            out_dir="",
        )
        tm_cfg = tm_model.tm_config
        tm_cfg.update_from_engine_config(engine_cfg)
        self._tm_model = tm_model
        return tm_cfg.to_dict()

    def _create_model_comm(self):
        weight_type = self.config_dict.get("model_config", {}).get("weight_type", "float16")
        return _tm.TurboMind.create(
            model_dir="",
            config=yaml.safe_dump(self.config_dict),
            weight_type=weight_type,
        )

    def _create_weights(self) -> None:
        for device_id in range(len(self.engine_config.devices or [])):
            self.model_comm.create_weights(device_id)

    def _process_weights(self) -> None:
        for device_id in range(len(self.engine_config.devices or [])):
            self.model_comm.process_weight(device_id)

    def _create_engine(self) -> None:
        for device_id in range(len(self.engine_config.devices or [])):
            self.model_comm.create_engine(device_id)

    def _load_weights_from_dir(self) -> None:
        tensor_maps = []
        for device_id in range(len(self.engine_config.devices or [])):
            tensor_maps.append(self.model_comm.get_weights(device_id))

        missing: list[str] = []
        for tensor_map in tensor_maps:
            for name, tm_tensor in tensor_map.items():
                file_path = os.path.join(self.model_dir, name)
                if not os.path.exists(file_path):
                    missing.append(name)
                    continue
                tensor = self._load_tensor_file(file_path, tm_tensor)
                tm_tensor.copy_from(tensor)
        if missing:
            sample = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"TurboMind weight files missing for {len(missing)} tensors. "
                f"Examples: {sample}"
            )

    def _load_weights_from_hf(self) -> None:
        if self._tm_model is None:
            raise RuntimeError("TurboMind conversion model is not initialized.")
        tm_params: dict[str, list] = {}
        for device_id in range(len(self.engine_config.devices or [])):
            tensor_map = self.model_comm.get_weights(device_id)
            for name, tm_tensor in tensor_map.items():
                tm_params.setdefault(name, []).append(tm_tensor)
        self._tm_model.tm_params = tm_params
        self._tm_model.export()
        if tm_params:
            missing = ", ".join(sorted(tm_params.keys())[:32])
            logger.warning("TurboMind conversion left unused weights: %s", missing)

    def _load_tensor_file(self, file_path: str, tm_tensor) -> torch.Tensor:
        dtype = tm_tensor.type
        shape = tm_tensor.shape
        np_dtype = self._np_dtype_from_tm(dtype)
        data = np.fromfile(file_path, dtype=np_dtype)
        expected = int(np.prod(shape))
        if data.size != expected:
            raise ValueError(
                f"Invalid weight file {file_path}: expected {expected} elements, got {data.size}."
            )
        data = data.reshape(shape)
        if dtype == _tm.DataType.TYPE_BF16:
            torch_tensor = torch.from_numpy(data).view(torch.bfloat16)
        else:
            torch_tensor = torch.from_numpy(data)
        return torch_tensor

    def _np_dtype_from_tm(self, dtype) -> np.dtype:
        if dtype == _tm.DataType.TYPE_FP16:
            return np.float16
        if dtype == _tm.DataType.TYPE_BF16:
            return np.uint16
        if dtype == _tm.DataType.TYPE_FP32:
            return np.float32
        if dtype == _tm.DataType.TYPE_INT32:
            return np.int32
        if dtype == _tm.DataType.TYPE_INT16:
            return np.int16
        if dtype == _tm.DataType.TYPE_INT8:
            return np.int8
        if dtype == _tm.DataType.TYPE_UINT32:
            return np.uint32
        raise ValueError(f"Unsupported TurboMind dtype: {dtype}")

    def create_request(self) -> "TurboMindRequestHandle":
        return TurboMindRequestHandle(self)

    def sleep(self, level: int = 1) -> None:
        for device_id in range(len(self.engine_config.devices or [])):
            self.model_comm.sleep(device_id, level)

    def wakeup(self, tags: list[str] | None = None) -> None:
        tags = tags or ["weights", "kv_cache"]
        for device_id in range(len(self.engine_config.devices or [])):
            self.model_comm.wakeup(device_id, tags)

    def close(self) -> None:
        self.model_comm = None


class TurboMindRequestHandle:
    def __init__(self, runtime: TurboMindRuntime):
        self.runtime = runtime
        self.model_inst = runtime.model_comm.create_request()

    async def async_cancel(self) -> None:
        self.model_inst.cancel()

    def async_end_cb(self, fut: asyncio.Future, status: int):
        fut.get_loop().call_soon_threadsafe(fut.set_result, status)

    async def async_end(self, session_id: int) -> None:
        fut = asyncio.get_running_loop().create_future()
        self.model_inst.end(lambda status: self.async_end_cb(fut, status), session_id)
        await fut

    async def async_stream_infer(
        self,
        session_id: int,
        input_ids: list[int],
        gen_cfg,
        stream_output: bool = True,
        sequence_start: bool = True,
        sequence_end: bool = True,
        step: int = 0,
    ) -> AsyncGenerator[TMOutput, None]:
        input_ids_tensor = torch.IntTensor(input_ids)
        inputs = {"input_ids": input_ids_tensor}
        inputs = _np_dict_to_tm_dict(inputs)

        session = _tm.SessionParam(
            id=session_id, step=step, start=sequence_start, end=sequence_end
        )

        sem = StreamingSemaphore()
        outputs, shared_state, _metrics = self.model_inst.forward(
            inputs,
            session,
            gen_cfg,
            stream_output,
            False,
            sem.release_threadsafe,
        )

        outputs = _tm_dict_to_torch_dict(outputs)
        output_ids_buf = outputs["output_ids"]

        output_logprobs = getattr(gen_cfg, "output_logprobs", None)
        extra_logprobs = (
            _get_logprobs(outputs, output_logprobs)
            if output_logprobs
            else None
        )

        prev_len = step + len(input_ids)
        state = None
        try:
            while True:
                await sem.acquire()
                state = shared_state.consume()
                status, seq_len = state.status, state.seq_len

                if seq_len == prev_len and status == 0:
                    continue

                if seq_len < prev_len:
                    yield TMOutput(token_ids=[], status=status)
                    break

                token_ids = output_ids_buf[prev_len:seq_len].tolist()
                out = TMOutput(token_ids=token_ids, status=status)
                if extra_logprobs is not None:
                    extra_logprobs(out)
                prev_len = seq_len
                yield out
                if status in (7, 8):  # finish / cancel
                    break
        finally:
            while not state or state.status == 0:
                await sem.acquire()
                state = shared_state.consume()


def _np_dict_to_tm_dict(np_dict: dict):
    ret = _tm.TensorMap()
    for k, v in np_dict.items():
        ret[k] = _tm.from_dlpack(v)
    return ret


def _tm_dict_to_torch_dict(tm_dict):
    ret = {}
    for k, v in tm_dict.items():
        if v.type == _tm.DataType.TYPE_UINT32:
            v = v.view(_tm.DataType.TYPE_INT32)
        ret[k] = torch.from_dlpack(v)
    return ret


class TurbomindEngineClient(EngineClient):
    """EngineClient adapter that uses TurboMind backend directly."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        engine_config: TurbomindEngineConfig | None = None,
    ) -> None:
        _require_turbomind()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.input_processor = InputProcessor(vllm_config)
        self.io_processor = get_io_processor(
            vllm_config,
            self.model_config.io_processor_plugin,
        )
        self._engine_config = engine_config or self._build_engine_config(vllm_config)
        self._runtime = TurboMindRuntime(
            model_dir=self.model_config.model,
            engine_config=self._engine_config,
            vllm_config=self.vllm_config,
        )
        self._req_id_to_session: dict[str, int] = {}
        self._session_id_gen = itertools.count(1)
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._stopped = False
        self._errored = False
        self._dead_error: BaseException | None = None
        logger.info(
            "TurboMind backend enabled (no external lmdeploy). "
            "HF models will be auto-converted in-memory if config.yaml is missing."
        )

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
    ) -> "TurbomindEngineClient":
        return cls(vllm_config=vllm_config)

    @property
    def renderer(self) -> BaseRenderer:
        return self.input_processor.renderer

    @property
    def is_running(self) -> bool:
        return not self._stopped

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def errored(self) -> bool:
        return self._errored

    @property
    def dead_error(self) -> BaseException:
        if self._dead_error is None:
            return RuntimeError("TurboMind engine error state not set.")
        return self._dead_error

    async def generate(
        self,
        prompt: EngineCoreRequest | PromptType | AsyncGenerator[Any, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request=None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        if isinstance(prompt, AsyncGenerator):
            raise ValueError("Streaming inputs are not supported by turbomind backend.")
        if lora_request is not None:
            raise ValueError("LoRA is not supported by turbomind backend.")
        if tokenization_kwargs:
            logger.debug("tokenization_kwargs ignored by turbomind backend.")
        if trace_headers:
            logger.debug("trace_headers ignored by turbomind backend.")
        if priority or data_parallel_rank:
            logger.debug("priority/data_parallel_rank ignored by turbomind backend.")

        await self._pause_event.wait()

        input_ids: list[int] | None = None
        prompt_str: str | None = None
        params = sampling_params

        if isinstance(prompt, EngineCoreRequest):
            if prompt_text is not None:
                logger.debug(
                    "prompt_text provided with EngineCoreRequest; it is used only for output metadata."
                )
            if prompt.prompt_token_ids is None:
                raise ValueError("EngineCoreRequest.prompt_token_ids is required.")
            input_ids = prompt.prompt_token_ids
            prompt_str = prompt_text
        else:
            if prompt_text is not None:
                raise ValueError("prompt_text should only be provided with EngineCoreRequest.")
            if is_tokens_prompt(prompt):
                input_ids = prompt["prompt_token_ids"]
                prompt_str = prompt.get("prompt")
            else:
                prompt_str = get_prompt_text(prompt)
                if prompt_str is None:
                    raise ValueError("Unsupported prompt type for turbomind backend.")
                req = self.input_processor.process_inputs(
                    request_id=request_id,
                    prompt=prompt,
                    params=params,
                    tokenization_kwargs=None,
                )
                input_ids = req.prompt_token_ids
                params = req.params

        if input_ids is None:
            raise ValueError("prompt_token_ids are required for turbomind backend.")

        gen_cfg, stop_ids = self._sampling_params_to_tm_config(params)

        session_id = self._req_id_to_session.get(request_id)
        if session_id is None:
            session_id = next(self._session_id_gen)
            self._req_id_to_session[request_id] = session_id

        output_kind = params.output_kind
        stream_response = output_kind != RequestOutputKind.FINAL_ONLY

        cumulative_text = ""
        cumulative_token_ids: list[int] = []
        cumulative_logprobs: list[dict[int, float]] = []

        handle = self._runtime.create_request()
        try:
            async for out in handle.async_stream_infer(
                session_id=session_id,
                input_ids=input_ids,
                gen_cfg=gen_cfg,
                stream_output=stream_response,
                sequence_start=True,
                sequence_end=True,
                step=0,
            ):
                delta_token_ids = out.token_ids or []
                delta_logprobs = out.logprobs
                if output_kind == RequestOutputKind.FINAL_ONLY:
                    cumulative_token_ids.extend(delta_token_ids)
                    if delta_logprobs is not None:
                        cumulative_logprobs.extend(delta_logprobs)
                    if out.status not in (7, 8):
                        continue
                    output_token_ids = list(cumulative_token_ids)
                    output_logprobs = (
                        list(cumulative_logprobs) if cumulative_logprobs else None
                    )
                elif output_kind == RequestOutputKind.CUMULATIVE:
                    cumulative_token_ids.extend(delta_token_ids)
                    if delta_logprobs is not None:
                        cumulative_logprobs.extend(delta_logprobs)
                    output_token_ids = list(cumulative_token_ids)
                    output_logprobs = (
                        list(cumulative_logprobs) if cumulative_logprobs else None
                    )
                else:
                    output_token_ids = list(delta_token_ids)
                    output_logprobs = delta_logprobs

                if params.detokenize:
                    tokenizer = self.input_processor.tokenizer
                    if tokenizer is None:
                        raise ValueError("Tokenizer unavailable for detokenize.")
                    if output_kind == RequestOutputKind.DELTA:
                        delta_text = tokenizer.decode(delta_token_ids)
                        cumulative_text += delta_text
                        output_text = delta_text
                    else:
                        output_text = tokenizer.decode(output_token_ids)
                else:
                    output_text = ""

                finish_reason = None
                if out.status == 7:
                    finish_reason = "stop" if (stop_ids and delta_token_ids and delta_token_ids[-1] in stop_ids) else "length"
                elif out.status == 8:
                    finish_reason = "abort"
                elif out.status != 0:
                    finish_reason = "error"

                completion = CompletionOutput(
                    index=0,
                    text=output_text,
                    token_ids=output_token_ids,
                    cumulative_logprob=None,
                    logprobs=output_logprobs,
                    routed_experts=None,
                    finish_reason=finish_reason,
                    stop_reason=None,
                    lora_request=lora_request,
                )

                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt_str,
                    prompt_token_ids=input_ids,
                    prompt_logprobs=None,
                    outputs=[completion],
                    finished=finish_reason is not None,
                    lora_request=lora_request,
                )
        except Exception as exc:
            self._errored = True
            self._dead_error = exc
            logger.exception("TurboMind generation failed for request %s", request_id)
            await handle.async_cancel()
            raise
        finally:
            self._req_id_to_session.pop(request_id, None)
            # TurboMind ends sessions internally when sequence_end is True.

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request=None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        raise NotImplementedError("Pooling is not supported by turbomind backend.")

    async def abort(self, request_id: str | Iterable[str]) -> None:
        req_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        for req_id in req_ids:
            session_id = self._req_id_to_session.pop(req_id, None)
            if session_id is None:
                continue
            handle = self._runtime.create_request()
            await handle.async_cancel()

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(self) -> None:
        return

    async def check_health(self) -> None:
        if self.errored:
            raise self.dead_error

    async def start_profile(self) -> None:
        logger.warning("Profiling is not supported by turbomind backend.")

    async def stop_profile(self) -> None:
        logger.warning("Profiling is not supported by turbomind backend.")

    async def reset_mm_cache(self) -> None:
        self.input_processor.clear_mm_cache()

    async def reset_encoder_cache(self) -> None:
        return

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return False

    async def sleep(self, level: int = 1) -> None:
        self._runtime.sleep(level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        self._runtime.wakeup(tags)

    async def is_sleeping(self) -> bool:
        return False

    async def add_lora(self, lora_request) -> bool:
        logger.warning("LoRA is not supported by turbomind backend.")
        return False

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        self._paused = True
        self._pause_event.clear()

    async def resume_generation(self) -> None:
        self._paused = False
        self._pause_event.set()

    async def is_paused(self) -> bool:
        return self._paused

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        raise NotImplementedError("collective_rpc is not supported by turbomind backend.")

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    def shutdown(self) -> None:
        if self._stopped:
            return
        self._runtime.close()
        self._stopped = True

    def _build_engine_config(self, vllm_config: VllmConfig) -> TurbomindEngineConfig:
        config = TurbomindEngineConfig()
        config.tp = vllm_config.parallel_config.tensor_parallel_size
        config.cp = vllm_config.parallel_config.decode_context_parallel_size
        if vllm_config.model_config.max_model_len and vllm_config.model_config.max_model_len > 0:
            config.session_len = vllm_config.model_config.max_model_len
        if vllm_config.scheduler_config.max_num_seqs is not None:
            config.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        if vllm_config.cache_config.block_size:
            config.cache_block_seq_len = vllm_config.cache_config.block_size
        config.cache_max_entry_count = vllm_config.cache_config.gpu_memory_utilization
        config.enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching
        config.devices = list(range(config.tp))
        config.enable_metrics = False
        return config

    def _sampling_params_to_tm_config(
        self, params: SamplingParams
    ) -> tuple[Any, list[int]]:
        if params.n > 1:
            raise ValueError("TurboMind backend does not support n > 1.")
        if params.presence_penalty or params.frequency_penalty:
            raise ValueError("presence_penalty/frequency_penalty are not supported by turbomind backend.")
        if params.prompt_logprobs is not None:
            raise ValueError("prompt_logprobs are not supported by turbomind backend.")
        if params.logits_processors is not None:
            raise ValueError("logits_processors are not supported by turbomind backend.")
        if params.logit_bias is not None or params.allowed_token_ids is not None:
            raise ValueError("logit_bias/allowed_token_ids are not supported by turbomind backend.")
        if params.structured_outputs is not None:
            raise ValueError("structured_outputs are not supported by turbomind backend.")
        if params.stop:
            raise ValueError("stop strings are not supported by turbomind backend.")

        cfg = _tm.GenerationConfig()
        cfg.max_new_tokens = params.max_tokens or 0
        cfg.min_new_tokens = params.min_tokens or 0
        cfg.top_k = 0 if params.top_k in (0, -1) else params.top_k
        cfg.top_p = params.top_p
        cfg.min_p = params.min_p
        cfg.temperature = params.temperature
        cfg.repetition_penalty = params.repetition_penalty
        if params.stop_token_ids:
            cfg.eos_ids = params.stop_token_ids
            if not params.ignore_eos:
                cfg.stop_ids = _construct_stop_or_bad_words(params.stop_token_ids)
        if params.bad_words:
            cfg.bad_ids = _construct_stop_or_bad_words(params.bad_words)
        if params.logprobs:
            cfg.output_logprobs = min(params.logprobs, 1024)
        if params.seed is not None:
            cfg.random_seed = params.seed
        return cfg, list(params.stop_token_ids or [])
