# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
import ctypes
import os
import shutil
import tempfile

import torch

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

_TRTLLM_IMPORT_ERROR: BaseException | None = None


def _ensure_real_libcuda_loaded() -> None:
    if os.name != "posix":
        return
    cuda_lib_dirs = []
    for candidate in (
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
    ):
        if os.path.isdir(candidate):
            cuda_lib_dirs.append(candidate)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_path:
        parts = [p for p in ld_path.split(":") if p and "stubs" not in p]
    else:
        parts = []
    for libdir in reversed(cuda_lib_dirs):
        if libdir not in parts:
            parts.insert(0, libdir)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    for candidate in (
        "/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    ):
        if os.path.exists(candidate):
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:  # pragma: no cover - best effort
                logger.warning("Failed to load %s: %s", candidate, exc)
            break
    for candidate in (
        "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/lib/x86_64-linux-gnu/libcudart.so.12",
        "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
    ):
        if os.path.exists(candidate):
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:  # pragma: no cover - best effort
                logger.warning("Failed to load %s: %s", candidate, exc)
            break


def _lazy_import_tensorrt_llm():
    global _TRTLLM_IMPORT_ERROR
    try:
        _ensure_real_libcuda_loaded()
        from tensorrt_llm.runtime.generation import SamplingConfig  # type: ignore
        from tensorrt_llm.runtime.model_runner_cpp import (  # type: ignore
            ModelRunnerCpp,
        )
    except BaseException as exc:  # pragma: no cover - optional dependency
        _TRTLLM_IMPORT_ERROR = exc
        return None, None
    return SamplingConfig, ModelRunnerCpp


def _require_tensorrt_llm() -> None:
    sampling_cfg, runner = _lazy_import_tensorrt_llm()
    if sampling_cfg is None or runner is None:
        raise ModuleNotFoundError(
            "TensorRT-LLM backend requires the vendored `tensorrt_llm` package "
            "and its compiled bindings."
        ) from _TRTLLM_IMPORT_ERROR


def _dtype_to_trtllm_str(dtype: Any) -> str:
    if dtype is None:
        return "auto"
    if isinstance(dtype, torch.dtype):
        if dtype is torch.float16:
            return "float16"
        if dtype is torch.bfloat16:
            return "bfloat16"
        if dtype is torch.float32:
            return "float32"
        if dtype is torch.float64:
            return "float64"
        return str(dtype)
    return str(dtype)


def _resolve_trtllm_engine_dir(
    vllm_config: VllmConfig,
    engine_config: "TRTLLMEngineConfig",
) -> tuple[str, str | None]:
    model_dir = vllm_config.model_config.model
    trust_remote_code = vllm_config.model_config.trust_remote_code
    workspace = None

    try:
        from tensorrt_llm.llmapi.llm_args import (  # type: ignore
            TrtLlmArgs,
            _ModelFormatKind,
            get_model_format,
        )
        from tensorrt_llm.llmapi.llm_utils import (  # type: ignore
            CachedModelLoader,
            LlmBuildStats,
        )
    except BaseException as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "TensorRT-LLM backend requires the vendored `tensorrt_llm` package "
            "and its compiled bindings."
        ) from exc

    try:
        model_format = get_model_format(model_dir,
                                        trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to infer TensorRT-LLM model format for %s; assuming engine dir. Error: %s",
            model_dir,
            exc,
        )
        return model_dir, None

    if model_format is _ModelFormatKind.TLLM_ENGINE:
        return model_dir, None

    tmp_root = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    workspace = os.path.join(tmp_root, "vllm_trtllm_engine")
    if os.path.isdir(workspace):
        shutil.rmtree(workspace)
    os.makedirs(workspace, exist_ok=True)

    llm_args_kwargs: dict[str, Any] = {
        "model": model_dir,
        "tensor_parallel_size": vllm_config.parallel_config.tensor_parallel_size,
        "dtype": _dtype_to_trtllm_str(vllm_config.model_config.dtype),
        "trust_remote_code": trust_remote_code,
        "enable_build_cache": False,
    }

    if engine_config.max_batch_size is not None:
        llm_args_kwargs["max_batch_size"] = engine_config.max_batch_size
    if engine_config.max_input_len is not None:
        llm_args_kwargs["max_input_len"] = engine_config.max_input_len
        llm_args_kwargs["max_seq_len"] = engine_config.max_input_len
    if engine_config.max_beam_width is not None:
        llm_args_kwargs["max_beam_width"] = engine_config.max_beam_width

    max_num_tokens = getattr(vllm_config.scheduler_config,
                             "max_num_batched_tokens", None)
    if max_num_tokens is not None:
        llm_args_kwargs["max_num_tokens"] = max_num_tokens

    llm_args = TrtLlmArgs(**llm_args_kwargs)
    stats = LlmBuildStats()
    loader = CachedModelLoader(llm_args=llm_args,
                               llm_build_stats=stats,
                               workspace=workspace)
    engine_dir, _ = loader()

    if stats.cache_hitted:
        logger.info("Reusing cached TensorRT-LLM engine at %s", engine_dir)
    else:
        logger.info("Built TensorRT-LLM engine at %s", engine_dir)

    return str(engine_dir), workspace


@dataclass
class TRTLLMEngineConfig:
    max_batch_size: int | None = None
    max_input_len: int | None = None
    max_output_len: int | None = None
    max_beam_width: int | None = None
    device_ids: list[int] | None = None


class TensorRTLLMRuntime:
    def __init__(self, engine_dir: str, config: TRTLLMEngineConfig) -> None:
        _require_tensorrt_llm()
        SamplingConfig, ModelRunnerCpp = _lazy_import_tensorrt_llm()
        if SamplingConfig is None or ModelRunnerCpp is None:
            raise ModuleNotFoundError(
                "TensorRT-LLM backend requires the vendored `tensorrt_llm` package "
                "and its compiled bindings."
            ) from _TRTLLM_IMPORT_ERROR
        self._SamplingConfig = SamplingConfig
        self.engine_dir = engine_dir
        self.config = config
        self.runner = ModelRunnerCpp.from_dir(
            engine_dir,
            max_batch_size=config.max_batch_size,
            max_input_len=config.max_input_len,
            max_output_len=config.max_output_len,
            max_beam_width=config.max_beam_width,
            device_ids=config.device_ids,
        )

    def generate(
        self,
        input_ids: list[int],
        sampling_config,
        *,
        stop_words_list: list[list[list[int]]] | None,
        bad_words_list: list[list[list[int]]] | None,
        stream: bool,
    ):
        batch_input_ids = [torch.tensor(input_ids, dtype=torch.int32)]
        return self.runner.generate(
            batch_input_ids,
            sampling_config=sampling_config,
            streaming=stream,
            return_dict=True,
            output_sequence_lengths=True,
            max_new_tokens=sampling_config.max_new_tokens,
            end_id=sampling_config.end_id,
            pad_id=sampling_config.pad_id,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
        )


class TRTLLMStreamWorker:
    def __init__(self) -> None:
        self.queue: asyncio.Queue = asyncio.Queue()
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self, generator) -> None:
        def _run():
            try:
                for item in generator:
                    self.queue.put_nowait(item)
            except BaseException as exc:  # noqa: BLE001
                self.error = exc
            finally:
                self.queue.put_nowait(None)

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    async def iter_items(self):
        while True:
            item = await self.queue.get()
            if item is None:
                break
            yield item
        if self.error is not None:
            raise self.error


class TensorRTLLMEngineClient(EngineClient):
    """EngineClient adapter that uses TensorRT-LLM backend directly."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        engine_config: TRTLLMEngineConfig | None = None,
    ) -> None:
        _require_tensorrt_llm()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.input_processor = InputProcessor(vllm_config)
        self.io_processor = get_io_processor(
            vllm_config,
            self.model_config.io_processor_plugin,
        )
        self._engine_config = engine_config or self._build_engine_config(vllm_config)
        engine_dir, workspace = _resolve_trtllm_engine_dir(vllm_config, self._engine_config)
        self._tllm_workspace = workspace
        self._runtime = TensorRTLLMRuntime(
            engine_dir=engine_dir,
            config=self._engine_config,
        )
        self._stopped = False
        self._errored = False
        self._dead_error: BaseException | None = None
        logger.info("TensorRT-LLM backend enabled (vendored).")

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
    ) -> "TensorRTLLMEngineClient":
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
            return RuntimeError("TensorRT-LLM engine error state not set.")
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
            raise ValueError("Streaming inputs are not supported by TensorRT-LLM backend.")
        if lora_request is not None:
            raise ValueError("LoRA is not supported by TensorRT-LLM backend.")
        if tokenization_kwargs:
            logger.debug("tokenization_kwargs ignored by TensorRT-LLM backend.")
        if trace_headers:
            logger.debug("trace_headers ignored by TensorRT-LLM backend.")
        if priority or data_parallel_rank:
            logger.debug("priority/data_parallel_rank ignored by TensorRT-LLM backend.")

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
                    raise ValueError("Unsupported prompt type for TensorRT-LLM backend.")
                req = self.input_processor.process_inputs(
                    request_id=request_id,
                    prompt=prompt,
                    params=params,
                    tokenization_kwargs=None,
                )
                input_ids = req.prompt_token_ids
                params = req.params

        if input_ids is None:
            raise ValueError("prompt_token_ids are required for TensorRT-LLM backend.")

        sampling_config, stop_words, bad_words = self._sampling_params_to_trt(params)
        stream_response = params.output_kind != RequestOutputKind.FINAL_ONLY

        worker = TRTLLMStreamWorker()
        generator = self._runtime.generate(
            input_ids,
            sampling_config,
            stop_words_list=stop_words,
            bad_words_list=bad_words,
            stream=stream_response,
        )
        worker.start(generator)

        output_kind = params.output_kind
        cumulative_token_ids: list[int] = []
        prev_len = len(input_ids)
        last_tokens: list[int] | None = None

        try:
            async for out in worker.iter_items():
                if isinstance(out, dict):
                    output_ids = out.get("output_ids")
                    if isinstance(output_ids, torch.Tensor):
                        output_ids = output_ids.cpu().tolist()
                else:
                    output_ids = out

                if not output_ids:
                    continue

                tokens = output_ids[0][0]
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.cpu().tolist()
                last_tokens = tokens

                if len(tokens) < prev_len:
                    continue
                delta = tokens[prev_len:]
                prev_len = len(tokens)

                if output_kind == RequestOutputKind.FINAL_ONLY:
                    cumulative_token_ids.extend(delta)
                    continue

                output_token_ids = delta if output_kind == RequestOutputKind.DELTA else tokens[len(input_ids):]

                output_text = ""
                if params.detokenize:
                    tokenizer = self.input_processor.tokenizer
                    if tokenizer is None:
                        raise ValueError("Tokenizer unavailable for detokenize.")
                    if output_kind == RequestOutputKind.DELTA:
                        output_text = tokenizer.decode(output_token_ids)
                    else:
                        output_text = tokenizer.decode(output_token_ids)

                completion = CompletionOutput(
                    index=0,
                    text=output_text,
                    token_ids=output_token_ids,
                    cumulative_logprob=None,
                    logprobs=None,
                    routed_experts=None,
                    finish_reason=None,
                    stop_reason=None,
                    lora_request=lora_request,
                )

                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt_str,
                    prompt_token_ids=input_ids,
                    prompt_logprobs=None,
                    outputs=[completion],
                    finished=False,
                    lora_request=lora_request,
                )

            # Final output if needed
            if output_kind == RequestOutputKind.FINAL_ONLY:
                output_token_ids = list(cumulative_token_ids)
            else:
                output_token_ids = (
                    last_tokens[len(input_ids):] if last_tokens is not None else []
                )

            output_text = ""
            if params.detokenize and output_token_ids:
                tokenizer = self.input_processor.tokenizer
                if tokenizer is None:
                    raise ValueError("Tokenizer unavailable for detokenize.")
                output_text = tokenizer.decode(output_token_ids)

            completion = CompletionOutput(
                index=0,
                text=output_text,
                token_ids=output_token_ids,
                cumulative_logprob=None,
                logprobs=None,
                routed_experts=None,
                finish_reason="stop",
                stop_reason=None,
                lora_request=lora_request,
            )

            yield RequestOutput(
                request_id=request_id,
                prompt=prompt_str,
                prompt_token_ids=input_ids,
                prompt_logprobs=None,
                outputs=[completion],
                finished=True,
                lora_request=lora_request,
            )
        except Exception as exc:
            self._errored = True
            self._dead_error = exc
            logger.exception("TensorRT-LLM generation failed for request %s", request_id)
            raise

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
        raise NotImplementedError("Pooling is not supported by TensorRT-LLM backend.")

    async def abort(self, request_id: str | Iterable[str]) -> None:
        return

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(self) -> None:
        return

    async def check_health(self) -> None:
        if self.errored:
            raise self.dead_error

    async def start_profile(self) -> None:
        logger.warning("Profiling is not supported by TensorRT-LLM backend.")

    async def stop_profile(self) -> None:
        logger.warning("Profiling is not supported by TensorRT-LLM backend.")

    async def reset_mm_cache(self) -> None:
        self.input_processor.clear_mm_cache()

    async def reset_encoder_cache(self) -> None:
        return

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return False

    async def sleep(self, level: int = 1) -> None:
        return

    async def wake_up(self, tags: list[str] | None = None) -> None:
        return

    async def is_sleeping(self) -> bool:
        return False

    async def add_lora(self, lora_request) -> bool:
        logger.warning("LoRA is not supported by TensorRT-LLM backend.")
        return False

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        return

    async def resume_generation(self) -> None:
        return

    async def is_paused(self) -> bool:
        return False

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        raise NotImplementedError("collective_rpc is not supported by TensorRT-LLM backend.")

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    def shutdown(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._tllm_workspace:
            try:
                shutil.rmtree(self._tllm_workspace)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to remove TensorRT-LLM workspace %s: %s",
                               self._tllm_workspace, exc)
            self._tllm_workspace = None

    def _build_engine_config(self, vllm_config: VllmConfig) -> TRTLLMEngineConfig:
        config = TRTLLMEngineConfig()
        config.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        config.max_input_len = vllm_config.model_config.max_model_len
        config.max_output_len = None
        config.max_beam_width = 1
        config.device_ids = list(range(vllm_config.parallel_config.tensor_parallel_size))
        return config

    def _sampling_params_to_trt(
        self, params: SamplingParams
    ) -> tuple[Any, list[list[list[int]]] | None, list[list[list[int]]] | None]:
        if params.n > 1:
            raise ValueError("TensorRT-LLM backend does not support n > 1.")
        if params.logits_processors is not None:
            raise ValueError("logits_processors are not supported by TensorRT-LLM backend.")
        if params.logit_bias is not None or params.allowed_token_ids is not None:
            raise ValueError("logit_bias/allowed_token_ids are not supported by TensorRT-LLM backend.")
        if params.structured_outputs is not None:
            raise ValueError("structured_outputs are not supported by TensorRT-LLM backend.")
        if params.stop:
            raise ValueError("stop strings are not supported by TensorRT-LLM backend.")

        tokenizer = self.input_processor.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer unavailable for TensorRT-LLM backend.")
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("Tokenizer eos_token_id is required.")
        pad_id = tokenizer.pad_token_id or eos_id

        SamplingConfig, _ = _lazy_import_tensorrt_llm()
        if SamplingConfig is None:
            raise ModuleNotFoundError(
                "TensorRT-LLM backend requires the vendored `tensorrt_llm` package "
                "and its compiled bindings."
            ) from _TRTLLM_IMPORT_ERROR
        cfg = SamplingConfig(end_id=eos_id, pad_id=pad_id)
        cfg.max_new_tokens = params.max_tokens or 0
        cfg.top_k = params.top_k if params.top_k not in (0, -1) else 1
        cfg.top_p = params.top_p
        cfg.min_p = params.min_p
        cfg.temperature = params.temperature
        cfg.repetition_penalty = params.repetition_penalty
        cfg.presence_penalty = params.presence_penalty
        cfg.frequency_penalty = params.frequency_penalty
        if params.min_tokens:
            cfg.min_length = params.min_tokens
        if params.seed is not None:
            cfg.random_seed = params.seed

        stop_words = None
        if params.stop_token_ids:
            stop_words = [[ [token_id] for token_id in params.stop_token_ids ]]
        bad_words = None
        if params.bad_words:
            bad_words = [[ [token_id] for token_id in params.bad_words ]]

        return cfg, stop_words, bad_words
