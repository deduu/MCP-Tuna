from datetime import time
from app.generation.registry import get_registry
from agentsoul.providers.hf import HuggingFaceProvider
import torch
import logging
import asyncio
import time

logger = logging.getLogger(__name__)

# Public UI names
QWEN_BASE_NAME = "Qwen/Qwen3-32B"
QWEN_AUTO_NAME = "Auto"
QWEN_BASIC_NAME = "Instant"
QWEN_REASONING_NAME = "Thinking"
QWEN_FT_NAME = "Fine-Tuned"

# Hidden cache keys (one per family)
QWEN3_BASE_FAMILY = "Auto"
QWEN3_FT_FAMILY = "Fine-Tuned"

# LoRA path
QWEN_LORA_PATH = "models/qwen3_trl_sft_finetuned_model"

# Sticky memory: last loaded family
_current = {"key": None, "family": None}  # family: "base" | "fine_tuned"

# Serialize switches
_switch_lock = asyncio.Lock()

_load_events = {
    "base": asyncio.Event(),
    "fine_tuned": asyncio.Event(),
}
# Initially **idle / not loading**
_load_events["base"].set()
_load_events["fine_tuned"].set()

# Families
BASE_FAMILY_NAMES = {QWEN_BASE_NAME, QWEN_AUTO_NAME,
                     QWEN_BASIC_NAME, QWEN_REASONING_NAME}
FT_FAMILY_NAMES = {QWEN_FT_NAME}


def _model_family(model_name: str) -> tuple[str, str]:
    """Return (family, cache_key)."""
    if model_name == QWEN_FT_NAME:
        return "fine_tuned", QWEN3_FT_FAMILY
    return "base", QWEN3_BASE_FAMILY


def _family_busy(family: str) -> bool:
    reg = get_registry()
    for name in _family_aliases(family):
        inst = reg.get_cached(name)
        if inst is not None and getattr(inst, "_busy", 0) > 0:
            return True
    return False


def _ensure_factory_for_family(family: str) -> None:
    reg = get_registry()
    key = QWEN3_FT_FAMILY if family == "fine_tuned" else QWEN3_BASE_FAMILY
    if key in reg.factories:
        return

    # IMPORTANT: wrap construction so we only set the event AFTER model is actually built
    def _factory(api_key=None):
        _load_events[family].clear()  # mark "loading"
        try:
            if family == "fine_tuned":
                prov = HuggingFaceProvider(
                    QWEN_BASE_NAME,
                    lora_adapt_path=QWEN_LORA_PATH,
                    device=["cuda:0", "cuda:1"],
                    dtype="float16",
                    quantization="bitsandbytes",
                    system_prompt="You are a helpful assistant.",
                )
            else:
                prov = HuggingFaceProvider(
                    QWEN_BASE_NAME,
                    lora_adapt_path=None,
                    device=["cuda:0", "cuda:1"],
                    dtype="float16",
                    quantization="bitsandbytes",
                    system_prompt="You are a helpful assistant.",
                )
            return prov
        finally:
            # construction finished successfully (or failed): release waiters
            _load_events[family].set()

    reg.set_factory(key, _factory)


def _family_aliases(family: str) -> set[str]:
    if family == "fine_tuned":
        return FT_FAMILY_NAMES
    elif family == "base":
        return BASE_FAMILY_NAMES
    return set()


async def _evict_family(family: str, wait_idle: bool = True) -> None:
    reg = get_registry()
    print(
        f"[qwen3] evicting family {family}: aliases={_family_aliases(family)}")
    for name in _family_aliases(family):
        try:
            await reg.evict(name, wait_idle=wait_idle, timeout=300.0)
        except TypeError:
            await reg.evict(name)  # older signature
        except Exception:
            pass

MAX_SWITCH_WAIT = 30.0


async def route_qwen3(model_name: str, model_route: dict):
    """
    Decide which Qwen3 variant to use and ensure the correct factory is installed.

    Returns:
        selected_key (str): the cache key to request from the registry
        enable_thinking (bool): True for agentic, False for basic
    """

    # Decide requested family from UI name
    requested_family, cache_key = _model_family(model_name)

    # Agentic/basic are **modes** on the base family; never trigger reloads
    # Only the fine-tuned name selects the FT family.

    reg = get_registry()

    async with _switch_lock:
        inst = reg.get_cached(cache_key)
        print(
            f"[qwen3] requested_family={requested_family}, cache_key={cache_key}, current={_current}, inst={inst}")

        # FAST PATH A: requested family already active AND instance exists
        if inst is not None:
            print(
                f"[qwen3] cached instance found using FAST PATH A for {cache_key}")
            _current["key"], _current["family"] = cache_key, requested_family
            logger.debug(
                f"[qwen3] reuse {cache_key} (family={requested_family})")
            return cache_key

        # ---- FAST PATH B: model_route.get("route") == "follow_up_and_chat_history"
        if inst is not None and model_route.get("route") == "follow_up_and_chat_history":
            # same model name AND follow_up_and_chat_history route → keep it; no unload/reload
            print(
                f"[qwen3] Reusing cached FAST PATH B {_current['key']} (follow_up_and_chat_history)")
            # _current["key"], _current["variant"] = cache_key, requested_family
            return _current['key']

        # SLOW PATH: we need to load this family
        # If the *other* family is busy, wait bounded time before switching

        # --- A) Queue with timeout (recommended)
        other_family = "fine_tuned" if requested_family == "base" else "base"
        print(
            f"[qwen3] requested family={requested_family}, other_family={other_family}")

        wait_deadline = time.monotonic() + MAX_SWITCH_WAIT
        # Wait while the other family is either RUNNING (_busy>0) or LOADING (event not set)
        print(
            f"[qwen3] waiting for other family to be free/built for {MAX_SWITCH_WAIT} seconds...")
        while (_family_busy(other_family) or not _load_events[other_family].is_set()):
            if time.monotonic() > wait_deadline:
                running_model = reg.get_cached(_current["key"])
                other_family_busy = _family_busy(other_family)
                logger.warning(
                    f"[qwen3] timed out waiting for other family to be free/built; current={_current}, running_model={running_model}, other_family_busy={other_family_busy}")
                if _current['family'] == "base":
                    free_model_name = [QWEN_REASONING_NAME,
                                       QWEN_BASIC_NAME, QWEN_AUTO_NAME]
                else:
                    free_model_name = [QWEN_FT_NAME]
                raise RuntimeError(
                    f"GPU busy running {_current['family']} Qwen3 model. Try again in a moment or use {free_model_name}.")
            logger.info(
                f"[qwen3] waiting other family to finish (busy={_family_busy(other_family)}, loading={not _load_events[other_family].is_set()})")
            await asyncio.sleep(1)

        # --- B) Fail fast immediately (alternative policy)
        # if _family_busy(other_family):
        #     raise RuntimeError("GPU busy: another Qwen job is running. Try again in a moment.")

        # Evict only the other family; never evict our target key
        await _evict_family(other_family, wait_idle=True)

        # Install factory for requested family once
        _ensure_factory_for_family(requested_family)

        _current["key"], _current["family"] = cache_key, requested_family

        # quick allocator log
        for i in range(torch.cuda.device_count()):
            a = torch.cuda.memory_allocated(i)/1024**2
            r = torch.cuda.memory_reserved(i)/1024**2
            logger.info(f"[post-evict] GPU{i} alloc={a:.1f}MB resv={r:.1f}MB")
            print(
                f"[post-evict] GPU{i} allocated={a:.1f}MB reserved={r:.1f}MB")

        return cache_key
