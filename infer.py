"""Single-request streaming inference for Breeze TTS 2 (with Voice Profile Matrix Caching)."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import soundfile as sf
import torch

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

REPO_ROOT = Path(__file__).resolve().parent
FAST_CONFIG = REPO_ROOT / "configs" / "fast.json"
DEFAULT_CFG_SCALE = 1.0
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1


class CachedAudioTokenizer:
    """Transparent proxy around audio_tokenizer to cache and reuse extracted reference codes.
    
    Bypasses redundant neural audio codec encoding across chunks, ensuring:
      1. Immediate (near-zero latency) prompt preparation.
      2. Perfect acoustic timbre consistency across sentence chunks.
    """

    def __init__(
        self,
        real_audio_tokenizer,
        save_path: Path | None = None,
        load_path: Path | None = None,
        device=None,
    ):
        self._real = real_audio_tokenizer
        self._save_path = save_path
        self._load_path = load_path
        self._device = device

    def __getattr__(self, name):
        return getattr(self._real, name)

    def _intercept_result(self, res):
        if self._save_path is not None:
            try:
                self._save_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(res, torch.Tensor):
                    save_data = res.detach().cpu()
                elif isinstance(res, (tuple, list)):
                    save_data = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in res]
                elif isinstance(res, dict):
                    save_data = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in res.items()}
                else:
                    save_data = res
                torch.save(save_data, self._save_path)
                print(f"💾 [Voice Cache] Reference matrix successfully saved to: {self._save_path}")
            except Exception as e:
                print(f"⚠️ [Voice Cache] Warning: could not save cache file: {e}")
        return res

    def _load_cached(self):
        if self._load_path is not None and self._load_path.is_file():
            print(f"⚡ [Voice Cache] Bypassing audio encoder. Loading matrix directly from: {self._load_path}")
            cached_data = torch.load(self._load_path, map_location=self._device)
            return cached_data
        return None

    def encode(self, *args, **kwargs):
        cached = self._load_cached()
        if cached is not None:
            return cached
        res = self._real.encode(*args, **kwargs)
        return self._intercept_result(res)

    def encode_file(self, *args, **kwargs):
        cached = self._load_cached()
        if cached is not None:
            return cached
        if hasattr(self._real, "encode_file"):
            res = self._real.encode_file(*args, **kwargs)
        else:
            res = self._real.encode(*args, **kwargs)
        return self._intercept_result(res)

    def tokenize(self, *args, **kwargs):
        cached = self._load_cached()
        if cached is not None:
            return cached
        if hasattr(self._real, "tokenize"):
            res = self._real.tokenize(*args, **kwargs)
        else:
            res = self._real(*args, **kwargs)
        return self._intercept_result(res)

    def __call__(self, *args, **kwargs):
        cached = self._load_cached()
        if cached is not None:
            return cached
        res = self._real(*args, **kwargs)
        return self._intercept_result(res)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one WAV with Breeze TTS 2")
    parser.add_argument("model", type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction")
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text")
    # New arguments for Voice Profile Matrix Caching
    parser.add_argument(
        "--save-ref-codes",
        type=Path,
        help="Path to save extracted reference audio codes (.pt) for subsequent chunks",
    )
    parser.add_argument(
        "--ref-codes",
        type=Path,
        help="Path to pre-extracted reference audio codes (.pt) to bypass audio encoding",
    )
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument(
        "--fast-all", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--fast-text-encoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-prefill", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-backbone-decode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-depth-decoder", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fast-codec", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    if not math.isfinite(args.cfg_scale) or args.cfg_scale <= 0:
        raise ValueError("--cfg-scale must be greater than 0")

    has_ref_audio = (args.ref_audio is not None) or (args.ref_codes is not None)
    has_ref_text = bool(args.ref_text and args.ref_text.strip())
    if has_ref_audio != has_ref_text:
        raise ValueError("--ref-audio (or --ref-codes) and --ref-text must be provided together")
    if args.ref_audio is not None and not args.ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")
    if args.ref_codes is not None and args.ref_audio is None and not args.ref_codes.is_file():
        raise FileNotFoundError(f"Reference codes cache file not found: {args.ref_codes}")

    device = resolve_device()
    tokenizer, model, raw_audio_tokenizer = load_runtime(
        args.model,
        device=device,
        attn_implementation="eager",
    )
    update_generation_config_for_breeze(model)

    # Wrap audio_tokenizer with Voice Profile Matrix Caching proxy
    audio_tokenizer = CachedAudioTokenizer(
        raw_audio_tokenizer,
        save_path=args.save_ref_codes,
        load_path=args.ref_codes,
        device=device,
    )

    config = FastStreamingConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        max_seq_len=MAX_SEQ_LEN,
        fast_all=args.fast_all,
        fast_text_encoder=args.fast_text_encoder,
        fast_backbone_prefill=args.fast_backbone_prefill,
        fast_backbone_decode=args.fast_backbone_decode,
        fast_depth_decoder=args.fast_depth_decoder,
        fast_codec=args.fast_codec,
        repetition_penalty=REPETITION_PENALTY,
    )
    runtime = FastBreezeStreamingRuntime(
        model, audio_tokenizer, config, tokenizer=tokenizer
    )

    if runtime.fast_enabled:
        profile = load_warmup_profile(FAST_CONFIG)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        manifest = runtime.warmup_from_profile(profile)
        print(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms")

    instruction = args.instruction.strip() if args.instruction else None
    request = {
        "id": "single-request",
        "text": args.text,
        "speaker": "S0",
    }
    if instruction:
        request["instruction"] = instruction

    if args.ref_audio is not None:
        request["ref_audio_path"] = str(args.ref_audio)
        request["ref_text"] = args.ref_text.strip()
    elif args.ref_codes is not None:
        # Fallback path pointer when running purely from pre-extracted codes
        request["ref_audio_path"] = str(args.ref_codes)
        request["ref_text"] = args.ref_text.strip()

    template_name = select_template_name(request)

    set_all_seeds(args.seed)
    inputs = prepare_inputs(
        tokenizer,
        audio_tokenizer,
        model,
        [request],
        get_template(template_name),
        guidance_scale=args.cfg_scale,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(
        args.output,
        mode="w",
        samplerate=runtime.sample_rate,
        channels=1,
        subtype="PCM_16",
    ) as output_file:
        for chunk in runtime.iter_audio_chunks(
            inputs, request_id="single-request", seed=args.seed
        ):
            output_file.write(chunk.audio)

    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
