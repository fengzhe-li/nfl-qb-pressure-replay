"""Real Vision-Language Model (VLM) inference module for QB Pressure Replay.

Integrates `Qwen/Qwen2.5-VL-3B-Instruct` via PyTorch / Hugging Face Transformers
(supporting Apple Silicon MPS, CUDA, and CPU execution) with strict visual-only
inputs and graceful fallback when the model or runtime dependencies are unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
import json
import re
import time

from PIL import Image

from vlm_evaluation import (
    validate_and_parse_temporal_vlm_json,
    validate_and_parse_vlm_json,
)
from vlm_frame_generator import FORBIDDEN_ENTITY_KEYS, FrameSequenceBundle

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_QUESTION = (
    "Based only on these sequential football replay frames, identify which visible "
    "pass rusher approaches the quarterback most closely, describe the movement "
    "pattern, and indicate any uncertainty."
)
DEFAULT_TEMPORAL_QUESTION = (
    "Based only on these sequential football replay frames, describe how the pass rushers "
    "approach the quarterback over time across Frames 1 to 4, identify the closest pass rusher, "
    "classify the rusher's distance trend and timing of closest approach, and separate direct "
    "visual observations from football interpretations that cannot be verified from 2D dots."
)


def detect_inference_device() -> str:
    """Detect the best available PyTorch execution device ('cuda', 'mps', or 'cpu')."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "unavailable"


def build_vlm_prompt(
    candidate_rushers: Sequence[Dict[str, Any]],
    question: str = DEFAULT_QUESTION,
    sampled_times: Optional[Sequence[float]] = None,
) -> str:
    """Construct the text prompt for the VLM using only visual identity metadata.

    Verifies that no ground-truth distances or PFF labels are present in `candidate_rushers`.
    """
    for r in candidate_rushers:
        for key in r:
            if key in FORBIDDEN_ENTITY_KEYS:
                raise ValueError(
                    f"Ground-truth leakage prevented: forbidden key '{key}' in candidate_rushers."
                )

    rusher_lines = []
    for r in candidate_rushers:
        num_str = f"#{r['number']} " if r.get("number") else ""
        rusher_lines.append(
            f"- {r['tag']}: {num_str}{r['name']} (nflId: {r['id']}, ring color: {r.get('color', 'colored')})"
        )
    rushers_block = "\n".join(rusher_lines)

    timing_line = ""
    if sampled_times:
        formatted_times = ", ".join(f"Frame {i+1}: +{t:.1f}s" for i, t in enumerate(sampled_times))
        timing_line = f"\nSequence frame times after snap: {formatted_times}.\n"

    prompt = (
        f"{question}\n\n"
        "Visual Legend & Field Conventions:\n"
        "- The offense attacks from left to right (+x downfield). The gold vertical line is the Line of Scrimmage.\n"
        "- The Quarterback is the bright YELLOW circle labeled 'QB'.\n"
        "- Offensive blockers/receivers are BLUE circles.\n"
        "- Pass rushers are RED circles highlighted with distinct colored rings and text labels (R1, R2, ...), "
        "with matching colored movement trails showing their trajectory up to each frame.\n"
        f"{timing_line}\n"
        "Visible Candidate Pass Rushers (listed in ID order, NOT by distance):\n"
        f"{rushers_block}\n\n"
        "Instructions:\n"
        "1. Examine all sequential frames carefully, especially the later frames as rushers converge on the yellow 'QB' circle.\n"
        "2. Determine which candidate pass rusher (R1, R2, ...) reaches the shortest visual distance to the yellow 'QB' circle at any point in the shown frames.\n"
        "3. If the frames are genuinely too ambiguous to tell which rusher is closest, set \"abstained\": true and \"selected_rusher_id\": null.\n"
        "4. Respond with ONLY a valid JSON object matching this exact schema (no markdown commentary outside the JSON):\n"
        "{\n"
        "  \"selected_rusher_id\": <integer nflId from the list above, or null if abstaining>,\n"
        "  \"selected_rusher_tag\": \"<R1/R2/... from the list above, or null if abstaining>\",\n"
        "  \"selected_rusher_name\": \"<full player name from the list above, or null if abstaining>\",\n"
        "  \"abstained\": <false or true>,\n"
        "  \"movement_pattern\": \"<1 sentence describing the rusher's trajectory and pocket movement across the frames>\",\n"
        "  \"visible_evidence\": \"<1 sentence citing the specific frames, marker labels, and position relative to the yellow QB marker>\",\n"
        "  \"uncertainty\": \"<1 sentence noting any close competing rushers, crowding, or 'Low uncertainty - clear closest approach'>\",\n"
        "  \"explanation\": \"<concise 1-2 sentence natural-language summary of the visual analysis>\"\n"
        "}"
    )
    return prompt


def build_temporal_vlm_prompt(
    candidate_rushers: Sequence[Dict[str, Any]],
    question: str = DEFAULT_TEMPORAL_QUESTION,
    sampled_times: Optional[Sequence[float]] = None,
) -> str:
    """Construct the temporal pass-rush analysis prompt for the VLM.

    Verifies that no ground-truth distances or PFF labels are present in `candidate_rushers`.
    """
    for r in candidate_rushers:
        for key in r:
            if key in FORBIDDEN_ENTITY_KEYS:
                raise ValueError(
                    f"Ground-truth leakage prevented: forbidden key '{key}' in candidate_rushers."
                )

    rusher_lines = []
    for r in candidate_rushers:
        num_str = f"#{r['number']} " if r.get("number") else ""
        rusher_lines.append(
            f"- {r['tag']}: {num_str}{r['name']} (nflId: {r['id']}, ring color: {r.get('color', 'colored')})"
        )
    rushers_block = "\n".join(rusher_lines)

    num_frames = len(sampled_times) if sampled_times else 4
    timing_line = ""
    if sampled_times:
        formatted_times = ", ".join(
            f"Frame {i+1}: +{t:.1f}s" for i, t in enumerate(sampled_times)
        )
        timing_line = f"\nSequence frame times after snap: {formatted_times}.\n"

    prompt = (
        f"{question}\n\n"
        "Visual Legend & Field Conventions:\n"
        "- The offense attacks from left to right (+x downfield). The gold vertical line is the Line of Scrimmage.\n"
        "- The Quarterback is the bright YELLOW circle labeled 'QB'.\n"
        "- Offensive blockers/receivers are BLUE circles.\n"
        "- Pass rushers are RED circles highlighted with distinct colored rings and callout boxes (R1, R2, ...), "
        "with matching colored trails showing their movement trajectory up to each frame.\n"
        f"{timing_line}\n"
        "Visible Candidate Pass Rushers (listed in ID order, NOT by distance; inspect all candidates R1..Rn equally):\n"
        f"{rushers_block}\n\n"
        "Instructions:\n"
        f"1. Track how each candidate pass rusher (R1..R{len(candidate_rushers)}) moves relative to the yellow 'QB' circle across Frame 1 through Frame {num_frames}.\n"
        "2. Identify which candidate pass rusher reaches the shortest visual Euclidean distance to the yellow 'QB' circle at any point in the sequence (including early, mid, or late frames).\n"
        "3. Classify the selected rusher's QB-distance trend across the sequence into one of:\n"
        "   - \"closing\": distance to the QB decreases toward the end of the sequence\n"
        "   - \"close_then_recede\": rusher approaches close to the QB in an earlier/middle frame and then widens away\n"
        "   - \"steady\": distance to the QB stays roughly constant\n"
        "   - \"receding\": distance to the QB increases across the sequence\n"
        "4. Identify when the closest approach occurs: `closest_approach_phase` (\"early\" for Frame 1, \"mid\" for middle frames, \"late\" for final frame) and `closest_frame_number` (integer 1 to "
        f"{num_frames}).\n"
        "5. Clearly separate direct 2D visual observations (`movement_pattern`, `visible_evidence`) from football interpretations that cannot be verified from 2D dots (`unverified_interpretation`, e.g., blocker technique, stunt coaching intent, or PFF pressure grade).\n"
        "6. Respond with ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        "  \"selected_rusher_id\": <integer nflId from the list above, or null if abstaining>,\n"
        "  \"selected_rusher_tag\": \"<R1/R2/... from the list above, or null if abstaining>\",\n"
        "  \"selected_rusher_name\": \"<full player name from the list above, or null if abstaining>\",\n"
        "  \"abstained\": <false or true>,\n"
        "  \"distance_trend\": \"<closing | close_then_recede | steady | receding>\",\n"
        "  \"closest_approach_phase\": \"<early | mid | late>\",\n"
        f"  \"closest_frame_number\": <integer 1..{num_frames}>,\n"
        "  \"collective_rush_description\": \"<1 sentence describing how all pass rushers move relative to the pocket over time>\",\n"
        "  \"movement_pattern\": \"<1 sentence describing the closest rusher's spatial trajectory across the frames>\",\n"
        "  \"visible_evidence\": \"<1 sentence citing specific frame numbers 1..4 and marker positions relative to the yellow QB marker>\",\n"
        "  \"unverified_interpretation\": \"<1 sentence stating any football interpretation (blocking leverage, scheme, pressure impact) that cannot be proven from 2D coordinates alone>\",\n"
        "  \"uncertainty\": \"<1 sentence noting competing rushers or visual ambiguity>\",\n"
        "  \"explanation\": \"<concise 1-2 sentence summary of the temporal and spatial pass-rush analysis>\"\n"
        "}"
    )
    return prompt


class QwenVLMAnalyzer:
    """Wrapper for genuine pretrained VLM inference with `Qwen/Qwen2.5-VL-3B-Instruct`."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        max_new_tokens: int = 240,
        custom_generate_fn: Optional[Callable[[Sequence[Image.Image], str], str]] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or detect_inference_device()
        self.max_new_tokens = max_new_tokens
        self.custom_generate_fn = custom_generate_fn
        self._model: Any = None
        self._processor: Any = None
        self._model_revision: str = "unknown"
        self._load_error: Optional[str] = None

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def is_available(self) -> bool:
        if self.custom_generate_fn is not None:
            return True
        return self._load_error is None and self.device != "unavailable"

    def _resolve_snapshot_revision(self, local_path_or_id: str) -> str:
        """Extract the git commit SHA revision from Hugging Face cache or config if available."""
        try:
            p = Path(local_path_or_id)
            if p.exists():
                for parent in [p] + list(p.parents):
                    if parent.parent.name == "snapshots":
                        return parent.name
        except Exception:
            pass
        if self._model is not None and hasattr(self._model, "config"):
            sha = getattr(self._model.config, "_commit_hash", None)
            if sha:
                return str(sha)
        try:
            from huggingface_hub import try_to_load_from_cache

            cached_cfg = try_to_load_from_cache(self.model_id, "config.json")
            if isinstance(cached_cfg, str):
                p = Path(cached_cfg)
                for parent in p.parents:
                    if parent.parent.name == "snapshots":
                        return parent.name
        except Exception:
            pass
        return "unknown"

    def ensure_loaded(self) -> bool:
        """Lazily load the pretrained VLM checkpoint and processor onto the target device."""
        if self.custom_generate_fn is not None:
            self._model_revision = "custom-fn"
            return True
        if self._model is not None and self._processor is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            dtype = (
                torch.bfloat16
                if self.device in {"mps", "cuda"}
                else torch.float32
            )
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=dtype,
                low_cpu_mem_usage=True,
            )
            if self.device in {"mps", "cuda"}:
                self._model = self._model.to(self.device)
            self._model.eval()
            self._model_revision = self._resolve_snapshot_revision(self.model_id)
            return True
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return False

    def generate_raw_response(
        self,
        images: Sequence[Image.Image],
        prompt_text: str,
    ) -> str:
        """Run forward pass on the sequence of PIL images and prompt text."""
        if not images:
            raise ValueError("At least one replay frame image is required for VLM inference.")
        if self.custom_generate_fn is not None:
            return self.custom_generate_fn(images, prompt_text)

        if not self.ensure_loaded():
            raise RuntimeError(self._load_error or "VLM model could not be loaded.")

        import torch

        # Ensure dimensions are multiples of 28 and bounded for responsive multi-frame ViT attention
        processed_images: List[Image.Image] = []
        for img in images:
            if img.width > 448 or img.height > 336:
                processed_images.append(img.resize((448, 336), Image.Resampling.LANCZOS))
            else:
                processed_images.append(img)

        content: List[Dict[str, Any]] = []
        for img in processed_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert football tracking vision analyst. "
                    "You analyze sequential 2D overhead replay frames and output strict JSON."
                ),
            },
            {"role": "user", "content": content},
        ]

        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[chat_text],
            images=processed_images,
            padding=True,
            return_tensors="pt",
        )
        if self.device in {"mps", "cuda"}:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self._processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if self.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        return output_texts[0] if output_texts else ""

    def analyze_frame_bundle(
        self,
        bundle: FrameSequenceBundle,
        question: str = DEFAULT_QUESTION,
    ) -> Dict[str, Any]:
        """Run VLM inference on a `FrameSequenceBundle` and return validated structured output."""
        if not isinstance(bundle, FrameSequenceBundle) or not bundle.images:
            raise ValueError("A FrameSequenceBundle with non-empty images is required.")

        prompt_text = build_vlm_prompt(
            candidate_rushers=bundle.candidate_rushers,
            question=question,
            sampled_times=bundle.sampled_times,
        )

        t0 = time.perf_counter()
        if not self.ensure_loaded():
            latency = time.perf_counter() - t0
            return {
                "is_valid_json": False,
                "status": "model_unavailable",
                "abstained": False,
                "selected_rusher_id": None,
                "selected_rusher_name": None,
                "selected_rusher_tag": None,
                "movement_pattern": "",
                "visible_evidence": "",
                "uncertainty": f"Model unavailable ({self._load_error or 'not loaded'}).",
                "explanation": "AI visual analysis is unavailable; tracking-data replay remains active.",
                "raw_text": "",
                "latency_seconds": round(latency, 4),
                "model_name": self.model_id,
                "model_revision": self._model_revision,
                "device": self.device,
                "prompt": prompt_text,
            }

        try:
            raw_text = self.generate_raw_response(bundle.images, prompt_text)
            latency = time.perf_counter() - t0
        except Exception as exc:
            latency = time.perf_counter() - t0
            return {
                "is_valid_json": False,
                "status": "model_unavailable",
                "abstained": False,
                "selected_rusher_id": None,
                "selected_rusher_name": None,
                "selected_rusher_tag": None,
                "movement_pattern": "",
                "visible_evidence": "",
                "uncertainty": f"Inference runtime error: {type(exc).__name__}: {exc}",
                "explanation": "AI visual inference encountered a runtime error; tracking-data replay remains active.",
                "raw_text": "",
                "latency_seconds": round(latency, 4),
                "model_name": self.model_id,
                "model_revision": self._model_revision,
                "device": self.device,
                "prompt": prompt_text,
            }

        parsed = validate_and_parse_vlm_json(
            raw_output=raw_text,
            candidate_rushers=bundle.candidate_rushers,
        )
        parsed["latency_seconds"] = round(latency, 4)
        parsed["model_name"] = self.model_id
        parsed["model_revision"] = self._model_revision
        parsed["device"] = self.device
        parsed["prompt"] = prompt_text
        return parsed

    def analyze_temporal_frame_bundle(
        self,
        bundle: FrameSequenceBundle,
        question: str = DEFAULT_TEMPORAL_QUESTION,
    ) -> Dict[str, Any]:
        """Run temporal pass-rush VLM inference on a `FrameSequenceBundle`."""
        if not isinstance(bundle, FrameSequenceBundle) or not bundle.images:
            raise ValueError("A FrameSequenceBundle with non-empty images is required.")

        prompt_text = build_temporal_vlm_prompt(
            candidate_rushers=bundle.candidate_rushers,
            question=question,
            sampled_times=bundle.sampled_times,
        )

        t0 = time.perf_counter()
        if not self.ensure_loaded():
            latency = time.perf_counter() - t0
            return {
                "is_valid_json": False,
                "status": "model_unavailable",
                "abstained": False,
                "selected_rusher_id": None,
                "selected_rusher_name": None,
                "selected_rusher_tag": None,
                "distance_trend": None,
                "closest_approach_phase": None,
                "closest_frame_number": None,
                "collective_rush_description": "",
                "movement_pattern": "",
                "visible_evidence": "",
                "unverified_interpretation": "",
                "uncertainty": f"Model unavailable ({self._load_error or 'not loaded'}).",
                "explanation": "AI temporal visual analysis is unavailable; tracking-data replay remains active.",
                "raw_text": "",
                "latency_seconds": round(latency, 4),
                "model_name": self.model_id,
                "model_revision": self._model_revision,
                "device": self.device,
                "prompt": prompt_text,
            }

        prev_tokens = self.max_new_tokens
        if self.max_new_tokens < 450:
            self.max_new_tokens = 450
        try:
            raw_text = self.generate_raw_response(bundle.images, prompt_text)
            latency = time.perf_counter() - t0
        except Exception as exc:
            latency = time.perf_counter() - t0
            self.max_new_tokens = prev_tokens
            return {
                "is_valid_json": False,
                "status": "model_unavailable",
                "abstained": False,
                "selected_rusher_id": None,
                "selected_rusher_name": None,
                "selected_rusher_tag": None,
                "distance_trend": None,
                "closest_approach_phase": None,
                "closest_frame_number": None,
                "collective_rush_description": "",
                "movement_pattern": "",
                "visible_evidence": "",
                "unverified_interpretation": "",
                "uncertainty": f"Inference runtime error: {type(exc).__name__}: {exc}",
                "explanation": "AI temporal inference encountered a runtime error; tracking-data replay remains active.",
                "raw_text": "",
                "latency_seconds": round(latency, 4),
                "model_name": self.model_id,
                "model_revision": self._model_revision,
                "device": self.device,
                "prompt": prompt_text,
            }
        finally:
            self.max_new_tokens = prev_tokens

        parsed = validate_and_parse_temporal_vlm_json(
            raw_output=raw_text,
            candidate_rushers=bundle.candidate_rushers,
            num_sampled_frames=len(bundle.images),
        )
        parsed["latency_seconds"] = round(latency, 4)
        parsed["model_name"] = self.model_id
        parsed["model_revision"] = self._model_revision
        parsed["device"] = self.device
        parsed["prompt"] = prompt_text
        return parsed

