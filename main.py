import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM


MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-04-14"

OCR_PROMPT = (
    "Extract all visible text from this image. "
    "Return every readable word in reading order. "
    "Preserve line breaks where possible. "
    "Do not describe the image. "
    "Do not explain anything. "
    "Return only the visible text."
)


def format_time(seconds):
    seconds = int(seconds)

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class DigitalDocument:
    def __init__(
        self,
        model_id=MODEL_ID,
        revision=REVISION,
        device="cuda:0",
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = device

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but CUDA is not available."
                )

            gpu_index = (
                int(device.split(":")[1])
                if ":" in device
                else 0
            )

            print(
                f"GPU: {torch.cuda.get_device_name(gpu_index)}"
            )

        print(f"Loading model: {model_id}")

        kwargs = {
            "revision": revision,
            "trust_remote_code": True,
        }

        if device.startswith("cuda"):
            kwargs["device_map"] = {"": device}
            kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **kwargs,
        )

        self.model.eval()

        print("Model loaded.")
        print()

    def _load_image(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        return image

    def extract_text(
        self,
        image_path,
        prompt=OCR_PROMPT,
    ):
        image = self._load_image(image_path)

        with torch.inference_mode():
            result = self.model.query(
                image,
                prompt,
            )

        return result["answer"].strip()

    def extract_text_batch(
        self,
        image_paths,
        output_file="ocr_results.jsonl",
        prompt=OCR_PROMPT,
    ):
        image_paths = [
            str(Path(path))
            for path in image_paths
        ]

        output_file = Path(output_file)

        total = len(image_paths)

        if total == 0:
            print("No images supplied.")
            return []

        print(f"Images: {total:,}")
        print(f"Device: {self.device}")
        print(f"Output: {output_file}")
        print()

        start_time = time.time()

        success_count = 0
        error_count = 0
        saved_count = 0

        results = []

        with output_file.open(
            "a",
            encoding="utf-8",
        ) as fout:

            for index, image_path in enumerate(
                image_paths,
                start=1,
            ):
                image_start = time.time()

                try:
                    text = self.extract_text(
                        image_path,
                        prompt=prompt,
                    )

                    status = "success"
                    error = None

                    success_count += 1

                except Exception as e:
                    text = None
                    status = "error"
                    error = str(e)

                    error_count += 1

                image_elapsed = (
                    time.time()
                    - image_start
                )

                row = {
                    "image": image_path,
                    "text": text,
                    "status": status,
                    "error": error,
                    "elapsed": round(
                        image_elapsed,
                        3,
                    ),
                }

                # Save immediately
                fout.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                fout.flush()

                saved_count += 1

                results.append(row)

                elapsed = (
                    time.time()
                    - start_time
                )

                rate = (
                    index / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining = (
                    total - index
                )

                eta = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                if status == "success":
                    print(
                        f"[{index:,}/{total:,}] "
                        f"OK | "
                        f"{image_path} | "
                        f"chars={len(text):,} | "
                        f"saved={saved_count:,} | "
                        f"{image_elapsed:.2f}s | "
                        f"rate={rate:.2f} img/s | "
                        f"ETA={format_time(eta)}"
                    )

                else:
                    print(
                        f"[{index:,}/{total:,}] "
                        f"ERROR | "
                        f"{image_path} | "
                        f"saved={saved_count:,} | "
                        f"{error}"
                    )

        total_elapsed = (
            time.time()
            - start_time
        )

        print()
        print("DONE")
        print(f"Processed: {total:,}")
        print(f"Success:   {success_count:,}")
        print(f"Errors:    {error_count:,}")
        print(f"Saved:     {saved_count:,}")
        print(
            f"Elapsed:   "
            f"{format_time(total_elapsed)}"
        )

        if total_elapsed > 0:
            print(
                f"Rate:      "
                f"{total / total_elapsed:.2f} images/s"
            )

        print(f"Output:    {output_file}")

        return results
