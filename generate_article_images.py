#!/usr/bin/env python3
"""
Generate images from {IMAGE_PROMPT: ...} blocks in markdown articles and insert them back into the article.

Default behavior:
- Reads *.md from ./articles
- Finds {IMAGE_PROMPT: ...} blocks (supports multiline inside braces)
- Generates PNGs into ./articles/_images/<article-slug>/img-01.png, img-02.png, ...
- Replaces each {IMAGE_PROMPT: ...} block with:
    <!-- IMAGE_PROMPT: ... -->
    ![Generated image](./_images/<article-slug>/img-01.png)

API key:
- Set env var GOOGLE_API_KEY (recommended) or GENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from google import genai
from PIL import Image


IMAGE_PROMPT_RE = re.compile(r"\{IMAGE_PROMPT:\s*(.*?)\}", re.DOTALL)


@dataclass(frozen=True)
class PromptMatch:
    prompt_raw: str
    start: int
    end: int


def _slug_from_article_path(article_path: Path) -> str:
    return article_path.stem


def _compact_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _alt_from_prompt(prompt: str, max_len: int = 80) -> str:
    # Keep it simple and CMS-friendly: a short, cleaned snippet.
    cleaned = _compact_ws(prompt)
    cleaned = re.sub(r"(?i)\bno text overlays\b", "", cleaned).strip(" ,.-")
    if not cleaned:
        return "Generated image"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def _read_prompts(md: str) -> list[PromptMatch]:
    matches: list[PromptMatch] = []
    for m in IMAGE_PROMPT_RE.finditer(md):
        matches.append(PromptMatch(prompt_raw=m.group(1), start=m.start(), end=m.end()))
    return matches


def _api_key() -> str | None:
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GENAI_API_KEY")


def _client() -> genai.Client:
    key = _api_key()
    if not key:
        raise RuntimeError(
            "Missing API key. Set env var GOOGLE_API_KEY (preferred) or GENAI_API_KEY."
        )
    return genai.Client(api_key=key)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(_compact_ws(prompt).encode("utf-8")).hexdigest()[:12]


def _load_robot_rules(rules_path: Path) -> dict:
    """Load robot reference rules from JSON."""
    if not rules_path.exists():
        # Return sensible defaults if file missing
        return {
            "robot_folder_keywords": {
                "juno": ["warehouse", "logistics", "logistik", "lager", "fts", "agv", "amr"],
                "pollux": ["pallet", "palette", "paletten"],
                "mars": ["gastronomie", "restaurant", "gastro", "kellner", "hotel"],
                "vesta": ["gastronomie", "restaurant", "gastro", "kellner", "hotel"],
                "base": ["base", "platform"],
            },
            "scenario_keywords": {
                "krankenhaus": ["krankenhaus", "hospital"],
                "gastro": ["gastro", "restaurant"],
                "logistik": ["logistik", "logistics", "lager"],
            },
            "fallback_robot": "base",
        }
    with rules_path.open(encoding="utf-8") as f:
        return json.load(f)


def _select_robot_folder(
    article_slug: str, prompt: str, rules: dict, default_robot: str
) -> str:
    """Select robot folder based on article slug + prompt keywords."""
    text_to_match = (article_slug + " " + prompt).lower()
    robot_keywords = rules.get("robot_folder_keywords", {})
    
    # Score each robot by keyword matches
    scores: dict[str, int] = {}
    for robot, keywords in robot_keywords.items():
        score = sum(1 for kw in keywords if kw.lower() in text_to_match)
        if score > 0:
            scores[robot] = score
    
    if scores:
        # Return robot with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    # Fallback
    fallback = rules.get("fallback_robot", default_robot)
    return fallback


def _pick_best_reference_image(
    robot_folder: Path, prompt: str, rules: dict
) -> Path | None:
    """Pick the best reference image from robot folder based on scenario keywords."""
    if not robot_folder.exists():
        return None
    
    # Collect all image files
    image_exts = {".webp", ".png", ".jpg", ".jpeg"}
    candidates: list[Path] = [
        p for p in robot_folder.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    ]
    
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Score candidates by scenario keyword matches in filename
    prompt_lower = prompt.lower()
    scenario_keywords = rules.get("scenario_keywords", {})
    
    scores: dict[Path, int] = {}
    for candidate in candidates:
        filename_lower = candidate.stem.lower()
        score = 0
        # Check scenario keywords
        for scenario, keywords in scenario_keywords.items():
            if any(kw.lower() in filename_lower for kw in keywords):
                # Also boost if scenario keywords match prompt
                if any(kw.lower() in prompt_lower for kw in keywords):
                    score += 2
                else:
                    score += 1
        # Boost if keywords from filename appear in prompt
        filename_words = set(filename_lower.replace("_", " ").replace("-", " ").split())
        prompt_words = set(prompt_lower.split())
        common = filename_words & prompt_words
        score += len(common) * 0.5
        if score > 0:
            scores[candidate] = score
    
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    # Default: return first candidate
    return candidates[0]


def _load_reference_image(image_path: Path) -> Image.Image:
    """Load reference image using PIL."""
    try:
        return Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load reference image {image_path}: {e}") from e


def _generate_image_png(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    out_path: Path,
    reference_image: Image.Image | None = None,
) -> None:
    contents = [_compact_ws(prompt)]
    if reference_image is not None:
        contents.append(reference_image)
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
    )

    for part in getattr(response, "parts", []) or []:
        # Prefer image part
        if getattr(part, "inline_data", None) is not None:
            image = part.as_image()
            image.save(str(out_path))
            return

    # If no image part, surface any text for debugging.
    texts: list[str] = []
    for part in getattr(response, "parts", []) or []:
        t = getattr(part, "text", None)
        if t:
            texts.append(str(t).strip())
    extra = "\n".join(texts).strip()
    raise RuntimeError(
        "Model response contained no image data."
        + (f"\n\nModel text:\n{extra}" if extra else "")
    )


def _render_replacement(
    *,
    prompt: str,
    rel_image_path: str,
    keep_prompt_comment: bool,
) -> str:
    alt = _alt_from_prompt(prompt)
    lines: list[str] = []
    if keep_prompt_comment:
        lines.append(f"<!-- IMAGE_PROMPT: {_compact_ws(prompt)} -->")
    lines.append(f"![{alt}]({rel_image_path})")
    return "\n".join(lines)


def _replace_prompts_in_md(
    md: str,
    *,
    image_rel_paths: list[str],
    keep_prompt_comment: bool,
    reference_info: list[str] | None = None,
) -> str:
    idx = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal idx
        prompt = m.group(1)
        if idx >= len(image_rel_paths):
            raise RuntimeError("Internal error: replacement index out of range.")
        rep = _render_replacement(
            prompt=prompt,
            rel_image_path=image_rel_paths[idx],
            keep_prompt_comment=keep_prompt_comment,
        )
        # Add reference info if available
        if reference_info and idx < len(reference_info) and reference_info[idx]:
            rep = f"<!-- REFERENCE: {reference_info[idx]} -->\n{rep}"
        idx += 1
        # Ensure blank lines around embeds for CMS readability.
        return "\n\n" + rep + "\n\n"

    out = IMAGE_PROMPT_RE.sub(repl, md)
    return out


def _iter_md_files(articles_dir: Path) -> Iterable[Path]:
    for p in sorted(articles_dir.glob("*.md")):
        if p.is_file():
            yield p


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--articles-dir",
        default="articles",
        help="Directory containing markdown articles (default: articles)",
    )
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash-image",
        help='Image model name (default: "gemini-2.5-flash-image")',
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate images even if output files already exist",
    )
    ap.add_argument(
        "--no-write-md",
        action="store_true",
        help="Do not modify markdown files (still generates images)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call the API and do not write any files; just print planned actions",
    )
    ap.add_argument(
        "--no-prompt-comment",
        action="store_true",
        help="Do not keep the original prompt as an HTML comment above the image",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max total images to generate across all files (0 = no limit)",
    )
    ap.add_argument(
        "--product-images-dir",
        default="product-images",
        help="Directory containing robot reference images (default: product-images)",
    )
    ap.add_argument(
        "--robot-default",
        default="base",
        help="Default robot folder to use when no match found (default: base)",
    )
    ap.add_argument(
        "--rules-file",
        type=Path,
        default=None,
        help="Path to robot_reference_rules.json (default: scripts/robot_reference_rules.json)",
    )
    ap.add_argument(
        "--no-ref",
        action="store_true",
        help="Do not use reference images (generate without product photos)",
    )
    ap.add_argument(
        "--print-selection",
        action="store_true",
        help="Print which reference image was selected for each prompt",
    )

    args = ap.parse_args(argv)
    articles_dir = Path(args.articles_dir).resolve()
    if not articles_dir.exists():
        print(f"ERROR: articles dir not found: {articles_dir}", file=sys.stderr)
        return 2

    # Load robot reference rules
    if args.rules_file:
        rules_path = Path(args.rules_file).resolve()
    else:
        # Default: look for rules next to script
        script_dir = Path(__file__).parent
        rules_path = script_dir / "robot_reference_rules.json"
    rules = _load_robot_rules(rules_path)

    # Setup product images directory
    product_images_dir = Path(args.product_images_dir).resolve()
    if not args.no_ref and not product_images_dir.exists():
        print(
            f"WARNING: product-images dir not found: {product_images_dir}",
            file=sys.stderr,
        )
        print("  Images will be generated without reference photos.", file=sys.stderr)

    # Only instantiate client if we will actually call the API.
    client = None if args.dry_run else _client()

    total_images = 0
    touched_files = 0

    for article_path in _iter_md_files(articles_dir):
        md = article_path.read_text(encoding="utf-8")
        prompt_matches = _read_prompts(md)
        if not prompt_matches:
            continue

        slug = _slug_from_article_path(article_path)
        out_dir = articles_dir / "_images" / slug

        planned_rel_paths: list[str] = []
        planned_out_paths: list[Path] = []
        for i, pm in enumerate(prompt_matches, start=1):
            # Stable deterministic filename by index + short hash (helps keep images stable even if reordered).
            h = _hash_prompt(pm.prompt_raw)
            filename = f"img-{i:02d}-{h}.png"
            planned_out_paths.append(out_dir / filename)
            planned_rel_paths.append(f"./_images/{slug}/{filename}")

        # Select reference images for each prompt
        reference_images: list[Image.Image | None] = []
        reference_info: list[str] = []
        
        for pm in prompt_matches:
            ref_image: Image.Image | None = None
            ref_info = ""
            
            if not args.no_ref and product_images_dir.exists():
                # Select robot folder
                robot_folder_name = _select_robot_folder(
                    slug, pm.prompt_raw, rules, args.robot_default
                )
                robot_folder = product_images_dir / robot_folder_name
                
                # Pick best reference image
                ref_path = _pick_best_reference_image(robot_folder, pm.prompt_raw, rules)
                if ref_path and ref_path.exists():
                    ref_info = f"{robot_folder_name}/{ref_path.name}"
                    # Only load image if not in dry-run (we'll need it for generation)
                    if not args.dry_run:
                        ref_image = _load_reference_image(ref_path)
                    if args.print_selection:
                        print(
                            f"  [{slug}] Using reference: {ref_info}",
                            file=sys.stderr,
                        )
                elif args.print_selection:
                    print(
                        f"  [{slug}] No reference found in {robot_folder_name}, generating without reference",
                        file=sys.stderr,
                    )
            
            reference_images.append(ref_image)
            reference_info.append(ref_info)

        if args.dry_run:
            print(f"\n{article_path.name}: {len(prompt_matches)} prompt(s)")
            for pm, outp, ref_info in zip(
                prompt_matches, planned_out_paths, reference_info, strict=True
            ):
                ref_str = f" [REF: {ref_info}]" if ref_info else " [no ref]"
                print(
                    f"  - {outp.relative_to(articles_dir.parent)}{ref_str} :: {_compact_ws(pm.prompt_raw)[:120]}"
                )
            continue

        _ensure_dir(out_dir)

        # Generate images
        for pm, outp, ref_img in zip(
            prompt_matches, planned_out_paths, reference_images, strict=True
        ):
            if args.max_images and total_images >= args.max_images:
                break

            if outp.exists() and not args.overwrite:
                continue

            assert client is not None
            _generate_image_png(
                client,
                model=args.model,
                prompt=pm.prompt_raw,
                out_path=outp,
                reference_image=ref_img,
            )
            total_images += 1

        # Update markdown
        if not args.no_write_md:
            updated_md = _replace_prompts_in_md(
                md,
                image_rel_paths=planned_rel_paths,
                keep_prompt_comment=not bool(args.no_prompt_comment),
                reference_info=reference_info if args.print_selection else None,
            )
            if updated_md != md:
                article_path.write_text(updated_md, encoding="utf-8")
                touched_files += 1

    if args.dry_run:
        return 0

    print(f"Done. Generated {total_images} image(s). Updated {touched_files} markdown file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


