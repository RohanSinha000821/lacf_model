"""Bounded source-only checks for the cached-CLAP TF2 full-F1 run."""

import argparse
import gc
import json
import logging
import time
from pathlib import Path
import soundfile as sf
import torch
from transformers import ClapModel, ClapProcessor

from audio_deepfake_detection.data import resample_audio
from audio_deepfake_detection.experiments.tf2_frozen_clap.model_data import (
    CachedCLAPDataset, cache_identity, load_clap_split, make_loader,
)
from audio_deepfake_detection.lacf.diagnostic_cache import crop_intervals, projected_clap_audio
from audio_deepfake_detection.lacf.precision import keep_normalization_fp32
from audio_deepfake_detection.protocol import FOLDS, checkpoint_digest
from scripts.experiments.tf2_frozen_clap.train import (
    CACHE, FAMILY, ROOT, build_model, digest, recipe, verify_validation,
)
from scripts.token_fusion.train import check_guards, save_json
from scripts.token_fusion.train_adaptation import (
    adaptation_state, restore_adaptation, score_audio, train_audio_epoch,
)


def clap_parity(bank, execution, logger):
    """Check both kinds of crop and short-audio padding against the pinned encoder."""
    revision = execution["pretrained"]["clap_revision"]
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused", revision=revision, local_files_only=True)
    if processor.feature_extractor.to_dict() != execution["processors"]["clap"]:
        raise ValueError("CLAP processor differs from the source feature cache")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused", revision=revision,
                                     local_files_only=True, dtype=torch.float32)
    del model.text_model, model.text_projection
    keep_normalization_fp32(model)
    model.requires_grad_(False).cuda().eval()
    examples = []
    for (domain, part), split in bank.items():
        # Include the first short/long recording where available and partition boundaries.
        indices = {0, len(split.records)-1}
        for is_long in (False, True):
            match = next((i for i, row in enumerate(split.metadata)
                          if (row["frames"] > row["rate"]*10) == is_long), None)
            if match is not None:
                indices.add(match)
        for index in sorted(indices):
            row = split.metadata[index]
            intervals, mapping = crop_intervals(row["frames"], row["rate"], row["id"], training=part == "train")
            for view, physical in enumerate(mapping):
                start, length = intervals[physical]
                audio, rate = sf.read(split.records[index][0], start=start, frames=length,
                                      dtype="float32", always_2d=True)
                assert len(audio) == length and rate == row["rate"]
                wave = resample_audio(torch.from_numpy(audio).mean(1), rate, 48000)
                inputs = processor.feature_extractor([wave.numpy()], sampling_rate=48000,
                    padding="repeatpad", truncation="rand_trunc", return_tensors="pt")
                actual = torch.from_numpy(projected_clap_audio(model, dict(inputs), "cuda"))[0]
                cached = split.features[split.view_rows[index, view]]
                cosine = torch.nn.functional.cosine_similarity(actual[None], cached[None]).item()
                relative = ((actual-cached).norm()/cached.norm().clamp_min(1e-12)).item()
                row_result = {"id": row["id"], "view": view, "cosine": cosine,
                              "relative_l2": relative, "max_abs": (actual-cached).abs().max().item(),
                              "bit_exact": torch.equal(actual, cached)}
                examples.append(row_result)
                # The diagnostic cache used accelerated BF16 batches; report rounding explicitly.
                if cosine < .999 or relative > .02:
                    raise ValueError(f"Direct/cache CLAP discrepancy exceeds rounding check: {row_result}")
        logger.info("CLAP parity checked %s/%s", domain, part)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return {"passed": True, "checks": examples, "criterion": "cosine >=0.999 and relative L2 <=0.02",
            "limit": "Bounded source parity; accelerated-cache BF16 rounding is retained, not bit-exact equivalence"}


class RepeatedBatch:
    def __init__(self, batch, batch_size):
        self.batch, self.batch_size = batch, batch_size
        self.dataset = range(132)

    def __iter__(self):
        return iter([self.batch] * (132//self.batch_size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, default=FAMILY/"launch")
    parser.add_argument("--physical-batch", type=int, default=22)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-memory-budget-gib", type=float, default=10.0)
    args = parser.parse_args()
    args.seed, args.cache_root, args.data_root = 1234, CACHE, Path("/mnt/drive/audio-deepfake-cache")
    if args.physical_batch < 1 or 132 % args.physical_batch:
        parser.error("Physical batch must divide132")
    output = args.launch/"source_validation.json"
    if output.exists():
        raise FileExistsError("Preserve existing validation evidence")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger("tf2_frozen_clap_validation")
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)
    torch.set_num_threads(2)
    torch.backends.nnpack.set_flags(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_budget_gib*1024**3/torch.cuda.get_device_properties(0).total_memory)
    active_path = ROOT/"outputs/experiments/token_fusion/adaptation_restart/launch/source_guards.json"
    active = json.loads(active_path.read_text())
    check_guards({"code_sha256": active})
    free, total = torch.cuda.mem_get_info()
    if free < (args.gpu_memory_budget_gib+2)*1024**3:
        raise RuntimeError("Insufficient concurrent GPU headroom")
    manifest, execution = cache_identity(CACHE)
    config = recipe(args, manifest, execution)
    save_json(args.launch/"preparation.json", {"recipe": config, "recipe_sha256": digest(config)})
    bank = {}
    for domain in FOLDS["f1"]["sources"]:
        for part in ("train", "dev"):
            logger.info("Loading verified CLAP-only cache %s/%s", domain, part)
            bank[domain, part] = load_clap_split(CACHE, domain, part, manifest["splits"][f"{domain}/{part}"], args.data_root)
    parity = clap_parity(bank, execution, logger)
    save_json(args.launch/"clap_parity.json", parity)
    model, collator = build_model(execution, 1234, "cuda")
    fitting = [CachedCLAPDataset([bank[d, "train"]]) for d in FOLDS["f1"]["sources"]]
    combined = CachedCLAPDataset([bank[d, "train"] for d in FOLDS["f1"]["sources"]])
    # Actual longest source views exercise a complete accumulation group, with no shortened inputs.
    longest = []
    for split in (bank[d, "train"] for d in FOLDS["f1"]["sources"]):
        ds = CachedCLAPDataset([split])
        for i, row in enumerate(split.metadata):
            if row["frames"] >= row["rate"]*10:
                longest.append(ds[i, 0])
                if len(longest) >= args.physical_batch:
                    break
        if len(longest) >= args.physical_batch:
            break
    if len(longest) != args.physical_batch or any(len(x[0]) != 160000 for x in longest):
        raise ValueError("Missing representative full ten-second source batch")
    optimizer = model.optimizer()
    torch.cuda.reset_peak_memory_stats()
    stress = train_audio_epoch(model, RepeatedBatch(collator(longest), args.physical_batch), optimizer,
                               torch.device("cuda"), logger=logger, phase="10s source stress")
    loader = make_loader(combined, collator, batch_size=args.physical_batch, workers=args.workers,
                         seed=1234, domains=fitting)
    measured = train_audio_epoch(model, loader, optimizer, torch.device("cuda"), logger=logger,
                                 phase="real source timing", max_batches=3*(132//args.physical_batch))
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    # Fixed source development subset includes both partition boundaries in every domain.
    examples = []
    for domain in FOLDS["f1"]["sources"]:
        dataset = CachedCLAPDataset([bank[domain, "dev"]])
        indices = list(range(min(12, len(dataset)))) + list(range(max(12, len(dataset)-12), len(dataset)))
        examples.extend(dataset[index] for index in indices)
    dev_loader = torch.utils.data.DataLoader(examples, batch_size=args.physical_batch, collate_fn=collator)
    tick = time.monotonic()
    expected, labels = score_audio(model, dev_loader, torch.device("cuda"))
    dev_seconds = time.monotonic()-tick
    state = adaptation_state(model)
    del model, optimizer, loader
    gc.collect()
    torch.cuda.empty_cache()
    fresh, _ = build_model(execution, 1234, "cuda")
    restore_adaptation(fresh, state)
    actual, reloaded_labels = score_audio(fresh, dev_loader, torch.device("cuda"))
    if not torch.equal(actual, expected) or not torch.equal(labels, reloaded_labels):
        raise ValueError("Fresh pinned-base source checkpoint reload differs")
    check_guards({"code_sha256": active})
    report = {"passed": True, "target_access": False, "recipe_sha256": digest(config),
              "checks": dict.fromkeys(("longest_ten_second_forward_backward", "finite_gradients",
                  "direct_cached_clap_parity", "fresh_checkpoint_reload", "concurrent_memory_headroom"), True),
              "measurements": {"train_examples_per_second": measured["samples"]/measured["seconds"],
                  "source_dev_examples_per_second": len(examples)/dev_seconds,
                  "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                  "gpu_free_before_bytes": free, "gpu_total_bytes": total},
              "stress": stress, "real_source_timing": measured, "reload_examples": len(examples),
              "clap_parity_sha256": checkpoint_digest(args.launch/"clap_parity.json"),
              "validation_script_sha256": checkpoint_digest(Path(__file__)),
              "probe_weights_discarded": True, "active_source_guards_preserved": active,
              "limits": "Bounded source GPU checks, not full-source validation or a runtime guarantee; final training must complete full-source reload/calibration"}
    verify_validation_preview = args.launch/"validation_checked.json"
    save_json(verify_validation_preview, report)
    verify_validation(verify_validation_preview, config)
    verify_validation_preview.rename(output)
    logger.info("PASSED: %.2f train audio/s, %.2f dev audio/s, %.2f GiB allocated; final fit must start fresh",
                report["measurements"]["train_examples_per_second"], report["measurements"]["source_dev_examples_per_second"], allocated/1024**3)


if __name__ == "__main__":
    main()
