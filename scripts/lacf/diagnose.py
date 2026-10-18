"""Step 1 only: inspect preserved checkpoints and already-evaluated F1/F2 data."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from audio_deepfake_detection.lacf.data import LACFDataset, make_loader
from audio_deepfake_detection.lacf.evaluation import load_completed_model, load_development_cache
from audio_deepfake_detection.lacf.model import CONCEPTS, js_divergence
from audio_deepfake_detection.lacf.training import device_batch, verify_fold_completion
from audio_deepfake_detection.metrics import compute_eer, compute_auroc
from audio_deepfake_detection.protocol import (
    FOLDS, checkpoint_digest, load_score_manifest, read_dataset, utterance_id,
    verify_completed_run, verify_score_digest,
)
from scripts.lacf.train import development_cache_identity


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/lacf10s_optimized_20261009"
FIELDS = ("utterance_id", "dataset", "label", "p_w_spoof", "p_c_spoof", "d_js", "raw_score")


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def plotting():
    # Separate plotting dependencies; never replace the training environment's numpy.
    path = Path("/tmp/lacf_step1_plotting")
    if path.is_dir():
        sys.path.append(str(path))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def static_diagnostics(output):
    if (output / "static.json").exists():
        raise FileExistsError("Static diagnostic results already exist")
    plt = plotting()
    prototypes = {}
    hashes = {}
    for fold in ("f1", "f2"):
        run = RUNS / fold / "1234"
        completion = verify_fold_completion(run, fold)
        state = torch.load(run / "best.pt", map_location="cpu", weights_only=True, mmap=True)
        assert state["config"]["components"]["concepts"] == list(CONCEPTS)
        prototype = state["model_state_dict"]["prototypes"].float()
        assert prototype.shape == (8, 512)
        torch.testing.assert_close(prototype.norm(dim=-1), torch.ones(8), rtol=1e-6, atol=1e-6)
        matrix = (prototype @ prototype.T).numpy()
        prototypes[fold] = matrix.tolist()
        hashes["lacf_" + fold] = completion["best_checkpoint_sha256"]
        with (output / (fold + "_prototype_cosines.csv")).open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("concept", *CONCEPTS))
            writer.writerows((concept, *row) for concept, row in zip(CONCEPTS, matrix))
        del state
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for ax, (fold, values) in zip(axes, prototypes.items()):
        matrix = np.asarray(values)
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(8), [str(i + 1) for i in range(8)])
        ax.set_yticks(range(8), CONCEPTS, fontsize=8)
        ax.set_title(f"LACF {fold.upper()}: recorded prototype cosines")
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, label="Cosine similarity")
    fig.savefig(output / "prototype_cosines.png", dpi=160)
    plt.close(fig)
    alpha = {}
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    with (output / "wavlm_layer_weights.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("fold", "layer", "alpha"))
        for fold in ("f1", "f2", "f3"):
            run = ROOT / "outputs/wavlm_bs96" / fold / "1234"
            completion, digest = verify_completed_run(run, fold, 1234)
            manifest = load_score_manifest(run, fold, 1234, digest)
            report = json.loads((run / "metrics.json").read_text())
            assert report["checkpoint_sha256"] == manifest["checkpoint_sha256"] == digest
            state = torch.load(run / "best.pt", map_location="cpu", weights_only=True, mmap=True)
            assert state["epoch"] == completion["best_epoch"]
            weights = state["model_state_dict"]["layer_weights"].float()
            assert weights.shape == (13,) and torch.isfinite(weights).all()
            values = weights.softmax(0).tolist()
            alpha[fold] = values
            hashes["wavlm_" + fold] = digest
            writer.writerows((fold, i, value) for i, value in enumerate(values))
            ax.plot(range(13), values, marker="o", label=fold.upper())
            del state
    ax.set(xlabel="Representation: 0 = projected frontend; 1–12 = Transformer layers",
           ylabel="Learned softmax weight α", xticks=range(13), title="Selected WavLM-WA checkpoints")
    ax.legend()
    fig.savefig(output / "wavlm_layer_weights.png", dpi=180)
    plt.close(fig)
    save_json(output / "static.json", {"step": 1, "checkpoint_sha256": hashes,
        "concepts": list(CONCEPTS), "prototype_cosines": prototypes, "wavlm_alpha": alpha,
        "uniform_spoof_mass": 5 / 8, "uniform_bonafide_mass": 3 / 8,
        "scope": "CPU checkpoint inspection only; no encoder inference or retraining"})


def recorded_scores(run, domain, split, records, data_root, manifest):
    path = run / ("target_scores.csv" if split != "dev" else f"source_dev_{domain}.csv")
    verify_score_digest(path, manifest)
    expected = {utterance_id(p, domain, data_root): y for p, y in records}
    result = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["utterance_id"]
            if key in result or row["dataset"] != domain or expected.get(key) != int(row["label"]):
                raise ValueError("Original score IDs/labels differ from canonical diagnostic inputs")
            result[key] = float(row["raw_score"])
    if result.keys() != expected.keys():
        raise ValueError("Incomplete original score file")
    return result


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std(ddof=0)),
            "min": float(values.min()), "max": float(values.max()),
            "quantiles": dict(zip(("1%", "10%", "25%", "50%", "75%", "90%", "99%"),
                                  np.quantile(values, (.01, .1, .25, .5, .75, .9, .99)).tolist()))}


def summary(rows):
    labels = np.asarray([row[2] for row in rows])
    pw, pc, js, score = np.asarray([row[3:] for row in rows], dtype=np.float64).T
    return {"count": len(rows), "ranking": {name: {"eer": compute_eer(labels, values),
            "auroc": compute_auroc(labels, values)} for name, values in
            (("p_w_spoof", pw), ("p_c_spoof", pc), ("original_classifier", score))},
        "groups": {name: {"p_c_spoof": distribution(pc[labels == y]),
                           "p_c_bonafide": distribution(1 - pc[labels == y]),
                           "d_js": distribution(js[labels == y])}
                   for name, y in (("bona_fide", 0), ("spoof", 1))}}


@torch.inference_mode()
def diagnose_fold(fold, output, data_root, workers):
    directory = output / fold
    directory.mkdir()  # Fail rather than overwrite or silently resume an interrupted diagnostic.
    device = torch.device("cuda")
    model, collator, config, completion = load_completed_model(RUNS / fold / "1234", fold, device)
    run = RUNS / fold / "1234"
    manifest = load_score_manifest(run, fold, 1234, completion["best_checkpoint_sha256"])
    static = json.loads((output / "static.json").read_text())
    assert static["checkpoint_sha256"]["lacf_" + fold] == completion["best_checkpoint_sha256"]
    source, target, domains = [], [], {}
    for domain, split in [(d, "dev") for d in FOLDS[fold]["sources"]] + [(FOLDS[fold]["target"], FOLDS[fold]["target_split"])]:
        records = read_dataset(domain, split, data_root)
        original = recorded_scores(run, domain, split, records, data_root, manifest)
        rows = []
        path = directory / f"{domain}_{split}.csv"
        with path.with_suffix(".csv.tmp").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            if split == "dev":
                cache = load_development_cache(config["development_feature_caches"][domain], records,
                    {**development_cache_identity(config), "domain": domain}, device)
                audio = {k: v.to(device) for k, v in cache.features.items()}
                batches = ({k: v[i:i+1] for k, v in audio.items()} for i in range(len(records)))
            else:
                batches = make_loader([LACFDataset(records, training=False, segment_seconds=10)], collator,
                    training=False, seed=1234, batch_size=1, num_workers=workers, prefetch_factor=2)
            for i, batch in enumerate(batches):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if split == "dev":
                        result = model.classify_audio(batch)
                    else:
                        assert int(batch["labels"][0]) == records[i][1] and len(batch["labels"]) == 1
                        result = model(device_batch(batch, device))
                values = torch.stack((result["p_w"][0, 3:].sum(), result["p_c"][0, 3:].sum(),
                    js_divergence(result["p_w"], result["p_c"])[0], model.spoof_score(result)[0])).float().cpu().tolist()
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite diagnostic values")
                if (not all(0 <= value <= 1 + 1e-6 for value in values[:2])
                        or not -1e-6 <= values[2] <= np.log(2) + 1e-6):
                    raise ValueError("Probabilities or JS divergence are outside their mathematical range")
                key = utterance_id(records[i][0], domain, data_root)
                if values[-1] != original[key]:
                    raise ValueError(f"Diagnostic logit differs from preserved native score: {domain}/{key}")
                row = (key, domain, records[i][1], *values)
                writer.writerow(row)
                rows.append(row)
                if (i + 1) % 500 == 0:
                    print(f"Step 1 {fold} {domain}/{split}: {i+1:,}/{len(records):,}", flush=True)
            assert len(rows) == len(records)
        path.with_suffix(".csv.tmp").replace(path)
        domains[domain + "/" + split] = summary(rows)
        (source if split == "dev" else target).extend(rows)
        if split == "dev":
            del cache, audio, batches
        else:
            del batches  # Shut down the completed target loader's workers.
    result = {"step": 1, "fold": fold, "checkpoint_sha256": completion["best_checkpoint_sha256"],
        "uniform_spoof_mass": 5 / 8, "uniform_bonafide_mass": 3 / 8, "d_js_units": "nats",
        "source": summary(source), "target": summary(target), "domains": domains,
        "all_logits_bit_exact_to_original_csv": True, "precision": "native_batch_one_bf16_fp32_sensitive",
        "script_sha256": checkpoint_digest(Path(__file__)), "trained_parameters_changed": False,
        "target_use": "descriptive diagnostics only; not recipe/design selection"}
    result["source_macro"] = {name: {metric: float(np.mean([domains[d + "/dev"]["ranking"][name][metric]
        for d in FOLDS[fold]["sources"]])) for metric in ("eer", "auroc")}
        for name in ("p_w_spoof", "p_c_spoof", "original_classifier")}
    plt = plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for population, rows in (("Source", source), ("Target", target)):
        array = np.asarray([row[2:] for row in rows], dtype=np.float64)
        for name, y in (("bona fide", 0), ("spoof", 1)):
            axes[0].hist(array[array[:, 0] == y, 3], bins=np.linspace(-1e-6, np.log(2) + 1e-6, 51),
                         density=True, histtype="step", label=f"{population} {name}")
        axes[1].hist(array[array[:, 0] == 0, 2], bins=np.linspace(0, 1, 51), density=True,
                     histtype="step", label=f"{population} bona fide")
    axes[0].set(xlabel="WavLM/CLAP JS divergence (nats)", ylabel="Density", title=f"{fold.upper()} four-group disagreement")
    axes[1].axvline(5/8, color="black", linestyle="--", label="Uniform spoof mass: 5/8")
    axes[1].set(xlabel="CLAP summed spoof-concept probability", ylabel="Density", title=f"{fold.upper()} bona-fide recordings")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(directory / "distributions.png", dpi=180)
    plt.close(fig)
    save_json(directory / "summary.json", result)
    print(f"Step 1 {fold} complete; every diagnostic logit matched original scores", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lacf_diagnostics_step1")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/drive/audio-deepfake-cache"))
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--fold", choices=("f1", "f2"))
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()
    if args.num_workers < 0 or (not args.static_only and args.fold is None):
        parser.error("Specify --static-only or --fold f1/f2; workers must be nonnegative")
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.static_only:
        static_diagnostics(args.output_dir)
    else:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Native CUDA BF16 required; no precision fallback")
        torch.backends.nnpack.set_flags(False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        diagnose_fold(args.fold, args.output_dir, args.data_root, args.num_workers)


if __name__ == "__main__":
    main()
