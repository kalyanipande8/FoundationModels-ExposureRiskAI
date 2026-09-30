#!/usr/bin/env python3
"""
Run the full pipeline end-to-end:
  1. Transfer learning vs. from-scratch comparison on the small target task.
  2. Generative (VAE) data-augmentation experiment.
  3. Scenario modeling demo.

Writes all real, measured results to output/results.json and prints a
human-readable summary. No live LLM / foundation-model API calls are
made anywhere in this repository.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.transfer_experiment import run_transfer_vs_scratch
from src.augmentation_experiment import run_augmentation_experiment
from src.scenario_modeling import run_scenario_modeling


def main():
    print("=" * 70)
    print("1. TRANSFER LEARNING vs FROM-SCRATCH (small target exposure task)")
    print("=" * 70)
    transfer_results = run_transfer_vs_scratch(seed=42)
    print(json.dumps(transfer_results, indent=2))

    print("\n" + "=" * 70)
    print("2. GENERATIVE DATA AUGMENTATION (small VAE)")
    print("=" * 70)
    aug_results = run_augmentation_experiment(seed=42)
    print(json.dumps(
        {k: v for k, v in aug_results.items()
         if k in ("baseline", "augmented", "auc_delta", "accuracy_delta",
                   "n_train_real", "n_synth", "vae_initial_loss", "vae_final_loss")},
        indent=2,
    ))

    print("\n" + "=" * 70)
    print("3. SCENARIO MODELING (generative synthetic future scenarios)")
    print("=" * 70)
    scenario_results = run_scenario_modeling(seed=42)
    print(json.dumps(scenario_results, indent=2))

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    all_results = {
        "transfer_vs_scratch": transfer_results,
        "augmentation": aug_results,
        "scenario_modeling": scenario_results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {out_dir / 'results.json'}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    t = transfer_results
    print(f"Transfer test AUC: {t['transfer']['test_auc']:.4f} | "
          f"From-scratch test AUC: {t['scratch']['test_auc']:.4f}")
    print(f"Transfer test accuracy: {t['transfer']['test_accuracy']:.4f} | "
          f"From-scratch test accuracy: {t['scratch']['test_accuracy']:.4f}")
    print(f"Augmentation AUC delta (augmented - baseline): {aug_results['auc_delta']:+.4f}")


if __name__ == "__main__":
    main()
