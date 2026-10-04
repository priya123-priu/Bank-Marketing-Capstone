"""Run the complete bank marketing analysis from the project root.

Usage: python -m src.analysis --data data/bank_marketing.csv --target y
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clustering import segment_customers
from .data_preprocessing import DatasetError, clean_dataset, load_dataset
from .eda import profile_dataset
from .feature_engineering import add_features
from .modeling import train_models


def run_analysis(
	data_path: str | Path,
	target: str = "y",
	project_root: str | Path = ".",
	dataset_source: str = "Not recorded; verify and supply provenance before citing a source.",
) -> dict:
	project_root = Path(project_root).resolve()
	frame = load_dataset(data_path)
	if target not in frame.columns:
		likely = next((name for name in ("y", "deposit", "subscribed", "subscription") if name in frame), None)
		hint = f" Likely target column: '{likely}'." if likely else ""
		raise DatasetError(f"Target '{target}' not found. Available columns: {list(frame.columns)}.{hint}")

	cleaned, quality = clean_dataset(frame)
	featured = add_features(cleaned)
	processed_dir = project_root / "data" / "processed"
	processed_dir.mkdir(parents=True, exist_ok=True)
	featured.to_csv(processed_dir / "bank_marketing_cleaned.csv", index=False)

	profile = profile_dataset(featured, project_root / "visualizations", target)
	clustering = segment_customers(featured, target, project_root / "reports")
	try:
		modeling = train_models(featured, target, project_root / "models")
	except (ValueError, TypeError) as exc:
		modeling = {"status": "not_calculated", "reason": str(exc)}

	summary = {
		"project_title": "Bank Marketing Campaign Analysis, Customer Segmentation and Subscription Prediction Using Python",
		"dataset_file": str(Path(data_path)),
		"dataset_source": dataset_source,
		"dataset_dimensions": {"rows": int(len(cleaned)), "columns": int(cleaned.shape[1])},
		"target": target,
		"data_quality": quality,
		"profile": profile,
		"clustering": clustering,
		"modeling": modeling,
	}
	reports_dir = project_root / "reports"
	reports_dir.mkdir(parents=True, exist_ok=True)
	(reports_dir / "analysis_summary.json").write_text(
		json.dumps(summary, indent=2, default=str), encoding="utf-8"
	)
	return summary


def main() -> None:
	parser = argparse.ArgumentParser(description="Run the bank marketing capstone pipeline.")
	parser.add_argument("--data", default="data/bank_marketing.csv", help="Path to the source CSV/TSV file.")
	parser.add_argument("--target", default="y", help="Target column (default: y).")
	parser.add_argument(
		"--source",
		default="Not recorded; verify and supply provenance before citing a source.",
		help="Verified dataset provenance to include in the summary.",
	)
	args = parser.parse_args()
	try:
		summary = run_analysis(args.data, args.target, dataset_source=args.source)
	except DatasetError as exc:
		parser.error(str(exc))
	print(json.dumps({
		"rows": summary["dataset_dimensions"]["rows"],
		"columns": summary["dataset_dimensions"]["columns"],
		"modeling_status": summary["modeling"]["status"],
		"selected_model": summary["modeling"].get("selected_model"),
	}, indent=2))


if __name__ == "__main__":
	main()
