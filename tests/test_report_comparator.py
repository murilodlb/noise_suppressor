import csv
import json

from src.cli import main as cli_main
from src.report_comparator import (
    build_comparison_records,
    collect_evaluation_reports,
    compare_reports,
    get_ideal_metrics_reference,
)


def test_ideal_metrics_reference_contains_all_core_metrics():
    refs = get_ideal_metrics_reference()
    assert len(refs) >= 20

    keys = {item["metric_key"] for item in refs}
    expected_keys = {
        "si_sdr_mean_db",
        "delta_si_sdr_mean_db",
        "pesq_mean",
        "delta_pesq_mean",
        "stoi_mean",
        "delta_stoi_mean",
        "mask_f1_mean",
        "mask_precision_mean",
        "mask_recall_mean",
        "speech_attenuation_db_mean",
        "noise_attenuation_db_mean",
    }
    assert expected_keys.issubset(keys)

    for item in refs:
        assert item["better_direction"] in ("higher", "lower")
        assert len(item["description"]) > 10
        assert item["ideal_value"] is not None


def test_collect_evaluation_reports_on_existing_repo_reports():
    reports = collect_evaluation_reports("reports/evaluation")
    assert len(reports) >= 3

    for r in reports:
        assert "report_id" in r
        assert "model_name" in r
        assert "checkpoint" in r
        assert "si_sdr_mean_db" in r
        assert "pesq_mean" in r
        assert "stoi_mean" in r


def test_build_comparison_records_places_ideal_and_oracle_first():
    dummy_summaries = [
        {
            "report_id": "run_a",
            "model_name": "tinyunet",
            "base_channels": 8,
            "loss_name": "bce",
            "trained_epoch": 20,
            "total_samples": 30,
            "si_sdr_mean_db": 12.5,
            "pesq_mean": 2.10,
            "stoi_mean": 0.85,
            "oracle_si_sdr_mean_db": 18.0,
            "oracle_pesq_mean": 3.90,
            "oracle_stoi_mean": 0.95,
        },
        {
            "report_id": "run_b",
            "model_name": "res_tinyunet",
            "base_channels": 16,
            "loss_name": "asym_mag_mse",
            "trained_epoch": 80,
            "total_samples": 100,
            "si_sdr_mean_db": 15.2,
            "pesq_mean": 2.25,
            "stoi_mean": 0.89,
            "oracle_si_sdr_mean_db": 18.4,
            "oracle_pesq_mean": 3.94,
            "oracle_stoi_mean": 0.96,
        },
    ]

    records = build_comparison_records(dummy_summaries, sort_by="si_sdr_mean_db", descending=True)
    assert len(records) == 4

    # Row 0 must be IDEAL_MODEL
    assert records[0]["report_id"] == "[IDEAL_MODEL]"
    assert records[0]["model_name"] == "ideal_reference"
    assert records[0]["pesq_mean"] == 4.50
    assert records[0]["stoi_mean"] == 1.000
    assert records[0]["speech_attenuation_db_mean"] == 0.00

    # Row 1 must be ORACLE_IRM_BOUND
    assert records[1]["report_id"] == "[ORACLE_IRM_BOUND]"
    assert records[1]["model_name"] == "oracle_irm"
    assert 18.0 <= records[1]["si_sdr_mean_db"] <= 18.5

    # Rows 2 and 3 must be sorted descending by si_sdr_mean_db
    assert records[2]["report_id"] == "run_b"
    assert records[3]["report_id"] == "run_a"


def test_compare_reports_exports_valid_csv_and_markdown(tmp_path):
    # Setup mock reports directory with 2 runs
    run_1_dir = tmp_path / "20261108_070000_tinyunet_ch8_mse"
    run_1_dir.mkdir(parents=True)
    mock_summary_1 = {
        "timestamp": "20261108_070000",
        "model_name": "tinyunet",
        "base_channels": 8,
        "loss_name": "mse",
        "trained_epoch": 80,
        "total_samples": 50,
        "systems": {
            "model": {
                "si_sdr": {"mean": 13.8, "std": 6.2},
                "pesq": {"mean": 2.05, "std": 0.5},
                "stoi": {"mean": 0.87, "std": 0.1},
            },
            "noisy": {
                "si_sdr": {"mean": 4.5, "std": 5.0},
                "pesq": {"mean": 1.35, "std": 0.3},
                "stoi": {"mean": 0.80, "std": 0.1},
            },
            "oracle": {
                "si_sdr": {"mean": 18.1, "std": 5.1},
                "pesq": {"mean": 3.91, "std": 0.4},
                "stoi": {"mean": 0.95, "std": 0.05},
            },
        },
        "deltas": {
            "delta_model_vs_noisy_si_sdr": {"mean": 9.3},
            "delta_model_vs_noisy_pesq": {"mean": 0.70},
            "delta_model_vs_noisy_stoi": {"mean": 0.07},
            "delta_oracle_vs_noisy_si_sdr": {"mean": 13.6},
        },
        "mask_metrics": {
            "precision": {"mean": 0.81},
            "recall": {"mean": 0.62},
            "f1": {"mean": 0.70},
            "accuracy": {"mean": 0.85},
            "false_alarm": {"mean": 0.05},
            "hit_minus_fa": {"mean": 0.57},
            "mse": {"mean": 0.055},
            "mae": {"mean": 0.165},
        },
        "attenuation_metrics": {
            "speech_attenuation_db": {"mean": 0.40},
            "noise_attenuation_db": {"mean": 11.8},
        },
        "oracle_attenuation_metrics": {
            "speech_attenuation_db": {"mean": 0.15},
            "noise_attenuation_db": {"mean": 14.8},
        },
    }
    (run_1_dir / "summary.json").write_text(json.dumps(mock_summary_1), encoding="utf-8")

    out_csv = tmp_path / "comparison.csv"
    out_md = tmp_path / "comparison.md"
    out_ref = tmp_path / "reference.csv"

    res = compare_reports(
        reports_dir=tmp_path,
        output_csv=out_csv,
        output_md=out_md,
        output_ref_csv=out_ref,
        sort_by="si_sdr_mean_db",
    )

    assert res["num_reports"] == 1
    assert out_csv.exists()
    assert out_md.exists()
    assert out_ref.exists()

    # Verify CSV content
    with out_csv.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 3  # [IDEAL_MODEL], [ORACLE_IRM_BOUND], run_1
        assert reader[0]["report_id"] == "[IDEAL_MODEL]"
        assert reader[0]["pesq_mean"] == "4.500"
        assert reader[0]["stoi_mean"] == "1.000"
        assert reader[1]["report_id"] == "[ORACLE_IRM_BOUND]"
        assert reader[2]["model_name"] == "tinyunet"
        assert reader[2]["loss_name"] == "mse"

    # Verify Markdown content
    md_text = out_md.read_text(encoding="utf-8")
    assert "[IDEAL_MODEL]" in md_text
    assert "[ORACLE_IRM_BOUND]" in md_text
    assert "tinyunet" in md_text


def test_cli_compare_reports_smoke(tmp_path):
    out_csv = tmp_path / "smoke_comparison.csv"
    exit_code = cli_main(
        [
            "compare_reports",
            "--reports_dir",
            "reports/evaluation",
            "--output_csv",
            str(out_csv),
        ]
    )
    assert exit_code == 0
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) >= 3  # IDEAL_MODEL, ORACLE_IRM_BOUND, + actual evaluation reports
        assert rows[0]["report_id"] == "[IDEAL_MODEL]"


def test_collect_evaluation_reports_handles_empty_or_corrupt_folder(tmp_path):
    # Empty directory
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    assert collect_evaluation_reports(empty_dir) == []

    # Corrupt JSON file
    corrupt_dir = tmp_path / "corrupt_run"
    corrupt_dir.mkdir()
    (corrupt_dir / "summary.json").write_text("{invalid json...", encoding="utf-8")
    assert collect_evaluation_reports(tmp_path) == []
