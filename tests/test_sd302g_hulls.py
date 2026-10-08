"""Synthetic-only tests for the SD302g lower-bound builder and evaluator."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

BUILDER = importlib.import_module("build_sd302g_hulls")
EVALUATOR = importlib.import_module("evaluate_hull")
AN2K = importlib.import_module("build_sd302h_gt")


def make_record(record_type: int, fields: list[tuple[str, bytes]]) -> bytes:
    """Construct a length-correct synthetic traditional-encoding record."""
    non_length = [(tag, value) for tag, value in fields if tag != f"{record_type}.001"]
    declared = 0
    for _ in range(8):
        encoded = AN2K.GS.join(
            [f"{record_type}.001:{declared}".encode("ascii")]
            + [tag.encode("ascii") + b":" + value for tag, value in non_length]
        ) + AN2K.FS
        if len(encoded) == declared:
            return encoded
        declared = len(encoded)
    raise AssertionError("record length did not converge")


def frozen_split(subject: str) -> dict[str, object]:
    """Return a synthetic 136/24/40 assignment containing one chosen subject."""
    assignments: dict[str, str] = {subject: "train"}
    for index in range(1, 136):
        assignments[f"{index:08d}"] = "train"
    for index in range(200001, 200025):
        assignments[f"{index:08d}"] = "val"
    for index in range(300001, 300041):
        assignments[f"{index:08d}"] = "test"
    assert len(assignments) == 200
    return {"seed": 20260908, "splits": assignments}


def synthetic_transaction(subject: str = "99999999") -> tuple[str, bytes]:
    image = np.full((20, 20), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    payload = encoded.tobytes()
    type1 = make_record(1, [("1.003", b"0110")])
    minutiae = AN2K.RS.join(
        [
            AN2K.US.join((b"20", b"20", b"0", b"E", b"0", b"0")),
            AN2K.US.join((b"80", b"20", b"90", b"B", b"0", b"0")),
            AN2K.US.join((b"80", b"80", b"180", b"X", b"0", b"0")),
            AN2K.US.join((b"20", b"80", b"270", b"E", b"0", b"0")),
        ]
    )
    type9 = make_record(
        9,
        [
            ("9.002", b"1"),
            ("9.300", b"102" + AN2K.US + b"102"),
            ("9.302", b"04" + AN2K.US + b"DST"),
            ("9.331", minutiae),
        ],
    )
    type14 = make_record(
        14,
        [
            ("14.002", b"1"),
            ("14.003", b"1"),
            ("14.006", b"20"),
            ("14.007", b"20"),
            ("14.008", b"1"),
            ("14.009", b"500"),
            ("14.010", b"500"),
            ("14.011", b"PNG"),
            ("14.012", b"8"),
            ("14.996", hashlib.sha256(payload).hexdigest().encode("ascii")),
            ("14.999", payload),
        ],
    )
    stem = f"{subject}_Z_500_roll_04"
    member = f"sd302g/ebts/baseline/irr/Z/500/{subject}/{stem}.irr"
    return member, type1 + type9 + type14


def build_synthetic(root: Path) -> tuple[Path, Path, dict[str, str]]:
    subject = "99999999"
    member, transaction = synthetic_transaction(subject)
    archive = root / "synthetic_sd302g.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(member, transaction)
    splits = root / "splits.json"
    splits.write_text(json.dumps(frozen_split(subject)), encoding="utf-8")
    out = root / "private"
    args = BUILDER.parse_args(
        [
            str(archive),
            str(out),
            "--splits",
            str(splits),
            "--skip-release-validation",
        ]
    )
    BUILDER.build(args)
    with (out / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    return out, archive, row


def test_member_metadata_and_minutia_coordinate_convention() -> None:
    member, transaction = synthetic_transaction()
    metadata = BUILDER.parse_member_metadata(member)
    assert metadata.group == "baseline"
    assert metadata.device == "Z"
    assert metadata.capture == "roll"
    assert metadata.finger_position == "04"
    records = AN2K.parse_transaction(transaction)
    annotation = next(record for record in records if record.record_type == 9)
    image_record = next(record for record in records if record.record_type == 14)
    image, _ = BUILDER.decode_type14_image(image_record, metadata.image_id)
    points = BUILDER.parse_minutiae(
        annotation,
        AN2K.parse_roi(annotation.one("9.300")),
        image.width,
        image.height,
        image.ppi,
    )
    assert [(point.x_px, point.y_px) for point in points] == [
        (4, 4),
        (16, 4),
        (16, 16),
        (4, 16),
    ]
    assert points[0].x_px_float == 20 * 500 / 2540


def test_half_up_rounding_is_explicit_and_no_clipping_occurs() -> None:
    assert BUILDER._nearest_pixel(2.49) == 2
    assert BUILDER._nearest_pixel(2.5) == 3
    assert BUILDER._nearest_pixel(2.51) == 3
    try:
        BUILDER._nearest_pixel(-0.01)
    except AN2K.FormatError:
        pass
    else:
        raise AssertionError("negative coordinate was silently clipped")


def test_roi_origin_follows_polygon_bounding_box_and_nonfull_roi_is_rejected() -> None:
    roi = AN2K.ROI(
        width_units=1000,
        height_units=900,
        x_offset_units=10,
        y_offset_units=20,
        vertices=((100, 200), (900, 200), (900, 800), (100, 800)),
    )
    assert BUILDER._roi_feature_origin(roi) == (110, 220)
    try:
        BUILDER.validate_full_image_roi(roi, 200, 180, 500.0)
    except AN2K.FormatError as exc:
        assert "full-image" in str(exc)
    else:
        raise AssertionError("polygon ROI entered the full-image lower-bound protocol")


def test_convex_hull_requires_three_noncollinear_unique_points() -> None:
    points = [
        BUILDER.Minutia(i + 1, i, i, float(i), float(i), i, i, 0, "X", None, None)
        for i in range(3)
    ]
    try:
        BUILDER.rasterize_minutiae_hull(points, 10, 10)
    except BUILDER.HullUnavailable as exc:
        assert exc.reason == "degenerate_minutiae_hull"
    else:
        raise AssertionError("collinear points produced a two-dimensional hull")


def test_synthetic_builder_writes_private_relative_artifacts_atomically() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        out, _, row = build_synthetic(root)
        assert row["split"] == "train"
        assert row["group"] == "baseline"
        assert row["device"] == "Z"
        assert row["capture"] == "roll"
        assert row["finger_position"] == "04"
        assert row["n_minutiae"] == "4"
        for column in ("image_path", "hull_path", "minutiae_path"):
            assert not Path(row[column]).is_absolute()
            assert (out / row[column]).is_file()
        hull = cv2.imread(str(out / row["hull_path"]), cv2.IMREAD_GRAYSCALE)
        assert hull is not None
        with (out / row["minutiae_path"]).open(newline="", encoding="utf-8") as handle:
            points = list(csv.DictReader(handle))
        assert all(hull[int(point["y_px"]), int(point["x_px"])] == 255 for point in points)
        assert list(csv.DictReader((out / "errors.csv").open(encoding="utf-8"))) == []
        assert list(csv.DictReader((out / "exclusions.csv").open(encoding="utf-8"))) == []
        provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
        assert provenance["verification_status"] == "SYNTHETIC_TEST_ONLY"
        assert provenance["manifest_sha256"] == BUILDER._sha256_file(
            out / "manifest.csv"
        )


def test_existing_output_rejects_nonempty_fatal_error_file() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        out, archive, _ = build_synthetic(root)
        with (out / "errors.csv").open("a", encoding="utf-8") as handle:
            handle.write("member.irr,transaction,synthetic failure\n")
        args = BUILDER.parse_args(
            [
                str(archive),
                str(out),
                "--splits",
                str(root / "splits.json"),
                "--skip-release-validation",
            ]
        )
        try:
            BUILDER.build(args)
        except AN2K.FormatError as exc:
            assert "fatal error" in str(exc)
        else:
            raise AssertionError("existing output with a fatal error was accepted")


def test_production_mode_requires_independent_reference_gate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        _, archive, _ = build_synthetic(root)
        args = BUILDER.parse_args(
            [str(archive), str(root / "new-private"), "--splits", str(root / "splits.json")]
        )
        try:
            BUILDER.build(args)
        except ValueError as exc:
            assert "require --reference-json" in str(exc)
        else:
            raise AssertionError("production mode ran without an independent reference")


def test_evaluator_rejects_absolute_private_artifact_path() -> None:
    try:
        EVALUATOR._manifest_artifact(Path("private_data/manifest.csv"), "/tmp/hull.png")
    except EVALUATOR.ProtocolError as exc:
        assert "must be relative" in str(exc)
    else:
        raise AssertionError("absolute private artifact path was accepted")


def test_lower_bound_reference_identities_and_aggregates() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        out, _, row = build_synthetic(root)
        predictions = root / "predictions"
        predictions.mkdir()
        hull = cv2.imread(str(out / row["hull_path"]), cv2.IMREAD_GRAYSCALE)
        assert hull is not None
        assert cv2.imwrite(str(predictions / f"{row['image_id']}.png"), hull)
        results = root / "results"
        args = EVALUATOR.parse_args(
            [
                "--manifest",
                str(out / "manifest.csv"),
                "--pred-template",
                str(predictions / "{image_id}.png"),
                "--method",
                "synthetic_method",
                "--display-name",
                "Synthetic method",
                "--out",
                str(results),
                "--split",
                "train",
            ]
        )
        per_image, summary = EVALUATOR.evaluate(args)
        by_method = {entry["method"]: entry for entry in per_image}
        hull_area = int(np.count_nonzero(hull))
        image_area = int(hull.size)

        for method in ("synthetic_method", "hull_as_prediction"):
            assert by_method[method]["mc"] == 1.0
            assert by_method[method]["hr"] == 1.0
            assert by_method[method]["par"] == 1.0
            assert by_method[method]["ohf"] == 0.0
        assert by_method["all_background"]["mc"] == 0.0
        assert by_method["all_background"]["hr"] == 0.0
        assert by_method["all_background"]["par"] == 0.0
        assert by_method["all_background"]["ohf"] == 0.0
        assert by_method["all_foreground"]["mc"] == 1.0
        assert by_method["all_foreground"]["hr"] == 1.0
        assert by_method["all_foreground"]["par"] == image_area / hull_area
        assert by_method["all_foreground"]["ohf"] == (image_area - hull_area) / image_area

        overall = [entry for entry in summary if entry["aggregate"] == "overall"]
        assert len(overall) == 4
        dimensions = {(entry["aggregate"], entry["value"]) for entry in summary}
        assert ("group", "baseline") in dimensions
        assert ("device", "Z") in dimensions
        assert ("capture", "roll") in dimensions
        assert {"mc", "hr", "par", "ohf"}.issubset(by_method["synthetic_method"])


def test_subject_bootstrap_is_deterministic_and_method_paired() -> None:
    rows = []
    for method in ("left", "right"):
        for subject, value in (("A", 0.2), ("B", 0.8), ("B", 1.0)):
            rows.append(
                {
                    "method": method,
                    "display_name": method,
                    "subject_id": subject,
                    "split": "test",
                    "group": "baseline",
                    "device": "Z",
                    "capture": "roll",
                    "verification_status": "SYNTHETIC_TEST_ONLY",
                    "n_minutiae": 10,
                    "minutiae_covered": round(10 * value),
                    "hull_pixels": 10,
                    "hull_covered": round(10 * value),
                    "predicted_pixels": round(10 * value),
                    "outside_hull_pixels": 0,
                    "mc": value,
                    "hr": value,
                    "par": value,
                    "ohf": 0.0,
                    "missing": 0,
                }
            )
    first = EVALUATOR.aggregate_rows(rows, 1234, 1000)
    second = EVALUATOR.aggregate_rows(rows, 1234, 1000)
    assert first == second
    overall = {
        row["method"]: row
        for row in first
        if row["aggregate"] == "overall"
    }
    assert overall["left"]["bootstrap_status"] == "ok"
    assert overall["left"]["mc_ci_low"] == overall["right"]["mc_ci_low"]
    assert overall["left"]["mc_ci_high"] == overall["right"]["mc_ci_high"]
    assert overall["left"]["bootstrap_replicates"] == 1000


def test_missing_prediction_is_loudly_scored_as_all_background() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        out, _, _ = build_synthetic(root)
        args = EVALUATOR.parse_args(
            [
                "--manifest",
                str(out / "manifest.csv"),
                "--pred-template",
                str(root / "missing" / "{image_id}.png"),
                "--method",
                "missing_method",
                "--display-name",
                "Missing method",
                "--out",
                str(root / "results"),
                "--split",
                "train",
            ]
        )
        per_image, summary = EVALUATOR.evaluate(args)
        method = next(row for row in per_image if row["method"] == "missing_method")
        assert method["missing"] == 1
        assert (method["mc"], method["hr"], method["par"], method["ohf"]) == (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        overall = next(
            row
            for row in summary
            if row["method"] == "missing_method" and row["aggregate"] == "overall"
        )
        assert overall["n_missing"] == 1


def test_hull_gate_rejects_a_minutia_outside_the_hull() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        out, _, row = build_synthetic(root)
        points_path = out / row["minutiae_path"]
        with points_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            points = list(reader)
        assert fieldnames is not None
        points[0]["x_px"] = "0"
        points[0]["y_px"] = "0"
        with points_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(points)
        args = EVALUATOR.parse_args(
            [
                "--manifest",
                str(out / "manifest.csv"),
                "--pred-template",
                str(out / "hulls" / "{image_id}.png"),
                "--method",
                "synthetic_method",
                "--display-name",
                "Synthetic method",
                "--out",
                str(root / "results"),
                "--split",
                "train",
            ]
        )
        try:
            EVALUATOR.evaluate(args)
        except EVALUATOR.ProtocolError as exc:
            assert "hull gate failed" in str(exc)
        else:
            raise AssertionError("hull gate accepted MC below 100%")
