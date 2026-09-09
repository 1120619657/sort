from pathlib import Path
import runpy
import sys


SORT_SCRIPT = Path(__file__).resolve().parents[1] / "sort.py"


def test_demo_accepts_single_detection_row(tmp_path, monkeypatch):
    """A valid MOT detection file may contain just one detection row."""
    det_dir = tmp_path / "data" / "train" / "SINGLE" / "det"
    det_dir.mkdir(parents=True)
    (det_dir / "det.txt").write_text(
        "1,1,10,20,30,40,0.9,-1,-1,-1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SORT_SCRIPT), "--seq_path", str(tmp_path / "data"), "--phase", "train"],
    )

    runpy.run_path(str(SORT_SCRIPT), run_name="__main__")

    output_file = tmp_path / "output" / "SINGLE.txt"
    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8").strip() != ""
