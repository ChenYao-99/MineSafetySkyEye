"""Read-only public PDF source review using the existing GeoMMA environment."""
import hashlib
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\LMHLD")
from runtime_paths import configure_runtime_paths
configure_runtime_paths()
import fitz

from public_reference_screen import ROOT, write_json, now


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["operator", "rescue", "lmhld"], default="operator")
    args = parser.parse_args()
    source = ROOT / "sources" / ("xincheng_2024_operator_sustainability.pdf" if args.source == "operator" else "hualien_official_zhonghe_rescue_record.pdf")
    if args.source == "lmhld":
        source = next(Path(r"D:\LMHLD").glob("Liu*LMHLD*.pdf"))
        receipt = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "original_existing_source": True}
    else:
        receipt = json.loads((ROOT / "receipts" / (source.name + ".json")).read_text(encoding="utf-8"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["sha256"]:
        raise ValueError("public PDF identity changed")
    records = []
    with fitz.open(source) as doc:
        for index, page in enumerate(doc):
            text = page.get_text()
            compact = "".join(text.split())
            coordinate_match = "121.623746" in compact and "24.144416" in compact
            if args.source == "rescue":
                event_match = "中和" in compact
            elif args.source == "lmhld":
                event_match = "Taiwan" in compact and "2018" in compact
            else:
                event_match = "0403" in compact or ("新城山" in compact and any(x in compact for x in ["崩塌", "地震", "落石"]))
            if coordinate_match or event_match:
                records.append({"page_index_zero_based": index, "pdf_page_one_based": index + 1,
                                "coordinate_match": coordinate_match, "event_match": event_match,
                                "text": text})
        if args.source == "operator":
            render = [row for row in records if row["coordinate_match"] or row["pdf_page_one_based"] in [61, 62]]
        else:
            render = [row for row in records if row["event_match"]][:2]
        for row in render:
            path = ROOT / "qualification" / f"{args.source}_report_page_{row['pdf_page_one_based']}.png"
            doc[row["page_index_zero_based"]].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(path)
            row["rendered_page"] = str(path)
        report = {"completed_at": now(), "source": receipt, "pdf_pages": len(doc), "relevant_pages": records,
                  "published_mine_coordinate_verified": any(row["coordinate_match"] for row in records),
                  "mine_boundary_provided_by_coordinate": False, "new_model_predictions": 0}
    write_json(ROOT / "qualification" / f"{args.source}_report_evidence.json", report)
    print(json.dumps({"pdf_pages": report["pdf_pages"], "coordinate_verified": report["published_mine_coordinate_verified"],
                      "matches": [{k: v for k, v in row.items() if k != "text"} for row in records]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
