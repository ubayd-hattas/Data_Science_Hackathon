"""Inventory imports in repository notebooks and Python source files."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imports_from_source(source: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [{"parse_error": f"line {error.lineno}: {error.msg}"}]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                records.append(
                    {"module": alias.name, "name": None, "line": node.lineno}
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                records.append(
                    {
                        "module": node.module,
                        "name": alias.name,
                        "line": node.lineno,
                    }
                )
    return records


def main() -> None:
    files: dict[str, list[dict[str, object]]] = {}
    notebook_cells: dict[str, dict[str, list[dict[str, object]]]] = {}

    for path in sorted(ROOT.rglob("*.py")):
        if ".venv" in path.parts:
            continue
        files[str(path.relative_to(ROOT))] = imports_from_source(
            path.read_text(encoding="utf-8")
        )

    for path in sorted(ROOT.rglob("*.ipynb")):
        if any(part in {".venv", ".ipynb_checkpoints"} for part in path.parts):
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells: dict[str, list[dict[str, object]]] = {}
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            records = imports_from_source("".join(cell.get("source", [])))
            if records:
                cells[str(index)] = records
        notebook_cells[str(path.relative_to(ROOT))] = cells

    all_records = [record for records in files.values() for record in records]
    all_records += [
        record
        for cells in notebook_cells.values()
        for records in cells.values()
        for record in records
    ]
    top_level = sorted(
        {
            str(record["module"]).split(".")[0]
            for record in all_records
            if record.get("module")
        }
    )
    result = {
        "top_level_modules": top_level,
        "python_files": files,
        "notebooks": notebook_cells,
        "dependency_classification": {
            "third_party_baseline": [
                "matplotlib",
                "numpy",
                "pandas",
                "pyarrow",
                "sklearn",
            ],
            "standard_library": [
                "argparse",
                "ast",
                "json",
                "pathlib",
                "pickle",
                "platform",
                "time",
            ],
            "optional_not_required_locally": ["google.colab"],
        },
    }
    output = ROOT / "outputs" / "import_audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
