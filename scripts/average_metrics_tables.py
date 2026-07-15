"""Average saved metrics summary tables across datasets.

Example:
    python scripts/average_metrics_tables.py --root eval/real/translate_and_rotate
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
except ModuleNotFoundError:
    Console = None
    Table = None


DEFAULT_HEADERS = [
    "Region",
    "Frames/Pixels",
    "PSNR",
    "SSIM",
    "LPIPS",
    "Depth RMSE",
    "Depth MAE",
    "Reproj Mean",
    "Reproj Median",
]
CONSOLE = Console() if Console is not None else None


def _parse_markdown_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 3:
        raise ValueError("expected a markdown table with a header, separator, and at least one row")

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        values = [cell.strip() for cell in line.strip("|").split("|")]
        if len(values) != len(headers):
            raise ValueError(f"row has {len(values)} cells, expected {len(headers)}")
        rows.append(dict(zip(headers, values)))

    return headers, rows


def _to_float(value: str) -> float | None:
    if value in {"", "N/A", "nan", "NaN", "None"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_average(value: float | None, column: str) -> str:
    if value is None:
        return "N/A"
    if column == "Frames/Pixels":
        return f"{value:.0f}" if value.is_integer() else f"{value:.2f}"
    return f"{value:.4f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _rich_table(headers: list[str], rows: list[list[str]], table_count: int) -> Table:
    assert Table is not None
    table = Table(title=f"Average Evaluation Metrics ({table_count} tables)")
    for index, header in enumerate(headers):
        table.add_column(header, style="bold" if index == 0 else None, justify="left" if index == 0 else "right")
    for row in rows:
        table.add_row(*row)
    return table


def _matches_filters(path: Path, includes: list[str], excludes: list[str]) -> bool:
    path_text = path.as_posix()
    return all(text in path_text for text in includes) and not any(text in path_text for text in excludes)


def average_tables(paths: list[Path]) -> tuple[list[str], list[list[str]]]:
    headers: list[str] | None = None
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    region_order: list[str] = []

    for path in paths:
        table_headers, rows = _parse_markdown_table(path)
        if headers is None:
            headers = table_headers
        elif table_headers != headers:
            raise ValueError(f"{path} has different columns than previous tables")

        for row in rows:
            region = row.get("Region")
            if not region:
                raise ValueError(f"{path} has a row without a Region value")
            if region not in region_order:
                region_order.append(region)
            for column in headers:
                if column == "Region":
                    continue
                value = _to_float(row[column])
                if value is not None:
                    values[region][column].append(value)

    if headers is None:
        headers = DEFAULT_HEADERS

    averaged_rows: list[list[str]] = []
    for region in region_order:
        averaged_row = [region]
        for column in headers:
            if column == "Region":
                continue
            column_values = values[region][column]
            average = sum(column_values) / len(column_values) if column_values else None
            averaged_row.append(_format_average(average, column))
        averaged_rows.append(averaged_row)

    return headers, averaged_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="Root directory to search. Can be provided multiple times. Defaults to eval if it exists, otherwise outputs.",
    )
    parser.add_argument("--pattern", default="metrics_summary_*.md", help="Markdown table filename pattern to average.")
    parser.add_argument("--include", action="append", default=[], help="Only include paths containing this text.")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude paths containing this text.")
    args = parser.parse_args()

    roots = args.root
    if roots is None:
        roots = [Path("eval") if Path("eval").exists() else Path("outputs")]

    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(path for path in root.rglob(args.pattern) if _matches_filters(path, args.include, args.exclude))
    paths = sorted(paths)

    if not paths:
        searched_roots = ", ".join(str(root) for root in roots)
        raise SystemExit(f"No tables matching {args.pattern!r} found under: {searched_roots}")

    headers, rows = average_tables(paths)
    if CONSOLE is None:
        print(_markdown_table(headers, rows))
        print("\nInstall rich to render this as a formatted terminal table.")
    else:
        CONSOLE.print(_rich_table(headers, rows, len(paths)))


if __name__ == "__main__":
    main()
