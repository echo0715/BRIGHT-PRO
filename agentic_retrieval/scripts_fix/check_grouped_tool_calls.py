import json
import sys
from pathlib import Path


def file_has_grouped_tool_calls(file_path: Path) -> bool:
    """
    Return True if the JSON file contains two or more consecutive entries with
    type == "tool_call" that belong to the same round (i.e., grouped calls).
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    result = data.get("result")
    if not isinstance(result, list):
        return False

    prev_was_tool_call = False
    prev_round = None

    for entry in result:
        if not isinstance(entry, dict):
            prev_was_tool_call = False
            prev_round = None
            continue

        entry_type = entry.get("type")
        if entry_type == "tool_call":
            curr_round = entry.get("round")
            if prev_was_tool_call and curr_round == prev_round:
                return True
            prev_was_tool_call = True
            prev_round = curr_round
        else:
            prev_was_tool_call = False
            prev_round = None

    return False


def main() -> None:
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = Path("fixed_turn_runs")

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Directory not found: {target_dir}")
        sys.exit(1)

    json_files = list(target_dir.rglob("*.json"))
    matched_files = []

    for json_file in json_files:
        if file_has_grouped_tool_calls(json_file):
            matched_files.append(json_file)

    # Save summary JSON next to this script ("here")
    summary = {
        "target_dir": str(target_dir),
        "scanned": len(json_files),
        "matched_count": len(matched_files),
        "matched_files": [str(p) for p in matched_files],
    }
    out_path = Path(__file__).parent / "grouped_tool_calls.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write summary JSON to {out_path}: {e}")

    print(f"Total JSON files scanned: {len(json_files)}")
    print(f"Files with grouped tool_call entries: {len(matched_files)}")
    if matched_files:
        for p in matched_files:
            print(p)


if __name__ == "__main__":
    main()


