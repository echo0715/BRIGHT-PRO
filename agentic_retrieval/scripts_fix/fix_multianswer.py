#!/usr/bin/env python3
import json
from pathlib import Path

def dedupe_file(path: Path) -> bool:
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return False

    result = data.get('result')
    if not isinstance(result, list):
        return False

    seen_rounds = set()
    deduped = []
    changed = False

    for item in result:
        if isinstance(item, dict) and item.get('type') == 'final_answer':
            rnd = item.get('round')
            if rnd in seen_rounds:
                changed = True
                continue
            seen_rounds.add(rnd)
        deduped.append(item)

    if changed:
        data['result'] = deduped
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    return changed

def main():
    script_dir = Path(__file__).resolve().parent
    root = script_dir / 'fixed_turn_runs'
    if not root.exists():
        print(f'Not found: {root}')
        return

    processed = 0
    modified = 0
    errors = 0

    for path in root.rglob('run_*.json'):
        processed += 1
        try:
            if dedupe_file(path):
                modified += 1
        except Exception as e:
            errors += 1
            print(f'Error processing {path}: {e}')

    summary = {
        'processed': processed,
        'modified': modified,
        'errors': errors,
        'root': str(root),
    }
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()