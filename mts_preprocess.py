import csv
import json
import re
from pathlib import Path

DATA_DIR = Path("./MTS-Dialog_Data/Main-Dataset")
OUTPUT_FILE = Path("./mts_dialog_merged.json")


def process_dialogue(dialogue):
    """
    Remove Doctor:/Patient: prefixes and join the dialogue
    into one paragraph.
    """
    if not dialogue:
        return ""

    lines = dialogue.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Remove speaker prefixes
        line = re.sub(
            r"^(Doctor|Patient)\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )

        if line:
            cleaned_lines.append(line)

    return " ".join(cleaned_lines)


all_rows = []

# Read every CSV under Main-Dataset
for csv_file in sorted(DATA_DIR.glob("*.csv")):
    print(f"Processing {csv_file.name}")

    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Add processed dialogue while keeping all original fields
            row["processed_dialogue"] = process_dialogue(
                row.get("dialogue", "")
            )

            # Optional: record which CSV it came from
            row["source_file"] = csv_file.name

            all_rows.append(row)


# Save everything as one JSON array
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        all_rows,
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"\nSaved {len(all_rows)} records to {OUTPUT_FILE}")