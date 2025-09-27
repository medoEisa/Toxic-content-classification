# database/database.py
import csv
import os
from datetime import datetime
from typing import Optional, List, Tuple

class CSVDatabase:
    """
    Simple CSV-based 'database' for storing classified inputs.
    Automatically creates/updates combined_data.csv.
    """
    def __init__(self, csv_path: str = "combined_data.csv"):
        self.csv_path = csv_path
        self.headers = [
            "original_input",
            "caption",
            "predicted_class",
            "confidence",
        ]
        self._ensure_file()

    def _ensure_file(self):
        """Ensure CSV exists with header row."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def insert_record(
        self,
        original_input: str,
        caption: Optional[str],
        predicted_class: str,
        confidence: float,
    ) -> None:
        
        self._ensure_file()
        row = [
            original_input,
            caption if caption is not None else "",
            predicted_class,
            float(confidence),
        ]
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def fetch_all(self) -> List[Tuple]:
        """Return all stored rows as list of tuples (excluding header)."""
        if not os.path.exists(self.csv_path):
            return []
        with open(self.csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) <= 1:
                return []
            return [tuple(r) for r in rows[1:]]  # skip header
