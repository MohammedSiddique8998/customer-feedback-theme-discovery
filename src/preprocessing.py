import csv
import re
from dataclasses import dataclass
from pathlib import Path


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their", "them",
    "to", "use", "used", "using", "when", "who", "with", "within", "without",
}


@dataclass(frozen=True)
class TextRecord:
    record_id: str
    sentence: str
    cleaned_sentence: str


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if len(token) > 1 and token not in STOP_WORDS]
    return " ".join(tokens)


def load_dataset(path: Path) -> list[TextRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"ID", "Sentence"}
        if not reader.fieldnames or required.difference(reader.fieldnames):
            raise ValueError("Dataset must be a TSV file with ID and Sentence columns.")
        records = [
            TextRecord(
                record_id=row["ID"],
                sentence=row["Sentence"],
                cleaned_sentence=clean_text(row["Sentence"]),
            )
            for row in reader
            if row.get("Sentence", "").strip()
        ]
    if len(records) < 2:
        raise ValueError("At least two text records are required for clustering.")
    return records


def describe_records(records: list[TextRecord]) -> dict[str, float | int]:
    lengths = [len(record.sentence.split()) for record in records]
    return {
        "rows": len(records),
        "average_sentence_words": round(sum(lengths) / len(lengths), 2),
        "minimum_sentence_words": min(lengths),
        "maximum_sentence_words": max(lengths),
    }
