from pathlib import Path


def notes() -> str:
    return (
        "Optional extension: replace TF-IDF vectors with sentence-transformer embeddings "
        "when sentence-transformers and a permitted model are installed locally. The public "
        "repo keeps TF-IDF as the reproducible baseline so results can be regenerated without "
        "large model downloads."
    )


def main() -> None:
    output_path = Path("results") / "sentence_transformer_extension_note.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(notes(), encoding="utf-8")
    print(notes())


if __name__ == "__main__":
    main()
