import argparse
import json
import random
import re
from pathlib import Path


def clean_cell(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("Ã¢â‚¬Â¢", "•").replace("â€¢", "•")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text.strip()


def parse_record(markdown: str) -> dict[str, str]:
    lines = [ln.rstrip("\n") for ln in markdown.splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.lstrip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError("No markdown table found")
    header = [clean_cell(c) for c in table_lines[0].strip("|").split("|")]
    row = [clean_cell(c) for c in table_lines[2].strip("|").split("|")]
    if len(header) != len(row):
        raise ValueError("Invalid markdown table row")
    return {header[i]: row[i] for i in range(len(header))}


def split_jalur(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        item = re.sub(r"^[\s\-*•]+", "", line).strip()
        if item:
            items.append(item)
    return items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows(record: dict[str, str], n_train: int, n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    level = record["Level"]
    nama = record["Nama Level"]
    jalur = split_jalur(record["Jalur Proyek"])

    facts = {
        "nama level": nama,
        "status penemuan": record["Status Penemuan"],
        "hasil uji alir": record["Hasil Uji Alir"],
        "status produksi": record["Status Produksi"],
        "status izin berproduksi": record["Status Izin Berproduksi"],
        "status fid": record["Status FID"],
        "status pembangunan fasilitas": record["Status Pembangunan Fasilitas"],
        "batasan waktu": record["Batasan Waktu"],
        "status rencana pengembangan": record["Status Rencana Pengembangan"],
        "kelayakan komersial": record["Kelayakan Komersial"],
        "kecukupan data": record["Kecukupan Data"],
        "status eksplorasi": record["Status Eksplorasi"],
        "fase": record["Fase"],
        "status sumber daya": record["Status Sumber Daya"],
        "jenis izin berproduksi": record["Jenis Izin Berproduksi"],
    }

    train_q = [
        "Apa {field} pada level {level}?",
        "Sebutkan {field} untuk level {level}.",
        "Pada level {level}, apa nilai {field}?",
        "Tolong jawab singkat, {field} level {level} apa?",
        "Isi kolom {field} untuk level {level}.",
        "Untuk level {level}, {field}-nya apa?",
    ]
    eval_q = [
        "Kalau level {level}, apa {field}?",
        "Jawab cepat: {field} di {level}?",
        "Nilai {field} untuk {level} adalah?",
        "Pada klasifikasi {level}, isi {field} apa?",
    ]

    train_a = [
        "Pada level {level}, {field} adalah {value}.",
        "Nilai {field} untuk level {level} adalah {value}.",
        "Untuk level {level}, {field}: {value}.",
    ]
    eval_a = [
        "{field} pada level {level}: {value}.",
    ]

    train: list[dict] = []
    eval_rows: list[dict] = []
    items = list(facts.items())

    while len(train) < n_train:
        field, value = rng.choice(items)
        q = rng.choice(train_q).format(field=field, level=level)
        a = rng.choice(train_a).format(field=field, level=level, value=value)
        train.append({"prompt": q, "response": a})

        if len(train) % 25 == 0:
            train.append(
                {
                    "prompt": f"Sebutkan tiga jalur proyek level {level}.",
                    "response": f"Jalur proyek level {level}: {jalur[0]}; {jalur[1]}; {jalur[2]}.",
                }
            )

    while len(eval_rows) < n_eval:
        if len(eval_rows) % 10 == 0:
            eval_rows.append(
                {
                    "prompt": f"Apa saja jalur proyek pada level {level}?",
                    "response": f"Jalur proyek level {level}: {jalur[0]}; {jalur[1]}; {jalur[2]}.",
                }
            )
            continue
        field, value = rng.choice(items)
        q = rng.choice(eval_q).format(field=field, level=level)
        a = rng.choice(eval_a).format(field=field, level=level, value=value)
        eval_rows.append({"prompt": q, "response": a})

    return train[:n_train], eval_rows[:n_eval]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate focused E0 SFT dataset.")
    parser.add_argument("--input", default="data/test.md")
    parser.add_argument("--train-out", default="data/sft_e0_focused_train.jsonl")
    parser.add_argument("--eval-out", default="data/sft_e0_focused_eval.jsonl")
    parser.add_argument("--train-size", type=int, default=700)
    parser.add_argument("--eval-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    markdown = (repo / args.input).read_text(encoding="utf-8")
    record = parse_record(markdown)
    train, eval_rows = build_rows(record, args.train_size, args.eval_size, args.seed)

    write_jsonl(repo / args.train_out, train)
    write_jsonl(repo / args.eval_out, eval_rows)
    print(f"Wrote train={len(train)} -> {repo / args.train_out}")
    print(f"Wrote eval={len(eval_rows)} -> {repo / args.eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
