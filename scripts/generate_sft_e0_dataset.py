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


def parse_first_markdown_table(markdown: str) -> dict[str, str]:
    lines = [ln.rstrip("\n") for ln in markdown.splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.lstrip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError("No markdown table found")

    header = [clean_cell(c) for c in table_lines[0].strip("|").split("|")]
    row = [clean_cell(c) for c in table_lines[2].strip("|").split("|")]
    if len(header) != len(row):
        raise ValueError("Invalid markdown table row")
    return {header[i]: row[i] for i in range(len(header))}


def jalur_items(raw: str) -> list[str]:
    out: list[str] = []
    for line in raw.splitlines():
        item = re.sub(r"^[\s\-*•]+", "", line).strip()
        if item:
            out.append(item)
    return out


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_dataset(record: dict[str, str], n_train: int, n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    level = record["Level"]
    nama = record["Nama Level"]
    jalur = jalur_items(record["Jalur Proyek"])

    facts = {
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

    direct_prompts_train = [
        "Apa {field} pada level {level}?",
        "Sebutkan {field} untuk level {level}.",
        "Untuk level {level}, nilai {field} adalah apa?",
        "Mohon isi {field} pada level {level}.",
        "Di matriks, apa {field} untuk {level}?",
        "Berikan {field} di level {level}.",
    ]
    direct_prompts_eval = [
        "Nilai {field} di level {level} apa ya?",
        "Kalau level {level}, {field}-nya apa?",
        "Tolong jawab singkat: {field} untuk {level}?",
        "Pada klasifikasi {level}, apa isi kolom {field}?",
    ]
    direct_responses = [
        "Pada level {level}, {field} adalah {value}.",
        "Nilai {field} untuk level {level} adalah {value}.",
        "Untuk level {level}, {field}: {value}.",
    ]

    multi_train = [
        (
            "Ringkas karakteristik inti level {level} ({nama}) dalam 2-3 kalimat.",
            "Level {level} ({nama}) merepresentasikan kondisi on production. Penemuan sudah terkonfirmasi dengan aliran konklusif, FID disetujui, fasilitas beroperasi, dan proyek berproduksi komersial. Eksplorasi selesai, fase eksploitasi, serta status sumber daya reserve/cadangan.",
        ),
        (
            "Apa implikasi kombinasi 'status eksplorasi' dan 'fase' pada level {level}?",
            "Pada level {level}, status eksplorasi adalah {status_eksplorasi} dan fasenya {fase}. Artinya proyek sudah melewati tahap eksplorasi dan berada pada tahap pemanfaatan/produksi sumber daya.",
        ),
        (
            "Sebutkan indikator kesiapan operasi pada level {level}.",
            "Indikator kesiapan operasi level {level}: {status_produksi}, {status_izin_berproduksi}, {status_fid}, dan {status_pembangunan_fasilitas}.",
        ),
        (
            "Buat bullet list jalur proyek yang termasuk level {level}.",
            "- {jalur_1}\n- {jalur_2}\n- {jalur_3}",
        ),
        (
            "Apakah level {level} layak secara komersial? Jelaskan satu kalimat.",
            "Ya. Level {level} memiliki status kelayakan komersial: {kelayakan_komersial}.",
        ),
    ]
    multi_eval = [
        (
            "Jelaskan hubungan antara status produksi, izin berproduksi, dan FID pada level {level}.",
            "Pada level {level}, proyek sudah {status_produksi}, telah {status_izin_berproduksi}, dan {status_fid}. Kombinasi ini menunjukkan proyek telah siap dan berjalan pada tahap produksi komersial.",
        ),
        (
            "Tuliskan ringkasan satu kalimat tentang posisi level {level} di siklus proyek.",
            "Level {level} berada pada fase {fase} dengan status eksplorasi {status_eksplorasi}, sehingga fokusnya pada operasi/eksploitasi sumber daya.",
        ),
        (
            "Apa batasan waktu jika proyek level {level} tidak berproduksi?",
            "Batasan waktunya adalah {batasan_waktu}.",
        ),
    ]

    context_block = normalize_ws(
        f"Level {level}; Nama Level {nama}; Jalur Proyek: {', '.join(jalur)}; "
        f"Status Penemuan {facts['status penemuan']}; Hasil Uji Alir {facts['hasil uji alir']}; "
        f"Status Produksi {facts['status produksi']}; Status Izin Berproduksi {facts['status izin berproduksi']}; "
        f"Status FID {facts['status fid']}; Status Pembangunan Fasilitas {facts['status pembangunan fasilitas']}; "
        f"Batasan Waktu {facts['batasan waktu']}; Kelayakan Komersial {facts['kelayakan komersial']}; "
        f"Status Eksplorasi {facts['status eksplorasi']}; Fase {facts['fase']}; "
        f"Status Sumber Daya {facts['status sumber daya']}; Jenis Izin Berproduksi {facts['jenis izin berproduksi']}."
    )

    fill_values = {
        "level": level,
        "nama": nama,
        "status_eksplorasi": facts["status eksplorasi"],
        "fase": facts["fase"],
        "status_produksi": facts["status produksi"],
        "status_izin_berproduksi": facts["status izin berproduksi"],
        "status_fid": facts["status fid"],
        "status_pembangunan_fasilitas": facts["status pembangunan fasilitas"],
        "kelayakan_komersial": facts["kelayakan komersial"],
        "batasan_waktu": facts["batasan waktu"],
        "jalur_1": jalur[0] if len(jalur) > 0 else "-",
        "jalur_2": jalur[1] if len(jalur) > 1 else "-",
        "jalur_3": jalur[2] if len(jalur) > 2 else "-",
    }

    def make_direct_item(field: str, value: str, eval_mode: bool) -> dict:
        prompt_tpl = rng.choice(direct_prompts_eval if eval_mode else direct_prompts_train)
        response_tpl = rng.choice(direct_responses)
        prompt = prompt_tpl.format(field=field, level=level)
        if not eval_mode and rng.random() < 0.35:
            prompt = f"{prompt}\n\nKonteks: {context_block}"
        response = response_tpl.format(level=level, field=field, value=value)
        return {"prompt": prompt, "response": response}

    def make_multi_item(eval_mode: bool) -> dict:
        pool = multi_eval if eval_mode else multi_train
        prompt_tpl, response_tpl = rng.choice(pool)
        prompt = prompt_tpl.format(**fill_values)
        if not eval_mode and rng.random() < 0.35:
            prompt = f"{prompt}\n\nKonteks: {context_block}"
        response = response_tpl.format(**fill_values)
        return {"prompt": prompt, "response": response}

    train: list[dict] = []
    eval_set: list[dict] = []

    direct_items = list(facts.items())
    while len(train) < n_train:
        if rng.random() < 0.72:
            field, value = rng.choice(direct_items)
            train.append(make_direct_item(field, value, eval_mode=False))
        else:
            train.append(make_multi_item(eval_mode=False))

    while len(eval_set) < n_eval:
        if rng.random() < 0.65:
            field, value = rng.choice(direct_items)
            eval_set.append(make_direct_item(field, value, eval_mode=True))
        else:
            eval_set.append(make_multi_item(eval_mode=True))

    return train, eval_set


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate larger SFT dataset from data/test.md for E0 domain.")
    parser.add_argument("--input", default="data/test.md")
    parser.add_argument("--train-out", default="data/sft_e0_train.jsonl")
    parser.add_argument("--eval-out", default="data/sft_e0_eval.jsonl")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--eval-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    markdown = (repo / args.input).read_text(encoding="utf-8")
    record = parse_first_markdown_table(markdown)
    train, eval_set = build_dataset(record, args.train_size, args.eval_size, args.seed)

    write_jsonl(repo / args.train_out, train)
    write_jsonl(repo / args.eval_out, eval_set)
    print(f"Wrote train={len(train)} -> {repo / args.train_out}")
    print(f"Wrote eval={len(eval_set)} -> {repo / args.eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
