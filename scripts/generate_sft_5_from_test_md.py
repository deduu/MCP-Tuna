import json
import re
from pathlib import Path


def _clean_cell(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("â€¢", "•")
    text = re.sub(r"<br\\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    return text.strip()


def _parse_first_markdown_table(markdown: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln.rstrip("\n") for ln in markdown.splitlines() if ln.strip()]
    table_lines = [ln for ln in lines if ln.lstrip().startswith("|")]
    if len(table_lines) < 3:
        raise ValueError("No markdown table found")

    header = [_clean_cell(c) for c in table_lines[0].strip("|").split("|")]
    # Skip separator row at index 1
    rows = []
    for ln in table_lines[2:]:
        cells = [_clean_cell(c) for c in ln.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(cells)
    if not rows:
        raise ValueError("No data rows found in markdown table")
    return header, rows


def _to_snake(name: str) -> str:
    name = re.sub(r"[^0-9A-Za-z]+", "_", name.strip()).strip("_")
    return name.lower()


def build_5_datapoints(markdown: str, source_name: str) -> list[dict]:
    header, rows = _parse_first_markdown_table(markdown)
    row = rows[0]
    record = {header[i]: row[i] for i in range(len(header))}

    jalur_items = [s.strip("• ").strip() for s in record.get("Jalur Proyek", "").splitlines() if s.strip()]
    jalur_bullets = "\n".join([f"- {s}" for s in jalur_items]) if jalur_items else "-"

    compact = {
        _to_snake(k): v
        for k, v in record.items()
        if v and v != "N/A"
    }
    compact["jalur_proyek_items"] = jalur_items

    level = record.get("Level", "").strip()
    nama_level = record.get("Nama Level", "").strip()

    return [
        {
            "instruction": "Ringkas baris matriks untuk Level E0 (On Production) dalam 3-5 kalimat Bahasa Indonesia.",
            "input": markdown,
            "output": (
                f"Level {level} ({nama_level}) menggambarkan proyek/lapangan yang sudah memasuki tahap produksi komersial. "
                f"Status penemuannya terkonfirmasi, hasil uji alirnya konklusif, dan fasilitasnya sudah beroperasi. "
                f"Status produksi tercatat sedang berproduksi (komersial) dan sudah memiliki izin berproduksi, dengan FID disetujui. "
                f"Dari sisi siklus, eksplorasi sudah selesai dan fase berada pada eksploitasi, dengan status sumber daya reserve/cadangan."
            ),
            "metadata": {"source": source_name, "level": level, "technique": "sft"},
        },
        {
            "instruction": "Ekstrak informasi Level E0 menjadi JSON yang rapi. Ubah 'Jalur Proyek' menjadi array string, dan sertakan field penting terkait produksi, izin, FID, fasilitas, fase, dan sumber daya.",
            "input": markdown,
            "output": json.dumps(
                {
                    "level": level,
                    "nama_level": nama_level,
                    "jalur_proyek": jalur_items,
                    "status_penemuan": record.get("Status Penemuan"),
                    "hasil_uji_alir": record.get("Hasil Uji Alir"),
                    "status_produksi": record.get("Status Produksi"),
                    "status_izin_berproduksi": record.get("Status Izin Berproduksi"),
                    "status_fid": record.get("Status FID"),
                    "status_pembangunan_fasilitas": record.get("Status Pembangunan Fasilitas"),
                    "batasan_waktu": record.get("Batasan Waktu"),
                    "status_eksplorasi": record.get("Status Eksplorasi"),
                    "fase": record.get("Fase"),
                    "status_sumber_daya": record.get("Status Sumber Daya"),
                    "jenis_izin_berproduksi": record.get("Jenis Izin Berproduksi"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "metadata": {"source": source_name, "level": level, "technique": "sft"},
        },
        {
            "instruction": "Buat checklist singkat (bullet list) untuk memastikan sebuah proyek memenuhi kriteria Level E0 berdasarkan tabel.",
            "input": markdown,
            "output": "\n".join(
                [
                    "- Penemuan terkonfirmasi (Status Penemuan: Penemuan Terkonfirmasi).",
                    "- Uji alir konklusif (Hasil Uji Alir: Aliran Konklusif).",
                    "- Sedang berproduksi secara komersial (Status Produksi: Sedang Berproduksi (Komersial)).",
                    "- Memiliki izin berproduksi (Status Izin Berproduksi: Memiliki Izin Berproduksi).",
                    "- FID disetujui (Status FID: FID Disetujui).",
                    "- Fasilitas beroperasi (Status Pembangunan Fasilitas: Fasilitas Beroperasi).",
                    "- Eksplorasi selesai dan fase eksploitasi (Status Eksplorasi: Eksplorasi Selesai; Fase: Eksploitasi).",
                ]
            ),
            "metadata": {"source": source_name, "level": level, "technique": "sft"},
        },
        {
            "instruction": "Tuliskan kembali 'Jalur Proyek' untuk Level E0 sebagai bullet list markdown yang bersih (tanpa tag HTML).",
            "input": markdown,
            "output": jalur_bullets,
            "metadata": {"source": source_name, "level": level, "technique": "sft"},
        },
        {
            "instruction": "Buat ringkasan tabel mini (markdown) untuk Level E0 dengan kolom: Level, Nama Level, Status Produksi, Status Izin Berproduksi, Status FID, Status Pembangunan Fasilitas, Fase.",
            "input": markdown,
            "output": "\n".join(
                [
                    "| Level | Nama Level | Status Produksi | Status Izin Berproduksi | Status FID | Status Pembangunan Fasilitas | Fase |",
                    "|---|---|---|---|---|---|---|",
                    f"| {level} | {nama_level} | {record.get('Status Produksi')} | {record.get('Status Izin Berproduksi')} | {record.get('Status FID')} | {record.get('Status Pembangunan Fasilitas')} | {record.get('Fase')} |",
                ]
            ),
            "metadata": {"source": source_name, "level": level, "technique": "sft"},
        },
    ]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    in_path = root / "data" / "test.md"
    out_path = root / "data" / "sft_test_5.jsonl"

    markdown = in_path.read_text(encoding="utf-8")
    datapoints = build_5_datapoints(markdown, source_name="data/test.md")

    out_path.write_text(
        "".join(json.dumps(dp, ensure_ascii=False) + "\n" for dp in datapoints),
        encoding="utf-8",
    )
    print(f"Wrote {len(datapoints)} datapoints -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
