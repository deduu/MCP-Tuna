from typing import List, Dict, Tuple
import json
import os
import re


class PageLoader:
    def load(self, file_path: str) -> tuple[str, List[Dict]]:
        raise NotImplementedError


class MarkdownPageLoader(PageLoader):
    _SECTION_HEADING_RE = re.compile(r"(?m)^##\s+.+$")
    _SUBSECTION_HEADING_RE = re.compile(r"(?m)^###\s+.+$")
    _MAX_SECTION_CHARS = 6000

    def load(self, file_path: str) -> tuple[str, List[Dict]]:
        file_name = os.path.basename(file_path).rsplit('.', 1)[0]

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().replace('\r\n', '\n')

        pages = [
            {
                'index': i,
                'markdown': page,
            }
            for i, page in enumerate(self._split_markdown_document(text))
            if page.strip()
        ]

        return file_name, pages

    def _split_markdown_document(self, text: str) -> List[str]:
        front_matter, body = self._extract_front_matter(text)
        body = body.strip()
        if not body:
            return [front_matter.strip()] if front_matter.strip() else []

        headings = list(self._SECTION_HEADING_RE.finditer(body))
        if not headings:
            return self._split_oversized_section(
                self._compose_chunk(front_matter, body)
            )

        preamble = body[:headings[0].start()].strip()
        sections = []

        for idx, match in enumerate(headings):
            start = match.start()
            end = headings[idx + 1].start() if idx + 1 < len(headings) else len(body)
            section = body[start:end].strip()
            if not section:
                continue

            chunk = section
            if preamble:
                chunk = f"{preamble}\n\n{section}"

            sections.extend(
                self._split_oversized_section(
                    self._compose_chunk(front_matter, chunk)
                )
            )

        return sections

    def _extract_front_matter(self, text: str) -> Tuple[str, str]:
        if not text.startswith('---\n'):
            return '', text

        closing = text.find('\n---\n', 4)
        if closing == -1:
            return '', text

        front_matter_end = closing + len('\n---\n')
        return text[:front_matter_end].strip(), text[front_matter_end:]

    def _split_oversized_section(self, section: str) -> List[str]:
        if len(section) <= self._MAX_SECTION_CHARS:
            return [section.strip()]

        subsection_matches = list(self._SUBSECTION_HEADING_RE.finditer(section))
        if subsection_matches:
            return self._split_on_matches(section, subsection_matches)

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', section) if p.strip()]
        if len(paragraphs) <= 1:
            return [section[i:i + self._MAX_SECTION_CHARS].strip()
                    for i in range(0, len(section), self._MAX_SECTION_CHARS)]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for paragraph in paragraphs:
            para_len = len(paragraph) + 2
            if current and current_len + para_len > self._MAX_SECTION_CHARS:
                chunks.append('\n\n'.join(current).strip())
                current = [paragraph]
                current_len = len(paragraph)
                continue

            if len(paragraph) > self._MAX_SECTION_CHARS:
                if current:
                    chunks.append('\n\n'.join(current).strip())
                    current = []
                    current_len = 0
                chunks.extend(
                    paragraph[i:i + self._MAX_SECTION_CHARS].strip()
                    for i in range(0, len(paragraph), self._MAX_SECTION_CHARS)
                )
                continue

            current.append(paragraph)
            current_len += para_len

        if current:
            chunks.append('\n\n'.join(current).strip())

        return chunks

    def _split_on_matches(self, text: str, matches: List[re.Match[str]]) -> List[str]:
        chunks: List[str] = []
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _compose_chunk(self, front_matter: str, chunk: str) -> str:
        if front_matter:
            return f"{front_matter}\n\n{chunk.strip()}".strip()
        return chunk.strip()


class JsonPageLoader(PageLoader):
    def load(self, file_path: str) -> tuple[str, List[Dict]]:
        file_name = os.path.basename(file_path).rsplit('.', 1)[0]

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return file_name, data['pages']


class JsonlPageLoader(PageLoader):
    _TEXT_KEYS = ('markdown', 'text', 'content', 'page_content')

    def load(self, file_path: str) -> tuple[str, List[Dict]]:
        file_name = os.path.basename(file_path).rsplit('.', 1)[0]
        pages: List[Dict] = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f'Invalid JSONL row at line {i + 1}: expected an object.'
                    )

                text = None
                for key in self._TEXT_KEYS:
                    value = row.get(key)
                    if isinstance(value, str) and value.strip():
                        text = value.strip()
                        break

                if text is None:
                    raise ValueError(
                        'JSONL document rows must include one of: '
                        f'{", ".join(self._TEXT_KEYS)}. '
                        'For generated datasets, use the Dataset Library preview/load tools instead.'
                    )

                pages.append({
                    'index': i,
                    'markdown': text,
                })

        return file_name, pages


def get_loader(file_path: str) -> PageLoader:
    if file_path.endswith(('.md', '.txt')):
        return MarkdownPageLoader()
    if file_path.endswith('.json'):
        return JsonPageLoader()
    if file_path.endswith('.jsonl'):
        return JsonlPageLoader()
    raise ValueError(f'Unsupported file type: {file_path}')
