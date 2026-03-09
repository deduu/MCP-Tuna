from typing import List, Dict
import json
import os


class PageLoader:
    def load(self, file_path: str) -> tuple[str, List[Dict]]:
        raise NotImplementedError


class MarkdownPageLoader(PageLoader):
    def load(self, file_path: str) -> tuple[str, List[Dict]]:
        file_name = os.path.basename(file_path).rsplit('.', 1)[0]

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        pages = []
        for i, page in enumerate(text.split('\n\n---\n\n')):
            if page.strip():
                pages.append({
                    'index': i,
                    'markdown': page.strip(),
                })

        return file_name, pages


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
