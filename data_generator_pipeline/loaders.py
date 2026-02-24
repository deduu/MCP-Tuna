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


def get_loader(file_path: str) -> PageLoader:
    if file_path.endswith(('.md', '.txt')):
        return MarkdownPageLoader()
    if file_path.endswith(('.json', '.jsonl')):
        return JsonPageLoader()
    raise ValueError(f'Unsupported file type: {file_path}')
