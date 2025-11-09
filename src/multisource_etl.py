import os
from typing import List, Dict, Any
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transform_rules import TransformRules, DocumentMetadata
from vectorstore import VectorStore, DocumentChunk
from github_client import GitHubClient


class MultiSourceETL:

    def __init__(self, config: Dict[str, Any], vector_store: VectorStore):
        self.config = config
        self.vector_store = vector_store
        self.transform_rules = TransformRules(config.get('data_sources', {}))

        chunk_size = config.get('embedding', {}).get('chunk_size', 1000)
        chunk_overlap = config.get('embedding', {}).get('chunk_overlap', 200)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

    def process_data_sources(self, data_sources_path: str):
        base_path = Path(data_sources_path)

        for source_dir in ['hr', 'github', 'ats']:
            if source_dir == 'github':
                # Handle GitHub via API
                if self.config.get('data_sources', {}).get('github', {}).get('api_enabled', False):
                    self._process_github_api()
            else:
                # Handle file-based sources
                source_path = base_path / source_dir
                if source_path.exists():
                    self._process_source_directory(source_path, source_dir)

    def _process_source_directory(self, source_path: Path, source_system: str):
        chunks = []

        for file_path in source_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md', '.json']:
                file_chunks = self._process_file(file_path, source_system)
                chunks.extend(file_chunks)

        if chunks:
            self.vector_store.add_documents(chunks)

    def _process_file(self, file_path: Path, source_system: str) -> List[DocumentChunk]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            transformed_content, metadata = self.transform_rules.apply_transform(
                content, source_system, str(file_path)
            )

            text_chunks = self.text_splitter.split_text(transformed_content)

            document_chunks = []
            for i, chunk in enumerate(text_chunks):
                chunk_metadata = DocumentMetadata(
                    source_system=metadata.source_system,
                    classification=metadata.classification,
                    weight=metadata.weight,
                    file_path=f"{metadata.file_path}#chunk_{i}",
                    pii_masked=metadata.pii_masked
                )

                document_chunks.append(DocumentChunk(
                    content=chunk,
                    metadata=chunk_metadata
                ))

            return document_chunks

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []

    def _process_github_api(self):
        try:
            github_config = self.config.get('github', {})
            github_client = GitHubClient(github_config)

            profile_data = github_client.fetch_profile_data()
            chunks = self._convert_github_profile_to_chunks(profile_data)

            if chunks:
                self.vector_store.add_documents(chunks)
                print(f"Processed GitHub profile data: {len(chunks)} chunks")

        except Exception as e:
            print(f"Error processing GitHub API data: {e}")

    def _convert_github_profile_to_chunks(self, profile_data) -> List[DocumentChunk]:
        chunks = []

        # Profile summary chunk
        profile_text = f"""GitHub Profile: {profile_data.login}
Name: {profile_data.name or 'Not specified'}
Bio: {profile_data.bio or 'Not specified'}
Location: {profile_data.location or 'Not specified'}
Company: {profile_data.company or 'Not specified'}
Blog: {profile_data.blog or 'Not specified'}
Public Repositories: {profile_data.public_repos}
Followers: {profile_data.followers}
Following: {profile_data.following}"""

        profile_chunks = self._create_chunks_from_content(
            profile_text, "github", f"profile_{profile_data.login}"
        )
        chunks.extend(profile_chunks)

        # Repository chunks
        for repo in profile_data.repositories:
            repo_text = f"""Repository: {repo.name}
Description: {repo.description or 'No description'}
Primary Language: {repo.language or 'Not specified'}
Stars: {repo.stargazers_count}
Forks: {repo.forks_count}
Topics: {', '.join(repo.topics) if repo.topics else 'None'}"""

            if repo.languages:
                total_bytes = sum(repo.languages.values())
                language_percentages = {lang: (bytes_count / total_bytes) * 100
                                        for lang, bytes_count in repo.languages.items()}
                repo_text += f"\nLanguage Breakdown: {language_percentages}"

            if repo.readme_content:
                repo_text += f"\nREADME Content: {repo.readme_content}"

            repo_chunks = self._create_chunks_from_content(
                repo_text, "github", f"repo_{repo.name}"
            )
            chunks.extend(repo_chunks)

        return chunks

    def _create_chunks_from_content(self, content: str, source_system: str, file_path: str) -> List[DocumentChunk]:
        transformed_content, metadata = self.transform_rules.apply_transform(
            content, source_system, file_path
        )

        text_chunks = self.text_splitter.split_text(transformed_content)

        document_chunks = []
        for i, chunk in enumerate(text_chunks):
            chunk_metadata = DocumentMetadata(
                source_system=metadata.source_system,
                classification=metadata.classification,
                weight=metadata.weight,
                file_path=f"{metadata.file_path}#chunk_{i}",
                pii_masked=metadata.pii_masked
            )

            document_chunks.append(DocumentChunk(
                content=chunk,
                metadata=chunk_metadata
            ))

        return document_chunks

    def process_uploaded_file(self, content: str, filename: str, source_system: str) -> bool:
        try:
            transformed_content, metadata = self.transform_rules.apply_transform(
                content, source_system, filename
            )

            text_chunks = self.text_splitter.split_text(transformed_content)

            document_chunks = []
            for i, chunk in enumerate(text_chunks):
                chunk_metadata = DocumentMetadata(
                    source_system=metadata.source_system,
                    classification=metadata.classification,
                    weight=metadata.weight,
                    file_path=f"{filename}#chunk_{i}",
                    pii_masked=metadata.pii_masked
                )

                document_chunks.append(DocumentChunk(
                    content=chunk,
                    metadata=chunk_metadata
                ))

            self.vector_store.add_documents(document_chunks)
            return True

        except Exception as e:
            print(f"Error processing uploaded file {filename}: {e}")
            return False