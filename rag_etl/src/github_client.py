import os
import time
from typing import Dict, List, Any, Optional
import requests
from pydantic import BaseModel


class GitHubRepo(BaseModel):
    name: str
    description: Optional[str]
    language: Optional[str]
    stargazers_count: int
    forks_count: int
    topics: List[str]
    languages: Dict[str, int] = {}
    readme_content: Optional[str] = None


class GitHubProfile(BaseModel):
    login: str
    name: Optional[str]
    bio: Optional[str]
    location: Optional[str]
    company: Optional[str]
    blog: Optional[str]
    public_repos: int
    followers: int
    following: int
    repositories: List[GitHubRepo] = []


class GitHubClient:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', 'https://api.github.com')
        self.username = config.get('username')
        self.max_repos = config.get('fetch_options', {}).get('max_repos', 50)
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = time.time() + 3600

        self.token = os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("GITHUB_TOKEN environment variable is required")

        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Portfolio-RAG-System'
        })

    def fetch_profile_data(self) -> GitHubProfile:
        profile_data = self._fetch_user_profile()
        repositories = self._fetch_repositories()

        if self.config.get('fetch_options', {}).get('include_languages', True):
            repositories = self._fetch_repository_languages(repositories)

        if self.config.get('fetch_options', {}).get('include_readme', True):
            repositories = self._fetch_readme_files(repositories)

        profile_data['repositories'] = repositories
        return GitHubProfile(**profile_data)

    def _fetch_user_profile(self) -> Dict[str, Any]:
        url = f"{self.base_url}/users/{self.username}"
        response = self._make_request(url)

        return {
            'login': response['login'],
            'name': response.get('name'),
            'bio': response.get('bio'),
            'location': response.get('location'),
            'company': response.get('company'),
            'blog': response.get('blog'),
            'public_repos': response.get('public_repos', 0),
            'followers': response.get('followers', 0),
            'following': response.get('following', 0)
        }

    def _fetch_repositories(self) -> List[GitHubRepo]:
        url = f"{self.base_url}/users/{self.username}/repos"
        params = {
            'sort': 'updated',
            'direction': 'desc',
            'per_page': min(self.max_repos, 100)
        }

        response = self._make_request(url, params=params)

        repositories = []
        for repo_data in response[:self.max_repos]:
            if not repo_data.get('fork', False):  # Skip forked repositories
                repo = GitHubRepo(
                    name=repo_data['name'],
                    description=repo_data.get('description'),
                    language=repo_data.get('language'),
                    stargazers_count=repo_data.get('stargazers_count', 0),
                    forks_count=repo_data.get('forks_count', 0),
                    topics=repo_data.get('topics', [])
                )
                repositories.append(repo)

        return repositories

    def _fetch_repository_languages(self, repositories: List[GitHubRepo]) -> List[GitHubRepo]:
        for repo in repositories:
            try:
                url = f"{self.base_url}/repos/{self.username}/{repo.name}/languages"
                languages = self._make_request(url)
                repo.languages = languages if languages else {}
            except Exception as e:
                print(f"Failed to fetch languages for {repo.name}: {e}")
                repo.languages = {}

        return repositories

    def _fetch_readme_files(self, repositories: List[GitHubRepo]) -> List[GitHubRepo]:
        # Only fetch README for top repositories (by stars + activity)
        top_repos = sorted(repositories, key=lambda r: r.stargazers_count, reverse=True)[:10]

        for repo in top_repos:
            try:
                url = f"{self.base_url}/repos/{self.username}/{repo.name}/readme"
                response = self._make_request(url)

                if response and 'content' in response:
                    import base64
                    readme_content = base64.b64decode(response['content']).decode('utf-8')
                    repo.readme_content = readme_content[:2000]  # Limit README length

            except Exception as e:
                print(f"Failed to fetch README for {repo.name}: {e}")
                repo.readme_content = None

        return repositories

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Any:
        self._check_rate_limit()

        try:
            response = self.session.get(url, params=params, timeout=10)
            self._update_rate_limit_from_response(response)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                raise Exception(f"Rate limit exceeded or forbidden: {response.status_code}")
            elif response.status_code == 404:
                raise Exception(f"Resource not found: {url}")
            else:
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise Exception(f"GitHub API request failed: {e}")

    def _check_rate_limit(self):
        if self.rate_limit_remaining <= 10 and time.time() < self.rate_limit_reset:
            wait_time = self.rate_limit_reset - time.time()
            print(f"Rate limit low. Waiting {wait_time:.0f} seconds...")
            time.sleep(wait_time + 1)

    def _update_rate_limit_from_response(self, response: requests.Response):
        self.rate_limit_remaining = int(response.headers.get('x-ratelimit-remaining', self.rate_limit_remaining))
        reset_timestamp = response.headers.get('x-ratelimit-reset')
        if reset_timestamp:
            self.rate_limit_reset = int(reset_timestamp)

    def get_rate_limit_status(self) -> Dict[str, Any]:
        return {
            'remaining': self.rate_limit_remaining,
            'reset_time': self.rate_limit_reset,
            'reset_in_seconds': max(0, self.rate_limit_reset - time.time())
        }