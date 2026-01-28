"""
TDX API Wrapper that simplifies interactions with the TDX KB articles API
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from core.config import get_tdx_settings
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from collections import deque
from threading import Lock
from utils.logger import get_logger, PerformanceLogger

logger = get_logger(__name__)


class TDXClient:
    """
    Retrieves and processes content from our TDX Knowledge base.
    """

    def __init__(self):
        settings = get_tdx_settings()
        self.base_url = settings.BASE_URL
        self.app_id = settings.APP_ID
        self.web_services_key = settings.WEBSERVICES_KEY.get_secret_value()
        self.beid = settings.BEID.get_secret_value()
        self.bearer_token: Optional[str] = None
        """
        Initialize the TDX KB Client wrapper.

        Args:
            base_url: The base URL for your TeamDynamix instance
                     (e.g., 'https://your-instance.teamdynamix.com')
            beid: Business Entity Identifier
            webserviceskey: Second Identifier required for bearer token generation
            app_id: The Client Portal application ID
        """
        logger.info(f"Initializing TDXClient for {self.base_url}")

        if not settings.BASE_URL:
            logger.error("BASE_URL is not configured")
            raise ValueError("BASE_URL is required")
        if not settings.APP_ID:
            logger.error("APP_ID is not configured")
            raise ValueError("APP_ID is required")
        if not settings.WEBSERVICES_KEY:
            logger.error("WEBSERVICES_KEY is not configured")
            raise ValueError("WEBSERVICES_KEY is required")
        if not settings.BEID:
            logger.error("BEID is not configured")
            raise ValueError("BEID is required")

        self.rate_limiter = RateLimiter()
        self.session = self._create_session()
        logger.debug(f"TDXClient initialized successfully for app_id={self.app_id}")

    def _create_session(self) -> Session:
        """
        Initialize a shared requests.Session with retry configuration so we reuse
        connections and recover from transient socket closures.
        """
        session = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"Content-Type": "application/json"})
        return session

    def _reset_session(self) -> None:
        """Dispose of the current session and build a fresh one, preserving auth."""
        self.session.close()
        self.session = self._create_session()
        if self.bearer_token:
            self.session.headers.update(
                {"Authorization": f"Bearer {self.bearer_token}"}
            )

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Issue an HTTP request via the shared session with a simple retry that
        recreates the session on connection failures.
        """
        timeout = kwargs.pop("timeout", 30.0)
        for attempt in range(2):
            try:
                return self.session.request(
                    method=method, url=url, timeout=timeout, **kwargs
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ):
                if attempt == 0:
                    self._reset_session()
                    continue
                raise
        raise RuntimeError("Exceeded retry attempts for HTTP request")

    def close(self) -> None:
        """Explicitly close the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> "TDXClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def authenticate(self) -> str:
        """
        Authenticates with TeamDynamix API and returns a bearer token for future api calls.
        The bearer token has a validity of 24 hours.

        Returns:
            str: The Bearer Token (JWT) if authentication is successful

        Raises:
            requests.exceptions.RequestException: If the API call fails
            ValueError: If authentication fails (invalid credentials)

        Note:
            The admin service account must be set to Active in TDAdmin.
            The BEID and WebServicesKey are available in TDAdmin's organization
            detail page to administrators with "Add BE Administrators" permission.
        """
        if not self.bearer_token:
            logger.info("Authenticating with TDX API")
            auth_url = f"{self.base_url}/TDWebApi/api/auth/loginadmin"
            payload = {"BEID": self.beid, "WebServicesKey": self.web_services_key}

            try:
                with PerformanceLogger(
                    logger, "TDX authentication", level=logger.level
                ):
                    response = self._request("POST", auth_url, json=payload)

                if response.status_code == 200:
                    self.bearer_token = response.text.strip()
                    self.session.headers.update(
                        {"Authorization": f"Bearer {self.bearer_token}"}
                    )
                    logger.info("Successfully authenticated with TDX API")
                    logger.debug(f"Bearer token: {self.bearer_token[:20]}...")
                    return self.bearer_token
                else:
                    logger.error(
                        f"Authentication failed with status code: {response.status_code}"
                    )
                    logger.debug(f"Response: {response.text}")
                    raise requests.exceptions.RequestException(
                        f"Authentication failed with status code: {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error during authentication: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during authentication: {str(e)}")
                raise RuntimeError(f"Authentication failed: {str(e)}") from e
        else:
            logger.debug("Using existing bearer token")

        return self.bearer_token

    def list_article_ids(self) -> List[int]:
        """
        Retrieve list of all article IDs from TDX API, excluding phishing category.

        Returns:
            List of article IDs (excluding "Recent Phishing Emails" category)

        Raises:
            RuntimeError: If request fails after retries
        """
        logger.info("Fetching article IDs from TDX API")

        if not self.bearer_token:
            self.authenticate()

        try:
            article_ids = []
            filtered_count = 0
            search_url = (
                f"{self.base_url}/TDWebApi/api/{self.app_id}/knowledgebase/search"
            )
            payload = {"ReturnCount": 10000}

            logger.debug(f"Requesting article list from {search_url}")
            with PerformanceLogger(logger, "Fetch article IDs list"):
                response = self._request("POST", search_url, json=payload)

            if response.status_code == 401:
                logger.warning("Token expired, re-authenticating")
                self.bearer_token = None
                self.authenticate()
                response = self._request("POST", search_url, json=payload)

            if response.status_code == 200:
                articles_data = response.json()
                for article in articles_data:
                    article_id = article.get("ID")
                    category_name = article.get("CategoryName")

                    # Filter out phishing category early to reduce API calls
                    if category_name == "Recent Phishing Emails":
                        logger.debug(f"Filtering out phishing article {article_id}")
                        filtered_count += 1
                        continue

                    if article_id:
                        article_ids.append(article_id)
                    else:
                        logger.warning(f"Article missing ID: {article}")

                logger.info(
                    f"Successfully retrieved {len(article_ids)} article IDs "
                    f"({filtered_count} phishing articles filtered)"
                )
                return article_ids
            else:
                logger.error(f"Request failed with status code: {response.status_code}")
                logger.debug(f"Response: {response.text}")
                raise RuntimeError(
                    f"Request failed with status code: {response.status_code}"
                )

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while fetching article IDs: {str(e)}")
            raise RuntimeError(f"Failed to fetch article IDs: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error while fetching article IDs: {str(e)}")
            raise RuntimeError(f"Failed to fetch article IDs: {str(e)}") from e

    def retrieve_all_articles(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        """
        Retrieve all articles from TDX API with retry logic.

        Returns:
            Tuple of (successful_articles, skipped_articles_with_reasons)

        Raises:
            RuntimeError: If unable to fetch article list
        """
        logger.info("Starting retrieval of all articles from TDX API")

        if not self.bearer_token:
            self.authenticate()

        articles = []
        skipped_articles = []

        try:
            article_ids = self.list_article_ids()
            logger.info(
                f"Retrieved {len(article_ids)} article IDs, fetching full content"
            )

            with PerformanceLogger(logger, f"Fetch {len(article_ids)} full articles"):
                for idx, article_id in enumerate(article_ids, 1):
                    if idx % 50 == 0:
                        logger.info(
                            f"Progress: {idx}/{len(article_ids)} articles fetched"
                        )

                    article_url = f"{self.base_url}/TDWebApi/api/{self.app_id}/knowledgebase/{article_id}"
                    max_retries = 3
                    retry_delay = 2.0

                    for attempt in range(max_retries):
                        try:
                            self.rate_limiter.acquire()
                            response = self._request("GET", article_url)

                            if (
                                response.status_code == 401
                                and attempt < max_retries - 1
                            ):
                                logger.warning(
                                    f"Token expired while fetching article {article_id}, re-authenticating"
                                )
                                self.bearer_token = None
                                self.authenticate()
                                continue

                            if response.status_code == 200:
                                articles.append(response.json())
                                logger.debug(
                                    f"Successfully fetched article {article_id}"
                                )
                                break  # Success!

                            elif response.status_code == 429:
                                # Rate limit error: Apply exponential backoff and retry
                                if attempt < max_retries - 1:
                                    wait_time = retry_delay * (2**attempt)
                                    logger.warning(
                                        f"Rate limit hit for article {article_id}, "
                                        f"waiting {wait_time}s (attempt {attempt + 1}/{max_retries})"
                                    )
                                    time.sleep(wait_time)
                                    continue
                                else:
                                    # Failed all attempts due to 429
                                    logger.error(
                                        f"Skipping article {article_id}. Failed all {max_retries} "
                                        f"attempts due to rate limiting (429)"
                                    )
                                    skipped_articles.append(
                                        (article_id, "Rate Limit Failure")
                                    )
                                    break

                            else:
                                # Non-recoverable error
                                logger.error(
                                    f"Skipping article {article_id}. Non-recoverable error on attempt {attempt + 1}. "
                                    f"Status code: {response.status_code}. Response: {response.text[:200]}"
                                )
                                skipped_articles.append(
                                    (article_id, f"HTTP Error {response.status_code}")
                                )
                                break

                        except requests.exceptions.RequestException as e:
                            logger.error(
                                f"Network error fetching article {article_id}: {str(e)}"
                            )
                            if attempt == max_retries - 1:
                                skipped_articles.append(
                                    (article_id, f"Network Error: {str(e)}")
                                )
                            continue

            logger.info(
                f"Article retrieval complete: {len(articles)} retrieved, "
                f"{len(skipped_articles)} skipped"
            )

            if skipped_articles:
                logger.warning(f"Skipped articles: {skipped_articles}")

            return articles, skipped_articles

        except Exception as e:
            logger.error(f"Fatal error during article retrieval: {str(e)}")
            raise RuntimeError(f"Failed to retrieve articles: {str(e)}") from e


"""
Rate limiter for TDX API to handle 60 requests per 60 seconds limit.

This module provides a thread-safe rate limiter that ensures API calls
stay within the TDX API rate limit of 60 requests per 60 seconds.
"""


class RateLimiter:
    """
    Thread-safe rate limiter using a sliding window algorithm.

    The TDX API allows 60 requests per 60 seconds. This rate limiter
    tracks request timestamps and enforces delays when necessary.
    """

    def __init__(
        self,
        max_requests: int = 60,
        time_window: float = 60.0,
        safety_buffer: float = 0.5,
    ):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed (default: 60)
            time_window: Time window in seconds (default: 60.0)
            safety_buffer: Extra time to add when waiting (default: 0.5 seconds)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.safety_buffer = safety_buffer
        self.request_times = deque()
        self.lock = Lock()
        self._total_requests = 0
        self._total_wait_time = 0.0

    def acquire(self) -> float:
        """
        Acquire permission to make a request, waiting if necessary.

        This method blocks until a request can be made within the rate limit.
        It uses a sliding window algorithm to track requests over time.

        Returns:
            float: Time in seconds that was waited (0 if no wait was needed)

        Example:
            >>> limiter = RateLimiter(max_requests=60, time_window=60.0)
            >>> wait_time = limiter.acquire()
            >>> # Now safe to make API request
            >>> response = api_call()
        """
        with self.lock:
            current_time = time.time()

            # Remove requests outside the time window
            while (
                self.request_times
                and current_time - self.request_times[0] >= self.time_window
            ):
                self.request_times.popleft()

            # Check if we need to wait
            wait_time = 0.0
            if len(self.request_times) >= self.max_requests:
                # Calculate how long to wait (with safety buffer)
                oldest_request = self.request_times[0]
                time_to_wait = (
                    self.time_window
                    - (current_time - oldest_request)
                    + self.safety_buffer
                )

                if time_to_wait > 0:
                    wait_time = time_to_wait
                    # Release lock while sleeping to avoid blocking other threads
                    self.lock.release()
                    time.sleep(wait_time)
                    self.lock.acquire()

                    # Update current time after waiting
                    current_time = time.time()

                    # Clean up old requests again after waiting
                    while (
                        self.request_times
                        and current_time - self.request_times[0] >= self.time_window
                    ):
                        self.request_times.popleft()

            # Record this request
            self.request_times.append(current_time)
            self._total_requests += 1
            self._total_wait_time += wait_time

            return wait_time

    def reset(self):
        """
        Reset the rate limiter, clearing all tracked requests and statistics.

        Example:
            >>> limiter = RateLimiter()
            >>> # ... make some requests ...
            >>> limiter.reset()  # Start fresh
        """
        with self.lock:
            self.request_times.clear()
            self._total_requests = 0
            self._total_wait_time = 0.0

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
