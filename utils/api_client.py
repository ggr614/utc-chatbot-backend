"""
TDX API Wrapper that simplifies interactions with the TDX KB articles API
"""

import time
from typing import Any, Dict, List, Optional
from core.config import get_settings
import requests
from collections import deque
from threading import Lock


class TDXClient:
    """
    Retrieves and processes content from our TDX Knowledge base.
    """

    def __init__(
        self,
        base_url: str = get_settings().BASE_URL,
        app_id: int = get_settings().APP_ID,
        web_services_key: str = get_settings().WEBSERVICES_KEY.get_secret_value(),
        beid: str = get_settings().BEID.get_secret_value(),
    ):
        """
        Initialize the TDX KB Client wrapper.

        Args:
            base_url: The base URL for your TeamDynamix instance
                     (e.g., 'https://your-instance.teamdynamix.com')
            beid: Business Entity Identifier
            webserviceskey: Second Identifier required for bearer token generation
            app_id: The Client Portal application ID
        """
        self.base_url = base_url
        self.app_id = app_id
        self.web_services_key = web_services_key
        self.beid = beid
        self.bearer_token: Optional[str] = None
        self.rate_limiter = RateLimiter()

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
            auth_url = f"{self.base_url}/TDWebApi/api/auth/loginadmin"
            payload = {"BEID": self.beid, "WebServicesKey": self.web_services_key}
            headers = {"Content-Type": "application/json"}
            response = requests.post(auth_url, json=payload, headers=headers)
            if response.status_code == 200:
                self.bearer_token = response.text.strip()
                return self.bearer_token
            else:
                raise requests.exceptions.RequestException(
                    f"Authentication failed with status code: {response.status_code}"
                )
        return self.bearer_token

    def list_article_ids(self) -> List[str]:
        if not self.bearer_token:
            self.authenticate()

        article_ids = []
        search_url = f"{self.base_url}/TDWebApi/api/{self.app_id}/knowledgebase/search"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }
        payload = {"ReturnCount": 10000}
        response = requests.post(search_url, json=payload, headers=headers)
        if response.status_code == 200:
            for article in response.json():
                article_ids.append(article.get("ID"))
            return article_ids
        else:
            raise requests.exceptions.RequestException(
                f"Request failed with status code: {response.status_code}"
            )

    def retrieve_all_articles(self) -> List[Dict[str, Any]]:
        if not self.bearer_token:
            self.authenticate()

        articles = []
        article_ids = self.list_article_ids()

        for article_id in article_ids:
            article_url = (
                f"{self.base_url}/TDWebApi/api/{self.app_id}/knowledgebase/{article_id}"
            )
            headers = {
                "Authorization": f"Bearer {self.bearer_token}",
                "Content-Type": "application/json",
            }
            max_retries = 3
            retry_delay = 2.0

            for attempt in range(max_retries):
                self.rate_limiter.acquire()
                response = requests.get(article_url, headers=headers)

                if response.status_code == 200:
                    articles.append(response.json())
                    break

                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2**attempt)
                        time.sleep(wait_time)
                        continue
                    else:
                        raise ValueError(
                            f"Failed to get article {article_id} after {max_retries} attempts due to rate limiting (429)."
                        )

                else:
                    raise ValueError(
                        f"Failed to get article {article_id} on attempt {attempt + 1}. "
                        f"Status code: {response.status_code}. Response: {response.text}"
                    )

        return articles


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
