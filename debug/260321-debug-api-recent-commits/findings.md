# Debug Findings — api/ Recent Commits

Date: 2026-03-21
Scope: `api/**/*.py`, `api/templates/**`
Iterations: 15

---

## [HIGH] Bug: `chat_service` accessed without dependency injection

- **Location:** `api/routers/openai_compat.py:54`
- **Evidence:** `request.app.state.chat_service` accessed directly instead of using `get_chat_service()` from `api/dependencies.py:400-408`
- **Impact:** If `ChatService` fails to initialize at startup, requests to `/v1/chat/completions` raise unhandled `AttributeError` (500 ISE) instead of a clean 503 with descriptive message. The `get_chat_service` dependency was written for this exact purpose but never wired in.
- **Root cause:** Dependency exists but wasn't used during router implementation
- **Suggested fix:**
  ```python
  # In openai_compat.py
  from api.dependencies import get_chat_service

  @router.post("/chat/completions")
  async def chat_completions(
      body: ChatCompletionRequest,
      request: Request,
      chat_service = Depends(get_chat_service),  # Add this
  ):
      # Remove: chat_service = request.app.state.chat_service
  ```

---

## [MEDIUM] Bug: Session cookie missing `secure=True` flag

- **Location:** `api/routers/auth.py:94-101`
- **Evidence:** `set_cookie()` sets `httponly=True`, `samesite="lax"`, but omits `secure=True`
- **Impact:** JWT session token transmitted over plain HTTP connections, exposing it to network sniffing. In production behind HTTPS, the cookie should only be sent over secure connections.
- **Root cause:** Missing parameter in `set_cookie()` call
- **Suggested fix:** Add `secure=True` to the `set_cookie()` call, or make it configurable via `AuthSettings` for dev/prod flexibility:
  ```python
  response.set_cookie(
      key=settings.COOKIE_NAME,
      value=token,
      httponly=True,
      secure=True,  # Add this
      samesite="lax",
      path="/",
      max_age=settings.TOKEN_EXPIRE_MINUTES * 60,
  )
  ```

---

## [MEDIUM] Bug: Raw exception message leaked to clients in SSE error response

- **Location:** `api/routers/openai_compat.py:116`
- **Evidence:** `"message": str(e)` sends raw exception text to API consumers
- **Impact:** Information disclosure — database errors, file paths, configuration details could be exposed. The exception is already logged server-side on line 112.
- **Root cause:** No sanitization of exception before client response
- **Suggested fix:**
  ```python
  error_data = json.dumps({
      "error": {
          "message": "An internal error occurred",
          "type": "server_error",
          "code": 503,
      }
  })
  ```

---

## [LOW] Bug: Inactive user check reveals username existence

- **Location:** `api/routers/auth.py:74-76`
- **Evidence:** Returns "Account is disabled" for inactive users vs "Invalid username or password" for non-existent users
- **Impact:** Minor user enumeration — attacker can determine if a username exists by checking error messages
- **Suggested fix:** Return the same generic error message for all failure cases, or check `is_active` before password verification

---

## [LOW] Bug: Unsanitized username in log messages (log injection)

- **Location:** `api/routers/auth.py:65,71,75,87`
- **Evidence:** `f"Login attempt for non-existent user: {username}"` — username is raw form input
- **Impact:** Log forging — attacker can inject fake log entries with newlines/control characters
- **Suggested fix:** Strip newlines and control characters from username before logging, or use structured (JSON) logging

---

## [LOW] Bug: Health endpoint exposes raw exception messages without auth

- **Location:** `api/routers/health.py:77,102,143,204`
- **Evidence:** `str(e)` in unauthenticated health responses
- **Impact:** Internal error details (DB host, config) leaked to any caller
- **Suggested fix:** Replace `str(e)` with generic status messages in response; keep detailed errors in logs only

---

## [LOW] Bug: No upper bound on chat messages array length

- **Location:** `api/models/chat.py:18`
- **Evidence:** `messages: list[ChatMessage] = Field(min_length=1)` — no `max_length`
- **Impact:** Resource exhaustion from oversized requests (behind API key auth)
- **Suggested fix:** Add `max_length=100` (or appropriate limit) and content size validation
