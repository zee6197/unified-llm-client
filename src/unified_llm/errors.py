"""Package specific exception hierarchy."""


class UnifiedLLMError(Exception):
    """Base exception for unified_llm package."""


class ProviderNotAvailable(UnifiedLLMError):
    """Raised when the requested provider cannot be used."""


class UnsupportedProviderError(UnifiedLLMError):
    """Raised when a provider has not been configured."""

    def __init__(self, provider: str) -> None:
        super().__init__(f"Provider '{provider}' is not available.")


class UnsupportedFeatureError(UnifiedLLMError):
    """Raised when a requested feature is unsupported by a provider."""

    def __init__(self, feature: str) -> None:
        super().__init__(f"Feature '{feature}' is not supported.")


class ProviderError(UnifiedLLMError):
    """Represents provider-specific HTTP or API errors."""

    def __init__(self, provider: str, message: str, status_code: int | None = None) -> None:
        suffix = f" (status {status_code})" if status_code is not None else ""
        super().__init__(f"{provider}: {message}{suffix}")
        self.provider = provider
        self.status_code = status_code


class ToolNotAvailableError(UnifiedLLMError):
    """Raised when a provider does not support one or more requested tools."""

    def __init__(self, provider: str, tools: list[str]) -> None:
        joined = ", ".join(sorted(set(tools)))
        super().__init__(f"{provider}: tool(s) not available: {joined}")
