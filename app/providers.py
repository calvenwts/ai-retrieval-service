from abc import ABC, abstractmethod

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


class ChatRequest(BaseModel):
    messages: list[dict]
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    stream: bool = False


class ChatResponse(BaseModel):
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(ABC):
    @abstractmethod
    async def complete(self, req: ChatRequest) -> ChatResponse: ...


class AnthropicProvider(LLMProvider):
    def __init__(self):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete(self, req: ChatRequest) -> ChatResponse:
        resp = await self.client.messages.create(
            model=req.model,
            max_tokens=req.max_tokens,
            messages=req.messages,
        )
        return ChatResponse(
            text=resp.content[0].text,
            model=req.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete(self, req: ChatRequest) -> ChatResponse:
        resp = await self.client.chat.completions.create(
            model=req.model,
            max_tokens=req.max_tokens,
            messages=req.messages,
        )
        usage = resp.usage
        return ChatResponse(
            text=resp.choices[0].message.content,
            model=req.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


class OllamaProvider(LLMProvider):
    def __init__(self):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def complete(self, req: ChatRequest) -> ChatResponse:
        resp = await self.client.chat.completions.create(
            model=req.model,
            max_tokens=req.max_tokens,
            messages=req.messages,
        )
        usage = resp.usage
        return ChatResponse(
            text=resp.choices[0].message.content,
            model=req.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


def _resolve_provider(model: str) -> str:
    if "claude" in model:
        return "anthropic"
    if "llama" in model or "mistral" in model or "gemma" in model or "qwen" in model:
        return "ollama"
    return "openai"


class LLMRouter:

    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}

    def _get_provider(self, name: str) -> LLMProvider:
        if name not in self._providers:
            if name == "anthropic":
                self._providers[name] = AnthropicProvider()
            elif name == "ollama":
                self._providers[name] = OllamaProvider()
            else:
                self._providers[name] = OpenAIProvider()
        return self._providers[name]

    async def complete(
        self, req: ChatRequest, fallback_model: str | None = None
    ) -> ChatResponse:
        provider_name = _resolve_provider(req.model)
        provider = self._get_provider(provider_name)
        try:
            return await provider.complete(req)
        except Exception:
            if fallback_model is None:
                raise
            fallback_req = req.model_copy(update={"model": fallback_model})
            fallback_name = _resolve_provider(fallback_model)
            fallback_provider = self._get_provider(fallback_name)
            return await fallback_provider.complete(fallback_req)


router = LLMRouter()
