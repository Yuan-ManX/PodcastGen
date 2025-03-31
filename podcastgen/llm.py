import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum
from pydantic import BaseModel, SecretStr

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
from langchain_groq.chat_models import ChatGroq
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import ChatOpenAI


@dataclass
class BaseLLM(ABC):
    """
    Abstract base class for all LLM models.
    """
    model_name: Optional[str] = None
    max_tokens: Optional[int] = 850
    temperature: Optional[float] = 1.0
    streaming: bool = True
    top_p: Optional[float] = 0.9
    kwargs: Dict[str, Any] = field(default_factory=dict)
    json_output: bool = False

    @abstractmethod
    def create_langchain_model(self) -> BaseChatModel:
        """
        Convert to a LangChain-compatible chat model.
        """
        raise NotImplementedError


@dataclass
class OllamaLLM(BaseLLM):
    """LLM implementation for Ollama."""
    model_name: str
    base_url: str = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    max_tokens: Optional[int] = 650

    def create_langchain_model(self) -> ChatOllama:
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            num_predict=self.max_tokens,
            temperature=self.temperature or 0.5,
            verbose=True,
            top_p=self.top_p,
        )


@dataclass
class VertexAnthropicLLM(BaseLLM):
    """LLM implementation for Vertex Anthropic."""
    model_name: str
    project: Optional[str] = os.getenv("VERTEX_PROJECT", "no-project")
    location: Optional[str] = os.getenv("VERTEX_LOCATION", "us-central1")

    def create_langchain_model(self) -> ChatAnthropicVertex:
        return ChatAnthropicVertex(
            model=self.model_name,
            project=self.project,
            location=self.location,
            max_tokens=self.max_tokens,
            streaming=self.streaming,
            kwargs=self.kwargs,
            top_p=self.top_p,
            temperature=self.temperature or 0.5,
        )


@dataclass
class LiteLLMLLM(BaseLLM):
    """LLM implementation for LiteLLM."""
    model_name: str

    def create_langchain_model(self) -> ChatLiteLLM:
        return ChatLiteLLM(
            model=self.model_name,
            temperature=self.temperature or 0.5,
            max_tokens=self.max_tokens,
            streaming=self.streaming,
            top_p=self.top_p,
        )


@dataclass
class VertexAILLM(BaseLLM):
    """LLM implementation for Vertex AI."""
    model_name: str
    project: Optional[str] = os.getenv("VERTEX_PROJECT", "no-project")
    location: Optional[str] = os.getenv("VERTEX_LOCATION", "us-central1")

    def create_langchain_model(self) -> ChatVertexAI:
        return ChatVertexAI(
            model=self.model_name,
            streaming=self.streaming,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            location=self.location,
            project=self.project,
            safety_settings=None,
            temperature=self.temperature or 0.5,
        )


@dataclass
class GeminiLLM(BaseLLM):
    """LLM implementation for Gemini."""
    model_name: str

    def create_langchain_model(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature or 0.5,
        )


@dataclass
class OpenRouterLLM(BaseLLM):
    """LLM implementation for OpenRouter."""
    model_name: str

    def create_langchain_model(self) -> ChatOpenAI:
        model_args = self.kwargs.copy()
        if self.json_output:
            model_args["response_format"] = {"type": "json_object"}
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature or 0.5,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            max_tokens=self.max_tokens,
            model_kwargs=model_args,
            streaming=self.streaming,
            api_key=SecretStr(os.getenv("OPENROUTER_API_KEY", "openrouter")),
            top_p=self.top_p,
        )


@dataclass
class GroqLLM(BaseLLM):
    """LLM implementation for Groq."""
    model_name: str

    def create_langchain_model(self) -> ChatGroq:
        return ChatGroq(
            model=self.model_name,
            temperature=self.temperature or 0.5,
            max_tokens=self.max_tokens,
            model_kwargs=self.kwargs,
        )


@dataclass
class AnthropicLLM(BaseLLM):
    """LLM implementation for Anthropic."""
    model_name: str

    def create_langchain_model(self) -> ChatAnthropic:
        return ChatAnthropic(
            model_name=self.model_name,
            max_tokens_to_sample=self.max_tokens or 850,
            model_kwargs=self.kwargs,
            streaming=self.streaming,
            timeout=30,
            top_p=self.top_p,
            temperature=self.temperature or 0.5,
        )

class LLM(str, Enum):
    OPENAI = 'openai'
    LMSTUDIO = 'lmstudio'
    OLLAMA = 'ollama'
    GROQ = 'groq'
    AZURE = 'azure'
    GOOGLE = 'google'
    ANTHROPIC = 'anthropic'
    ELEVENLABS = 'elevenlabs'
    CUSTOM = 'custom'

class ClientConfig(BaseModel):
    name: Provider
    key: Optional[str] = None
    endpoint: Optional[str] = None
    version: Optional[str] = None

def initialize_client(config: Optional[Dict[str, Any]] = None) -> Any:
    if not config:
        raise ValueError("Configuration must be provided.")
    
    try:
        client_config = ClientConfig(**config)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    api_key = client_config.key

    if client_config.name == LLM.OPENAI:
        if not api_key:
            raise ValueError("API key is required for OpenAI provider.")
        return OpenAI(api_key=api_key)
    
    elif client_config.name == LLM.LMSTUDIO:
        if not api_key:
            raise ValueError("API key is required for LMStudio provider.")
        return OpenAI(base_url='http://localhost:1234/v1', api_key=api_key)
    
    elif client_config.name == LLM.OLLAMA:
        if not api_key:
            raise ValueError("API key is required for Ollama provider.")
        return OpenAI(base_url='http://localhost:11434/v1', api_key=api_key)
    
    elif client_config.name == LLM.GROQ:
        if not api_key:
            raise ValueError("API key is required for Groq provider.")
        return OpenAI(base_url='https://api.groq.com/openai/v1', api_key=api_key)
    
    elif client_config.name == LLM.AZURE:
        if not all([api_key, client_config.endpoint, client_config.version]):
            raise ValueError("API key, endpoint, and version are required for Azure provider.")
        return AzureOpenAI(
            azure_endpoint=client_config.endpoint,
            api_version=client_config.version,
            api_key=api_key
        )
    
    elif client_config.name == LLM.GOOGLE:
        if not api_key:
            raise ValueError("API key is required for Google provider.")
        return genai.Client(api_key=api_key)
    
    elif client_config.name == LLM.ANTHROPIC:
        if not api_key:
            raise ValueError("API key is required for Anthropic provider.")
        return Anthropic(api_key=api_key)
    
    elif client_config.name == LLM.ELEVENLABS:
        if not api_key:
            raise ValueError("API key is required for ElevenLabs provider.")
        return ElevenLabs(api_key=api_key)
    
    elif client_config.name == LLM.CUSTOM:
        if not all([api_key, client_config.endpoint]):
            raise ValueError("API key and endpoint are required for custom provider.")
        return OpenAI(base_url=client_config.endpoint, api_key=api_key)
    
    else:
        raise ValueError(f"Unsupported provider: {client_config.name}")



