from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

def create_chat_model(config, api_name="openai", openai_platform="agicto", *, temperature: float = 0.2):
    """Create a ChatOllama instance using repository configuration."""
    # print_and_write("Creating Agent with API {0} with temperature {1}...".format(api_name, temperature))
    if api_name == "openai":
        model = config["OPENAI_MODEL"]
        openai_config = config["OPENAI_CONFIG"][openai_platform]
        return ChatOpenAI(
            model=model,
            base_url=openai_config[0],
            api_key=openai_config[1],
            temperature=temperature
        )
    elif api_name == "ollama":
        model = config["OLLAMA_MODEL"]
        if not model:
            raise ValueError("OLLAMA_MODEL must be defined in the configuration.")
        kwargs: Dict[str, Any] = {"model": model, "temperature": temperature}
        base_url = config["OLLAMA_BASE_URL"]
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)