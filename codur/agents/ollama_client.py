#!/usr/bin/env python3
"""
Ollama Helper - Reusable client for interacting with local Ollama models

Usage:
    from ollama_client import OllamaClient

    client = OllamaClient(model="qwen2.5-coder:7b")
    response = client.generate("Write a function to sort a list")
    print(response)
"""

import requests
import json
from typing import List, Dict, Optional, Generator
import time


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:7b",
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        timeout: int = 300  # Increased from 120 to 300 seconds for large contexts
    ):
        """
        Initialize Ollama client

        Args:
            base_url: Ollama server URL
            model: Model name to use
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.options = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        max_retries: int = 3
    ) -> str:
        """
        Generate completion from prompt

        Args:
            prompt: Input prompt
            stream: Whether to stream response
            max_retries: Number of retry attempts

        Returns:
            Generated text
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": self.options
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                if stream:
                    return self._handle_stream(response)
                else:
                    return response.json()["response"]

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed after {max_retries} attempts: {str(e)}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> str:
        """
        Chat-based interaction with conversation history

        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream response

        Returns:
            Assistant's response
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": self.options
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            if stream:
                return self._handle_stream(response)
            else:
                return response.json()["message"]["content"]

        except requests.exceptions.RequestException as e:
            raise Exception(f"Chat request failed: {str(e)}")

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        """
        Generate completion with streaming

        Args:
            prompt: Input prompt

        Yields:
            Chunks of generated text
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": self.options
        }

        try:
            with requests.post(url, json=payload, stream=True, timeout=self.timeout) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                        if chunk.get("done"):
                            break

        except requests.exceptions.RequestException as e:
            raise Exception(f"Stream request failed: {str(e)}")

    def _handle_stream(self, response) -> str:
        """Handle streaming response and return complete text"""
        complete_text = []

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    complete_text.append(chunk["response"])
                elif "message" in chunk:
                    complete_text.append(chunk["message"]["content"])
                if chunk.get("done"):
                    break

        return "".join(complete_text)

    def list_models(self) -> List[Dict]:
        """List available models"""
        url = f"{self.base_url}/api/tags"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to list models: {str(e)}")

    def switch_model(self, model: str):
        """Switch to a different model"""
        self.model = model

    def set_temperature(self, temperature: float):
        """Adjust temperature (0.0 = deterministic, 1.0 = creative)"""
        self.options["temperature"] = temperature


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = OllamaClient(model="qwen2.5-coder:7b")

    # Simple generation
    print("=== Simple Generation ===")
    response = client.generate("Write a Python function to calculate fibonacci numbers")
    print(response)
    print()

    # Streaming generation
    print("=== Streaming Generation ===")
    for chunk in client.stream_generate("Explain how merge sort works"):
        print(chunk, end="", flush=True)
    print("\n")

    # Chat-based interaction
    print("=== Chat Interaction ===")
    conversation = [
        {"role": "user", "content": "I need to implement authentication"},
        {"role": "assistant", "content": "I'll help with authentication. What framework?"},
        {"role": "user", "content": "Flask with JWT tokens"}
    ]
    response = client.chat(conversation)
    print(response)
    print()

    # List available models
    print("=== Available Models ===")
    models = client.list_models()
    for model in models:
        print(f"- {model['name']}")
