# Local LLM Operations

Local LLM systems are constrained by RAM, CPU, context window, and disk speed.
Small quantized GGUF models can run on limited hardware, but they need careful
context management.

Practical rules for low-resource AI:
- keep prompts short;
- retrieve only relevant chunks;
- use small max token limits;
- avoid storing long chat history in every request;
- prefer SSD-backed knowledge storage for large document sets;
- clean up memory after inference when running in Python;
- use one or two concurrent requests on weak hardware.

OpenAI-compatible providers expose a common `/chat/completions` API shape.
That allows one application to work with LM Studio, Ollama, vLLM, OpenAI,
OpenRouter, Together AI, Groq, and similar providers by changing `base_url`,
model names, and API key.

