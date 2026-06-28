# CAG LLM Grounding with RAG

This Python script demonstrates Contextual Augmentation and Grounding (CAG) for LLMs using a simple Retrieval Augmented Generation (RAG) approach. It fetches context from a Wikipedia page, indexes it, and then uses it to ground LLM responses to user questions, reducing hallucinations.

## Language

`python`

## How to Run

1. Install required libraries: pip install langchain-openai langchain-text-splitters langchain-community python-dotenv
2. Create a .env file with your OPENAI_API_KEY.
3. Run the script: python cag_llm_grounding.py

## Original Article

This example accompanies the Turkish article: [CAG: LLM'lerinizi Topraklamanın Daha Basit Yolu](https://fatihsoysal.com/blog/cag-llmlerinizi-topraklamanin-daha-basit-yolu/).

## License

MIT — see [LICENSE](LICENSE).
