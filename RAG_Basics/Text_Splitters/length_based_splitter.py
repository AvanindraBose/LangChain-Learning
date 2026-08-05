from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 0,
    separator=""
)

docs = splitter.split_text(''' 
                    Artificial Intelligence (AI) is transforming industries by enabling machines to perform tasks that typically require human intelligence. These tasks include understanding natural language, recognizing images, making predictions, and generating content. Machine Learning, a subset of AI, allows systems to learn patterns from historical data without being explicitly programmed. Deep Learning further extends this capability using neural networks with multiple layers, making significant advances in fields such as computer vision and speech recognition.

In recent years, Large Language Models (LLMs) have become increasingly popular because of their ability to understand and generate human-like text. Models such as GPT, Llama, and Mistral can answer questions, summarize documents, generate code, and assist with a wide variety of knowledge-based tasks. However, these models are limited by the information they were trained on and may not always have access to the latest or domain-specific knowledge.

Retrieval-Augmented Generation (RAG) addresses this limitation by combining information retrieval with language generation. Instead of relying solely on the model's internal knowledge, a RAG system retrieves relevant documents from an external knowledge base and provides them as context to the language model. This approach improves factual accuracy, reduces hallucinations, and allows organizations to build AI applications using their own private data. A typical RAG pipeline consists of document loading, text splitting, embedding generation, vector storage, similarity search, retrieval, prompt construction, and response generation.

When building production-grade AI systems, it is important to consider scalability, security, and observability. Technologies such as Docker, Kubernetes, FastAPI, PostgreSQL, Redis, and cloud platforms like AWS are commonly used to deploy and manage these applications. Monitoring tools such as Prometheus and Grafana help track application performance, while CI/CD pipelines automate testing and deployment, ensuring that updates can be delivered reliably and efficiently.
''')

print(len(docs))

