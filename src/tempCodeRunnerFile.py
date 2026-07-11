 PROMPT_TEMPLATE = """You are a helpful assistant answering questions based only on the provided context.

# Context:
# {context}

# Question: {question}

# Instructions:
# - Answer using only the information in the context above.
# - If the context does not contain enough information to answer, say "I don't have enough information in the document to answer that."
# - Be concise.

# Answer:"""