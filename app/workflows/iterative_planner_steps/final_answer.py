from llama_index.core import ChatPromptTemplate

final_answer_system = """You are a highly intelligent assistant trained to provide concise and accurate answers. You will be given a context and a user question. Your task is to analyze the context and answer the user question based on the information provided in the context. If the context lacks sufficient information to answer the question, inform the user and suggest what additional details are needed.

Focus solely on the context to form your response. Avoid making assumptions or using external knowledge unless explicitly stated in the context.
Ensure the final answer is clear, relevant, and directly addresses the user’s question.
If the question is ambiguous, ask clarifying questions to ensure accuracy before proceeding."""

final_answer_user = """
Based on this context:
{context}

Answer the following question:
<question>
{question}
</question>

Provide your answer based on the context above explain your reasoning.
If clarification or additional information is needed, explain why and specify what is required.
"""

final_answer_msgs = [
    (
        "system",
        final_answer_system,
    ),
    ("user", final_answer_user),
]

final_answer_prompt = ChatPromptTemplate.from_messages(final_answer_msgs)
