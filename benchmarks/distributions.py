"""Prompt distributions for benchmark testing.

Each distribution generates prompts representative of a real use case.
Hit rates are measured against expected ranges from the project spec.
"""
from __future__ import annotations

import random


def faq_bot_prompts(n: int, seed: int = 42) -> list[str]:
    """FAQ/support bot distribution — high repetition, 40-70% hit rate expected.

    Uses a small pool of canonical questions with minor paraphrase variations.
    """
    rng = random.Random(seed)
    canonical = [
        "How do I reset my password?",
        "What are your business hours?",
        "How can I contact support?",
        "What is your refund policy?",
        "How do I cancel my subscription?",
        "Where can I find my invoice?",
        "How do I update my billing information?",
        "What payment methods do you accept?",
    ]
    variations = {
        "How do I reset my password?": [
            "How do I reset my password?",
            "I forgot my password, how do I reset it?",
            "Can you help me reset my password?",
            "How to reset password?",
        ],
        "What are your business hours?": [
            "What are your business hours?",
            "When are you open?",
            "What hours do you operate?",
            "What are your opening hours?",
        ],
        "How can I contact support?": [
            "How can I contact support?",
            "How do I reach customer support?",
            "What is your support email?",
            "How do I get help?",
        ],
        "What is your refund policy?": [
            "What is your refund policy?",
            "Can I get a refund?",
            "How do refunds work?",
            "What is the return policy?",
        ],
        "How do I cancel my subscription?": [
            "How do I cancel my subscription?",
            "How can I cancel my account?",
            "I want to cancel, how do I do that?",
            "Cancel subscription instructions?",
        ],
        "Where can I find my invoice?": [
            "Where can I find my invoice?",
            "How do I download my invoice?",
            "Where are my billing documents?",
            "How to get invoice?",
        ],
        "How do I update my billing information?": [
            "How do I update my billing information?",
            "How do I change my credit card?",
            "How to update payment method?",
            "Change billing details?",
        ],
        "What payment methods do you accept?": [
            "What payment methods do you accept?",
            "Do you accept PayPal?",
            "Can I pay with credit card?",
            "What forms of payment are accepted?",
        ],
    }
    result = []
    for _ in range(n):
        q = rng.choice(canonical)
        result.append(rng.choice(variations[q]))
    return result


def summarization_prompts(n: int, seed: int = 42) -> list[str]:
    """Document summarization distribution — 20-50% hit rate expected.

    Uses a small set of document topics with template prompts.
    """
    rng = random.Random(seed)
    docs = [
        "the quarterly earnings report for Q3 2024",
        "the product roadmap document",
        "the terms of service agreement",
        "the employee handbook",
        "the technical specification for the API",
    ]
    templates = [
        "Please summarize {}.",
        "Give me a brief summary of {}.",
        "What are the key points in {}?",
        "Summarize the main ideas in {}.",
    ]
    result = []
    for _ in range(n):
        doc = rng.choice(docs)
        tmpl = rng.choice(templates)
        result.append(tmpl.format(doc))
    return result


def general_chat_prompts(n: int, seed: int = 42) -> list[str]:
    """General chat assistant distribution — 5-15% hit rate expected.

    High diversity, dynamic context.
    """
    rng = random.Random(seed)
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "What is the difference between TCP and UDP?",
        "How does photosynthesis work?",
        "What is the meaning of life?",
        "Who wrote Hamlet?",
        "What year did World War II end?",
        "How do I make pasta carbonara?",
        "What is machine learning?",
        "Explain the concept of recursion.",
        "What is the speed of light?",
        "How do neural networks learn?",
        "What is the difference between SQL and NoSQL?",
        "How does GPS work?",
        "What is a blockchain?",
        "How do I sort a list in Python?",
        "What is the time complexity of quicksort?",
        "Explain gradient descent.",
        "What is a REST API?",
        "How does TLS/SSL work?",
    ]
    # Each call picks randomly, ensuring high diversity
    return [rng.choice(prompts) for _ in range(n)]


def code_generation_prompts(n: int, seed: int = 42) -> list[str]:
    """Code generation distribution — 3-10% hit rate expected.

    Strict threshold, exact problem statements vary.
    """
    rng = random.Random(seed)
    tasks = [
        "Write a Python function to reverse a string.",
        "Write a Python class for a binary search tree.",
        "Implement bubble sort in Python.",
        "Write a Python decorator that logs function calls.",
        "Implement a linked list in Python.",
        "Write a Python function to find the nth Fibonacci number.",
        "Implement a stack using two queues in Python.",
        "Write a Python function that validates an email address.",
        "Implement merge sort in Python.",
        "Write a Python context manager for timing code.",
    ]
    return [rng.choice(tasks) for _ in range(n)]
