import ollama


def prompt_model(message: str, model: str, system_prompt: str = None) -> str:
    # Initialize the Ollama client
    client = ollama.Client()

    if system_prompt and len(system_prompt) > 0:
        try:
            with open(system_prompt, "r", encoding="utf-8") as file:
                system_prompt = file.read().strip()
        except FileNotFoundError:
            print(f"Loading '{system_prompt}' as the system prompt.")
    else:
        system_prompt = "You are a professional AI assistant."

    # Create the message payload with system prompt
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': message}
    ]

    # Generate response using the specified model with additional options
    response = client.chat(
        model=model,
        messages=messages,
        options={"num_ctx": 32768, "num_predict": -1}  # Setting context length
    )

    # Extract and return the content of the assistant's reply
    return response['message']['content']
