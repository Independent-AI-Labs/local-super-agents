import ollama


def prompt_model(message: str, model: str) -> str:
    # Initialize the Ollama client
    client = ollama.Client()

    # Read the system prompt from file
    system_prompt_path = "..\prompts\SYSTEM"
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as file:
            system_prompt = file.read().strip()
    except FileNotFoundError:
        system_prompt = "You are an AI assistant."

    # Create the message payload with system prompt
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': message}
    ]

    # Generate response using the specified model with additional options
    response = client.chat(
        model=model,
        messages=messages,
        options={"context_length": 32768}  # Setting context length
    )

    # Extract and return the content of the assistant's reply
    return response['message']['content']
