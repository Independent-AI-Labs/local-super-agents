import time
import uuid
from typing import List, Dict

import requests

from integration.data.config import OPEN_WEBUI_BASE_URL, OPEN_WEBUI_EMAIL, OPEN_WEBUI_PASSWORD

API_TOKEN = None

CHATS = {}


# TODO Support Websocket (sessions).
# TODO Create models for all objects used below.

def authenticate(email: str = None, password: str = None) -> str:
    """
    Authenticates once and caches the token in the global API_TOKEN.
    """
    global API_TOKEN
    if API_TOKEN is not None:
        return API_TOKEN

    url = f'{OPEN_WEBUI_BASE_URL}/api/v1/auths/signin'
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }
    data = {
        "email": email if email else OPEN_WEBUI_EMAIL,
        "password": password if password else OPEN_WEBUI_PASSWORD
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    token = response.json().get("token")

    API_TOKEN = token
    return token


def create_new_chat(model: str, title: str = "New Chat") -> dict:
    """
    Calls /api/v1/chats/new to create a new chat on the server,
    returns the entire JSON object from the response (which includes 'id').
    """
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    new_chat_payload = {
        "chat": {
            "id": "",
            "title": title,
            "models": [model],
            "params": {},
            "history": {
                "messages": {},
                "currentId": None
            },
            "messages": [],
            "tags": [],
            "timestamp": int(time.time() * 1000)
        }
    }

    resp = requests.post(f"{OPEN_WEBUI_BASE_URL}/api/v1/chats/new", headers=headers, json=new_chat_payload)
    resp.raise_for_status()

    return resp.json()


def update_chat_on_server(chat_id: str, chat_data: dict) -> dict:
    """
    POST /api/v1/chats/{chat_id} with updated chat data.
    This is how we “sync” our local CHATS[chat_id] data back to the server.
    """
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    url = f"{OPEN_WEBUI_BASE_URL}/api/v1/chats/{chat_id}"
    resp = requests.post(url, headers=headers, json={"chat": chat_data})
    resp.raise_for_status()

    return resp.json()


def call_completions(chat_id: str, completion_id: str, messages: List, model: str) -> Dict:
    """
    Calls /api/chat/completions and returns task_id.
    """
    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json',
        'Referer': f'http://127.0.0.1:8080/c/{chat_id}'
    }

    # For minimal compliance, the server typically just needs:
    # - model
    # - messages (the user message, maybe partial history)
    # - chat_id
    # - session_id, id (some unique IDs)
    # - background_tasks, features, etc.
    completions_body = {
        "stream": False,
        "model": model,
        "params": {},
        "messages": messages,
        "features": {"web_search": False, "code_interpreter": False, "image_generation": False, },
        "chat_id": chat_id,
        "id": completion_id,
        "background_tasks": {
            "title_generation": False,
            "tags_generation": True
        }
    }

    resp = requests.post(f"{OPEN_WEBUI_BASE_URL}/api/chat/completions", headers=headers, json=completions_body)
    resp.raise_for_status()

    return resp.json()


#
# ----------------------- Utility Functions ----------------------------
#

def create_user_message_obj(
        message: str, user_id: str, assistant_id: str, now_secs: int, model: str
) -> dict:
    """
    Builds the user message object structure that the server expects.
    """
    return {
        "id": user_id,
        "parentId": None,  # Will link them in a moment
        "childrenIds": [assistant_id],
        "role": "user",
        "content": message,
        "timestamp": now_secs,
        "models": [model]
    }


def create_assistant_stub_obj(
        assistant_id: str, user_id: str, now_secs: int, model: str
) -> dict:
    """
    Builds the assistant's empty (stub) message object.
    """
    return {
        "id": assistant_id,
        "parentId": user_id,
        "childrenIds": [],
        "role": "assistant",
        "content": "",
        "model": model,
        "modelIdx": 0,
        "userContext": None,
        "timestamp": now_secs
    }


def fill_assistant_message(chat_data: dict, assistant_id: str, final_content: str):
    """
    Updates the local chat_data with the assistant's final content
    (filling in the previously empty assistant stub).
    """
    # Update in `history["messages"]`
    if assistant_id in chat_data["history"]["messages"]:
        chat_data["history"]["messages"][assistant_id]["content"] = final_content

    # Update in the `messages` array
    for msg in chat_data["messages"]:
        if msg["id"] == assistant_id:
            msg["content"] = final_content
            break


#
# ----------------------- prompt_model ----------------------------
#

def prompt_model(
        message: str,
        chat_id: str | None,
        model: str,
        transient: bool = True
) -> (str, str):
    global CHATS

    authenticate()

    chat_data = {}
    assistant_message_stub_id = None

    if not transient:
        chat_id, chat_data, assistant_message_stub_id = prepare_chat_history(message, chat_id, model)
    #
    # 4) Call /api/chat/completions
    #
    completion_id = str(uuid.uuid4())
    completion = call_completions(
        chat_id=chat_id,
        completion_id=completion_id,
        messages=[{"role": "user", "content": message}],
        model=model
    )
    assistant_message = completion["choices"][0]["message"]["content"]

    if not transient:
        #
        # 5) Fill in the final assistant content, then update the server again
        #
        fill_assistant_message(chat_data, assistant_message_stub_id, assistant_message)

        # Sync the updated assistant message back to the server
        update_chat_on_server(chat_id, chat_data)

        #
        # 6) Store updated chat data in local CHATS
        #
        CHATS[chat_id] = chat_data

    #
    # 7) Return final assistant message
    #
    return assistant_message, chat_id


def prepare_chat_history(
        message: str,
        chat_id: str | None,
        model: str
) -> (str, Dict, str):
    #
    # 1) Get or create chat data locally
    #
    if not chat_id:
        # Create a new chat on the server, store the returned data in CHATS
        new_chat_data = create_new_chat(model=model)
        chat_id = new_chat_data["id"]
        CHATS[chat_id] = new_chat_data["chat"]  # the 'chat' field from the server’s response
    else:
        # If we have an existing chat, ensure it’s in our local CHATS store
        if chat_id not in CHATS:
            raise ValueError(f"Chat ID '{chat_id}' not found in local store. "
                             f"Load it first, or pass chat_id=None to create a new chat.")

    chat_data = CHATS[chat_id]
    #
    # 2) Build the new user message + assistant stub
    #
    now_secs = int(time.time())
    user_message_id = str(uuid.uuid4())
    assistant_message_stub_id = str(uuid.uuid4())

    user_msg_obj = create_user_message_obj(message, user_message_id, assistant_message_stub_id, now_secs, model)
    assistant_msg_obj = create_assistant_stub_obj(assistant_message_stub_id, user_message_id, now_secs, model)

    # Insert into chat_data["history"]["messages"]
    chat_data["history"]["messages"][user_message_id] = user_msg_obj
    chat_data["history"]["messages"][assistant_message_stub_id] = assistant_msg_obj
    chat_data["history"]["currentId"] = assistant_message_stub_id

    # Insert into chat_data["messages"]
    chat_data["messages"].append(user_msg_obj)
    chat_data["messages"].append(assistant_msg_obj)

    #
    # 3) Update the remote chat with our new user + assistant stub
    #
    update_chat_on_server(chat_id, chat_data)

    return chat_id, chat_data, assistant_message_stub_id
