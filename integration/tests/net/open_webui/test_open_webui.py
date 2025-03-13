import time
import uuid

import integration.net.open_webui.api as api
import pytest

# Import the module to test
from integration.data.config import OPEN_WEBUI_EMAIL, OPEN_WEBUI_PASSWORD

# Test constants - you may need to adjust these based on your environment
TEST_MODEL = "qwen2.5-coder:14b"
TEST_CHAT_TITLE = "Integration Test Chat"
TEST_MESSAGE = "Hello, can you help me test the OpenWebUI API integration?"
TEST_FOLLOWUP_MESSAGE = "Can you provide more information about integration testing?"


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    """Setup before all tests and teardown after all tests."""
    # Setup: Clear cached token and chats
    api.API_TOKEN = None
    api.CHATS = {}

    # Run the tests
    yield

    # Teardown: Clear data again
    api.API_TOKEN = None
    api.CHATS = {}


@pytest.fixture
def authenticated_client():
    """Fixture to ensure client is authenticated before tests."""
    token = api.authenticate()
    assert token, "Authentication failed"
    return token


class TestOpenWebUIIntegration:

    def test_authentication(self):
        """Test that authentication works and returns a token."""
        # Clear any existing token
        api.API_TOKEN = None

        # Test with default credentials
        token = api.authenticate()
        assert token, "Authentication failed with default credentials"
        assert isinstance(token, str), "Token should be a string"

        # Test with explicit credentials
        api.API_TOKEN = None
        token = api.authenticate(OPEN_WEBUI_EMAIL, OPEN_WEBUI_PASSWORD)
        assert token, "Authentication failed with explicit credentials"

        # Test token caching
        cached_token = api.authenticate()
        assert cached_token == token, "Cached token does not match original token"

    def test_create_new_chat(self, authenticated_client):
        """Test creation of a new chat."""
        chat_data = api.create_new_chat(model=TEST_MODEL, title=TEST_CHAT_TITLE)

        # Verify chat data structure
        assert "id" in chat_data, "Chat data should contain an id"
        assert isinstance(chat_data["id"], str), "Chat id should be a string"

        # Verify chat object structure
        assert "chat" in chat_data, "Response should contain a chat object"
        chat = chat_data["chat"]

        assert "title" in chat, "Chat should have a title"
        assert chat["title"] == TEST_CHAT_TITLE, f"Chat title should be '{TEST_CHAT_TITLE}'"

        assert "models" in chat, "Chat should have models"
        assert TEST_MODEL in chat["models"], f"Chat models should include '{TEST_MODEL}'"

        assert "history" in chat, "Chat should have history"
        assert "messages" in chat["history"], "Chat history should have messages"
        assert "currentId" in chat["history"], "Chat history should have currentId"

        assert "messages" in chat, "Chat should have messages array"
        assert isinstance(chat["messages"], list), "Chat messages should be a list"

    def test_update_chat_on_server(self, authenticated_client):
        """Test updating a chat on the server."""
        # First create a new chat
        new_chat = api.create_new_chat(model=TEST_MODEL, title=TEST_CHAT_TITLE)
        chat_id = new_chat["id"]
        chat_data = new_chat["chat"]

        # Make a change to the chat data
        updated_title = f"{TEST_CHAT_TITLE} - Updated"
        chat_data["title"] = updated_title

        # Update the chat on the server
        updated_chat = api.update_chat_on_server(chat_id, chat_data)

        # Verify the update was successful
        assert "chat" in updated_chat, "Response should contain a chat object"
        assert updated_chat["chat"]["title"] == updated_title, f"Chat title should be updated to '{updated_title}'"

    def test_message_object_creation(self):
        """Test the utility functions for creating message objects."""
        # Test parameters
        test_message = "Test message"
        user_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        timestamp = int(time.time())

        # Test user message creation
        user_msg = api.create_user_message_obj(
            test_message, user_id, assistant_id, timestamp, TEST_MODEL
        )

        assert user_msg["id"] == user_id
        assert user_msg["role"] == "user"
        assert user_msg["content"] == test_message
        assert user_msg["childrenIds"] == [assistant_id]
        assert user_msg["timestamp"] == timestamp
        assert TEST_MODEL in user_msg["models"]

        # Test assistant message stub creation
        assistant_msg = api.create_assistant_stub_obj(
            assistant_id, user_id, timestamp, TEST_MODEL
        )

        assert assistant_msg["id"] == assistant_id
        assert assistant_msg["parentId"] == user_id
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == ""
        assert assistant_msg["model"] == TEST_MODEL
        assert assistant_msg["timestamp"] == timestamp

    def test_fill_assistant_message(self):
        """Test filling an assistant message with content."""
        # Create a mock chat data structure
        user_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        timestamp = int(time.time())

        chat_data = {
            "history": {
                "messages": {
                    assistant_id: {
                        "id": assistant_id,
                        "content": "",
                        "role": "assistant"
                    }
                },
                "currentId": assistant_id
            },
            "messages": [
                {
                    "id": user_id,
                    "content": "Test user message",
                    "role": "user"
                },
                {
                    "id": assistant_id,
                    "content": "",
                    "role": "assistant"
                }
            ]
        }

        # Test message content
        test_content = "This is the assistant's response."

        # Fill the assistant message
        api.fill_assistant_message(chat_data, assistant_id, test_content)

        # Verify the content was updated in both places
        assert chat_data["history"]["messages"][assistant_id]["content"] == test_content

        for msg in chat_data["messages"]:
            if msg["id"] == assistant_id:
                assert msg["content"] == test_content
                break
        else:
            pytest.fail("Assistant message not found in messages array")

    def test_prompt_model_transient(self, authenticated_client):
        """Test prompting the model in transient mode (no chat history)."""
        # Create a new chat first to get a valid chat_id
        new_chat = api.create_new_chat(model=TEST_MODEL)
        chat_id = new_chat["id"]

        # Test prompt_model with transient=True
        response, returned_chat_id = api.prompt_model(
            message=TEST_MESSAGE,
            chat_id=chat_id,
            model=TEST_MODEL,
            transient=True
        )

        # Verify response
        assert isinstance(response, str), "Response should be a string"
        assert response, "Response should not be empty"
        assert returned_chat_id == chat_id, "Returned chat_id should match the provided chat_id"

        # Verify that the chat history was not updated (because transient=True)
        if chat_id in api.CHATS:
            # If the chat exists in the cache, it should not contain our test message
            for msg in api.CHATS[chat_id]["messages"]:
                if msg["role"] == "user" and msg["content"] == TEST_MESSAGE:
                    pytest.fail("Transient message should not be saved in chat history")

    def test_prompt_model_with_history(self, authenticated_client):
        """Test prompting the model with chat history."""
        # Test prompt_model with transient=False and no existing chat_id
        response1, chat_id = api.prompt_model(
            message=TEST_MESSAGE,
            chat_id=None,
            model=TEST_MODEL,
            transient=False
        )

        # Verify response
        assert isinstance(response1, str), "Response should be a string"
        assert response1, "Response should not be empty"
        assert chat_id, "A chat_id should be returned"

        # Verify that the chat history was updated
        assert chat_id in api.CHATS, "Chat should be stored in CHATS"

        # Find our message in the chat history
        user_message_found = False
        assistant_message_found = False

        for msg in api.CHATS[chat_id]["messages"]:
            if msg["role"] == "user" and msg["content"] == TEST_MESSAGE:
                user_message_found = True
            elif msg["role"] == "assistant" and msg["content"] == response1:
                assistant_message_found = True

        assert user_message_found, "User message should be in chat history"
        assert assistant_message_found, "Assistant message should be in chat history"

        # Test follow-up message to the same chat
        response2, chat_id2 = api.prompt_model(
            message=TEST_FOLLOWUP_MESSAGE,
            chat_id=chat_id,
            model=TEST_MODEL,
            transient=False
        )

        # Verify second response
        assert isinstance(response2, str), "Second response should be a string"
        assert response2, "Second response should not be empty"
        assert chat_id2 == chat_id, "Chat ID should remain the same for follow-up messages"

        # Verify that the chat history now includes both messages
        message_count = 0
        followup_found = False

        for msg in api.CHATS[chat_id]["messages"]:
            if msg["role"] == "user":
                message_count += 1
                if msg["content"] == TEST_FOLLOWUP_MESSAGE:
                    followup_found = True

        assert message_count == 2, "Chat history should contain two user messages"
        assert followup_found, "Follow-up message should be in chat history"

    def test_prepare_chat_history(self, authenticated_client):
        """Test the prepare_chat_history function."""
        # Test with a new chat (chat_id=None)
        chat_id, chat_data, assistant_id = api.prepare_chat_history(
            message=TEST_MESSAGE,
            chat_id=None,
            model=TEST_MODEL
        )

        # Verify chat was created
        assert chat_id, "A chat_id should be returned"
        assert chat_id in api.CHATS, "Chat should be stored in CHATS"

        # Verify message structure
        assert len(chat_data["messages"]) == 2, "Chat should have 2 messages (user + assistant stub)"
        assert chat_data["messages"][0]["role"] == "user"
        assert chat_data["messages"][0]["content"] == TEST_MESSAGE
        assert chat_data["messages"][1]["role"] == "assistant"
        assert chat_data["messages"][1]["content"] == ""
        assert chat_data["messages"][1]["id"] == assistant_id

        # Test with an existing chat
        followup_chat_id, followup_chat_data, followup_assistant_id = api.prepare_chat_history(
            message=TEST_FOLLOWUP_MESSAGE,
            chat_id=chat_id,
            model=TEST_MODEL
        )

        # Verify chat continuity
        assert followup_chat_id == chat_id, "Chat ID should remain the same"
        assert len(followup_chat_data["messages"]) == 4, "Chat should now have 4 messages"

        # Verify the history currentId points to the new assistant message
        assert followup_chat_data["history"]["currentId"] == followup_assistant_id

        # Test with a non-existent chat_id
        with pytest.raises(ValueError, match="Chat ID .* not found in local store"):
            api.prepare_chat_history(
                message=TEST_MESSAGE,
                chat_id="non-existent-id",
                model=TEST_MODEL
            )

    def test_call_completions(self, authenticated_client):
        """Test the call_completions function."""
        # Create a new chat to get a valid chat_id
        new_chat = api.create_new_chat(model=TEST_MODEL)
        chat_id = new_chat["id"]

        # Generate test data
        completion_id = str(uuid.uuid4())
        messages = [{"role": "user", "content": TEST_MESSAGE}]

        # Call completions
        completion = api.call_completions(
            chat_id=chat_id,
            completion_id=completion_id,
            messages=messages,
            model=TEST_MODEL
        )

        # Verify response structure
        assert "choices" in completion, "Completion should have choices"
        assert len(completion["choices"]) > 0, "Completion should have at least one choice"
        assert "message" in completion["choices"][0], "Choice should have a message"
        assert "content" in completion["choices"][0]["message"], "Message should have content"

        # Verify content
        content = completion["choices"][0]["message"]["content"]
        assert isinstance(content, str), "Content should be a string"
        assert content, "Content should not be empty"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
