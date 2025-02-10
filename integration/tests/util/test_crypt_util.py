import pytest

from integration.util.crypt_util import (
    generate_key_pair,
    export_private_key,
    export_public_key,
    import_private_key,
    import_public_key,
    encrypt_with_private,
    decrypt_with_public,
)


def test_key_generation():
    # Generate a 1024-bit key pair (suitable for testing)
    private_key, public_key = generate_key_pair(key_size=1024)
    # Check that required attributes exist
    assert hasattr(private_key, "n")
    assert hasattr(private_key, "d")
    assert hasattr(public_key, "e")
    # The modulus should be the same in both keys
    assert private_key.n == public_key.n


def test_export_import_keys():
    private_key, public_key = generate_key_pair(key_size=1024)
    priv_pem = export_private_key(private_key)
    pub_pem = export_public_key(public_key)

    # Check that the PEM strings contain the expected header lines
    assert "BEGIN RSA PRIVATE KEY" in priv_pem or "BEGIN PRIVATE KEY" in priv_pem
    assert "BEGIN PUBLIC KEY" in pub_pem

    # Import the keys back and verify that key parameters remain the same
    imported_private = import_private_key(priv_pem)
    imported_public = import_public_key(pub_pem)
    assert imported_private.n == private_key.n
    assert imported_public.n == public_key.n


@pytest.mark.parametrize("message", [
    "Hello, world!",
    "Short",
    "A longer message with multiple words and punctuation! @#$%^&*()",
    "1234567890",
    "测试中文",  # Test with Chinese characters
])
def test_encryption_decryption(message):
    private_key, public_key = generate_key_pair(key_size=1024)
    ciphertext = encrypt_with_private(message, private_key)
    recovered = decrypt_with_public(ciphertext, public_key)
    assert recovered == message


def test_empty_string():
    private_key, public_key = generate_key_pair(key_size=1024)
    message = ""
    ciphertext = encrypt_with_private(message, private_key)
    recovered = decrypt_with_public(ciphertext, public_key)
    assert recovered == message


def test_ciphertext_length():
    """
    Ensure that the ciphertext length (in bytes) is equal to the
    length determined by the key modulus.
    """
    private_key, public_key = generate_key_pair(key_size=1024)
    message = "Hello"
    ciphertext = encrypt_with_private(message, private_key)
    key_size_bytes = (private_key.n.bit_length() + 7) // 8
    assert len(ciphertext) == key_size_bytes


def test_incorrect_decryption():
    """
    If you encrypt with one private key and try to decrypt with a mismatched public key,
    decryption should fail or produce garbage that is not the original message.

    This test passes if either a UnicodeDecodeError is raised during decoding or if
    the decrypted text (if it decodes) does not equal the original message.
    """
    private_key1, public_key1 = generate_key_pair(key_size=1024)
    private_key2, public_key2 = generate_key_pair(key_size=1024)
    message = "Test message"
    ciphertext = encrypt_with_private(message, private_key1)

    try:
        recovered = decrypt_with_public(ciphertext, public_key2)
    except UnicodeDecodeError:
        # If decryption with the wrong key produces invalid UTF-8, that's acceptable.
        recovered = None

    # Assert that the recovered message is not the same as the original message.
    assert recovered != message


def test_message_too_long(monkeypatch):
    """
    The encrypt_with_private() function should raise a ValueError if the integer
    representation of the message is greater than or equal to the modulus.
    """
    private_key, _ = generate_key_pair(key_size=1024)
    key_size_bytes = (private_key.n.bit_length() + 7) // 8

    class FakeStr(str):
        def encode(self, encoding='utf-8'):
            # Return a bytes object that is as long as the key modulus in bytes,
            # which makes the integer representation >= private_key.n.
            return b'\xff' * key_size_bytes

    fake_message = FakeStr("ignored")
    with pytest.raises(ValueError, match="Message too long for the RSA key size"):
        encrypt_with_private(fake_message, private_key)
