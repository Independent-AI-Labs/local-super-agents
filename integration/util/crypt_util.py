from Crypto.PublicKey import RSA


def generate_key_pair(key_size=2048):
    """
    Generates an RSA key pair.

    Returns:
        (private_key, public_key): A tuple of RSA key objects.
    """
    key = RSA.generate(key_size)
    return key, key.publickey()


def export_private_key(private_key):
    """
    Exports the RSA private key in PEM format (as a UTF-8 string).
    """
    return private_key.export_key().decode('utf-8')


def export_public_key(public_key):
    """
    Exports the RSA public key in PEM format (as a UTF-8 string).
    """
    return public_key.export_key().decode('utf-8')


def import_private_key(pem_str):
    """
    Imports an RSA private key from a PEM-formatted string.
    """
    return RSA.import_key(pem_str)


def import_public_key(pem_str):
    """
    Imports an RSA public key from a PEM-formatted string.
    """
    return RSA.import_key(pem_str)


def encrypt_with_private(message, private_key):
    """
    "Encrypts" a message using the RSA private key by performing raw RSA
    (i.e. m^d mod n) on the message.

    Args:
        message (str): The plaintext message to be encrypted.
        private_key: An RSA private key object (from PyCryptodome).

    Returns:
        bytes: The ciphertext as a byte string.

    Raises:
        ValueError: If the message (as an integer) is too long for the key.

    WARNING: This operation is not secure (no padding is used) and is intended
    only for demonstration purposes. In real applications you should use proper
    signing or encryption schemes.
    """
    # Convert message to bytes and then to an integer
    message_bytes = message.encode('utf-8')
    m_int = int.from_bytes(message_bytes, byteorder='big')

    # Ensure the message integer is less than the modulus
    if m_int >= private_key.n:
        raise ValueError("Message too long for the RSA key size")

    # Perform raw RSA operation with the private exponent: c = m^d mod n
    c_int = pow(m_int, private_key.d, private_key.n)

    # Convert the ciphertext integer back to bytes.
    # The byte length is determined by the key size.
    key_size_bytes = (private_key.n.bit_length() + 7) // 8
    cipher_bytes = c_int.to_bytes(key_size_bytes, byteorder='big')
    return cipher_bytes


def decrypt_with_public(cipher_bytes, public_key):
    """
    Decrypts the ciphertext using the RSA public key by performing raw RSA
    (i.e. c^e mod n) to recover the original message.

    Args:
        cipher_bytes (bytes): The ciphertext produced by encrypt_with_private().
        public_key: An RSA public key object (from PyCryptodome).

    Returns:
        str: The decrypted plaintext message.

    WARNING: This operation is not secure (no padding is used) and is intended
    only for demonstration purposes.
    """
    # Convert the ciphertext bytes to an integer
    c_int = int.from_bytes(cipher_bytes, byteorder='big')

    # Perform the RSA operation with the public exponent: m = c^e mod n
    m_int = pow(c_int, public_key.e, public_key.n)

    # Determine the byte length of the modulus and convert the integer back
    key_size_bytes = (public_key.n.bit_length() + 7) // 8
    message_bytes = m_int.to_bytes(key_size_bytes, byteorder='big')

    # Remove any leading zero bytes that may have been added during conversion
    message_bytes = message_bytes.lstrip(b'\x00')
    return message_bytes.decode('utf-8')
