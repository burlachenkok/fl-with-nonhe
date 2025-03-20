#!/usr/bin/env python3

#    python -m pip install pycryptodome
#    https://pycryptodome.readthedocs.io/en/latest/src/cipher/aes.html
#
#    16 bytes key the most secure

from Crypto.Cipher import AES
import hashlib

def get256bitDigest(shared_key):
    return hashlib.sha256(shared_key.encode()).digest()

def encode(plainMessage, keyDigest):
    cipher = AES.new(keyDigest, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(plainMessage)
    print(len(ciphertext))
    print(ciphertext)
    return (nonce, tag, ciphertext)

def decode(encryptedMessage, keyDigest, verify = True):
    nonce, tag, ciphertext = encryptedMessage

    cipher_decryptor = AES.new(keyDigest, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher_decryptor.decrypt(ciphertext)
    
    if verify:
        try:
            cipher_decryptor.verify(tag)
        except ValueError:
            print("Key incorrect or message corrupted [BAD]")

    return plaintext

shared_key = "Sixteen byte key 123"
shared_key_digest = get256bitDigest(shared_key)
print(decode(encode("123456789".encode(), shared_key_digest), shared_key_digest))


# EAX is a two-pass scheme, which means that encryption and authentication are done in separate operations. 
# https://www.cs.ucdavis.edu/~rogaway/papers/eax.pdf
