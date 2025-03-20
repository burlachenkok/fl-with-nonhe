#!/usr/bin/env python3

#    python -m pip install pycryptodome

import hashlib
from Crypto import Random
from Crypto.Cipher import AES
from base64 import b64encode, b64decode

class AESCipher(object):
    # @param key any length string. Then we proceed to generate a 256 bit hash from that key. 
    def __init__(self, key):
        self.block_size = AES.block_size
        # A hash is basically a unique identifier of a given length, 32 characters in this case, of any input no matter its length. 
        # A key is a unique 256 bit key for the cipher. 
        self.key = hashlib.sha256(key.encode()).digest()

    def __pad(self, plain_text):
        number_of_bytes_to_pad = self.block_size - len(plain_text) % self.block_size
        ascii_string = chr(number_of_bytes_to_pad)
        padding_str = number_of_bytes_to_pad * ascii_string
        padded_plain_text = plain_text + padding_str
        return padded_plain_text

    @staticmethod
    def __unpad(plain_text):
        last_character = plain_text[len(plain_text) - 1:]
        bytes_to_remove = ord(last_character)
        return plain_text[:-bytes_to_remove]

    def encrypt(self, plain_text):
        plain_text = self.__pad(plain_text)                  # First we pad that plain_text in order to be able to encrypt it
        iv = Random.new().read(self.block_size)              # After we generate a new random iv with the size of an AES block, 128bits
        cipher = AES.new(self.key, AES.MODE_CBC, iv)         # We now create our AES cipher

        # str.encode(encoding='utf-8', errors='strict')
        encrypted_text = cipher.encrypt(plain_text.encode())        
        return b64encode(iv + encrypted_text).decode("utf-8")  # iv || encrypted

    def decrypt(self, encrypted_text):
        encrypted_text = b64decode(encrypted_text)
        iv = encrypted_text[:self.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        plain_text = cipher.decrypt(encrypted_text[self.block_size:]).decode("utf-8")
        return self.__unpad(plain_text)


e = AESCipher("hello")

input_message = "test-a"
out_e = e.encrypt(input_message)
out_d = e.decrypt(out_e)
print(out_d)



