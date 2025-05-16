import hashlib
import random
import time
from collections import Counter
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
import math
import matplotlib.pyplot as plt
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.Cipher import ChaCha20

codons = np.array([a + b + c for a in 'ACGU' for b in 'ACGU' for c in 'ACGU'])

base64_chars = np.array(list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'))

def can_pair(base1, base2):
    pairs = {
        ('A', 'U'), ('U', 'A'),
        ('G', 'C'), ('C', 'G'),
        ('G', 'U'), ('U', 'G')
    }
    return (base1, base2) in pairs

def codon_generator():
    for a in 'ACGU':
        for b in 'ACGU':
            for c in 'ACGU':
                yield a + b + c

def generate_codon_substitution_matrix(seed):
    rng = random.Random(seed)
    codons_list = list(codon_generator())
    shuffled = codons_list.copy()
    rng.shuffle(shuffled)
    return {k: v for k, v in zip(codons_list, shuffled)}

def process_chunk(chunk_and_matrix):
    chunk, substitution_matrix = chunk_and_matrix
    return [substitution_matrix[codon] for codon in chunk]

def substitute_codons(codon_sequence, substitution_matrix):
    return [substitution_matrix[codon] for codon in codon_sequence]

def linear_fold(sequence):
    n = len(sequence)
    dp = [0] * n
    stack = []
    structure = ['.' for _ in range(n)]
    
    for i in range(n):
        while stack and can_pair(sequence[stack[-1]], sequence[i]):
            j = stack.pop()
            if i - j > 3:
                structure[j] = '('
                structure[i] = ')'
                dp[i] = dp[j] + 1
        
        if sequence[i] in 'ACGU':
            stack.append(i)
            
        if len(stack) > 30:
            stack = stack[-30:]
    
    return ''.join(structure)

def inverse_rna_secondary_structure(codon_sequence, indices_order):
    sequence = ''.join(codon_sequence)
    sequence_array = np.array(list(sequence))
    
    indices_order_array = np.array(indices_order)
    
    if len(indices_order_array) != len(sequence_array):
        min_len = min(len(indices_order_array), len(sequence_array))
        sequence_array = sequence_array[:min_len]
        indices_order_array = indices_order_array[:min_len]
        print(f"Warning: Sequence length adjusted to {min_len}")
    
    inverse_order = np.argsort(indices_order_array)
    
    original_sequence_array = sequence_array[inverse_order]
    original_sequence = ''.join(original_sequence_array)
    
    if len(original_sequence) % 3 != 0:
        padding_length = 3 - (len(original_sequence) % 3)
        original_sequence = original_sequence + 'N' * padding_length
        print(f"Warning: Added {padding_length} padding characters to ensure sequence length is a multiple of 3")
    
    original_codon_sequence = [original_sequence[i:i+3] for i in range(0, len(original_sequence), 3)]
    
    return original_codon_sequence

def apply_rna_secondary_structure(codon_sequence):
    base_sequence = ''.join(codon_sequence)
    structure = linear_fold(base_sequence)
    
    paired_indices = [i for i, c in enumerate(structure) if c in '()']
    unpaired_indices = [i for i, c in enumerate(structure) if c == '.']
    indices_order = paired_indices + unpaired_indices
    
    new_sequence = ''.join(base_sequence[i] for i in indices_order)
    new_codon_sequence = [new_sequence[i:i+3] for i in range(0, len(new_sequence), 3)]
    
    return new_codon_sequence, indices_order

def generate_dynamic_key_from_biological_data(seed_sequence, salt, iterations=100000):
    valid_bases = set('ACGU')
    if not set(seed_sequence.upper()).issubset(valid_bases):
        raise ValueError("Seed sequence contains invalid bases.")
    dynamic_key = PBKDF2(seed_sequence, salt, dkLen=32, count=iterations, hmac_hash_module=SHA256)
    return dynamic_key

def aes_encrypt(data_sequence, key):
    data_str = ''.join(data_sequence)
    data_bytes = data_str.encode()
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(pad(data_bytes, AES.block_size))
    return cipher.nonce + tag + ciphertext

def add_checksum(encrypted_data):
    checksum = hashlib.sha256(encrypted_data).digest()
    return encrypted_data + checksum

def cha_encrypt(data, key):
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = data
        
    cipher = ChaCha20.new(key=key)
    ciphertext = cipher.nonce + cipher.encrypt(data_bytes)
    return ciphertext

def inverse_substitute_codons(codon_sequence, substitution_matrix):
    inverse_matrix = {v: k for k, v in substitution_matrix.items()}
    original_codon_sequence = []
    
    for i, codon in enumerate(codon_sequence):
        if codon not in inverse_matrix:
            raise ValueError(f"Cannot find inverse substitution for codon '{codon}'")
        original_codon_sequence.append(inverse_matrix[codon])
    
    return original_codon_sequence

def decode_codons_to_plaintext(codon_sequence):
    try:
        codon_str = ''.join(codon_sequence)
        
        if len(codon_str) % 3 != 0:
            padding_length = 3 - (len(codon_str) % 3)
            codon_str = codon_str + 'N' * padding_length
        
        codons_list = [codon_str[i:i+3] for i in range(0, len(codon_str), 3)]
        
        codon_indices = []
        for codon in codons_list:
            try:
                idx = np.where(codons == codon)[0][0]
                codon_indices.append(idx)
            except IndexError:
                print(f"Warning: Skipping invalid codon '{codon}'")
                continue
        
        base64_str = ''.join(base64_chars[idx % 64] for idx in codon_indices)
        
        padding_length = -len(base64_str) % 4
        base64_padded = base64_str + '=' * padding_length
        
        try:
            plaintext_bytes = base64.b64decode(base64_padded)
            return plaintext_bytes.decode('utf-8')
        except Exception as e:
            print(f"Base64 decoding failed, trying other methods: {str(e)}")
            return base64_str
            
    except Exception as e:
        print(f"Decoding process error: {str(e)}")
        raise

def decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order):
    encrypted_data = verify_and_remove_checksum(encrypted_data_with_checksum)
    dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence, salt)
    decrypted_sequence = cha_decrypt(encrypted_data, dynamic_key)
    unfolded_sequence = inverse_rna_secondary_structure(decrypted_sequence, indices_order)
    original_codon_sequence = inverse_substitute_codons(unfolded_sequence, substitution_matrix)
    plaintext = decode_codons_to_plaintext(original_codon_sequence)
    return plaintext

def verify_and_remove_checksum(encrypted_data_with_checksum):
    encrypted_data = encrypted_data_with_checksum[:-32]
    checksum = encrypted_data_with_checksum[-32:]
    computed_checksum = hashlib.sha256(encrypted_data).digest()
    if checksum != computed_checksum:
        raise ValueError("Checksum does not match. Data may be corrupted.")
    return encrypted_data

def cha_decrypt(encrypted_data, key):
    try:
        nonce = encrypted_data[:8]
        ciphertext = encrypted_data[8:]
        cipher = ChaCha20.new(key=key, nonce=nonce)
        decrypted_data = cipher.decrypt(ciphertext)
        
        for encoding in ['utf-8', 'latin1', 'ascii']:
            try:
                data_str = decrypted_data.decode(encoding)
                if len(data_str) % 3 == 0 and all(c in 'ACGU' for c in data_str):
                    codon_sequence = [data_str[i:i+3] for i in range(0, len(data_str), 3)]
                    return codon_sequence
            except UnicodeDecodeError:
                continue
        
        data_str = ''.join(chr(b) for b in decrypted_data)
        codon_sequence = [data_str[i:i+3] for i in range(0, len(data_str), 3)]
        return codon_sequence
        
    except Exception as e:
        print(f"Decryption process error: {str(e)}")
        raise

def calculate_entropy(data):
    byte_data = data
    if isinstance(data, str):
        byte_data = data.encode()
    data_length = len(byte_data)
    if data_length == 0:
        return 0.0
    frequencies = Counter(byte_data)
    freqs = np.array(list(frequencies.values()), dtype=float)
    probabilities = freqs / data_length
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def plot_entropy_histogram(data):
    byte_data = data
    if isinstance(data, str):
        byte_data = data.encode()
    frequencies = Counter(byte_data)
    bytes_list = np.array(list(frequencies.keys()))
    counts = np.array(list(frequencies.values()))
    plt.bar(bytes_list, counts)
    plt.xlabel('Byte Value')
    plt.ylabel('Frequency')
    plt.title('Byte Frequency Distribution of Encrypted Data')
    plt.show()

def test_performance():
    plaintext_lengths = [50, 100, 200, 400, 800, 1600]
    encryption_times = []
    decryption_times = []
    seed = "123456789"
    seed_sequence = "ACGUACGUACGUACGUACGUACGUACGUACGU"
    salt = b'salt_123'

    def process_length(length):
        plaintext = 'A' * length
        start_time = time.time()
        encrypted_data_with_checksum, substitution_matrix, indices_order = encrypt(plaintext, seed, seed_sequence, salt)
        encryption_time = time.time() - start_time
        start_time = time.time()
        decrypted_plaintext = decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order)
        decryption_time = time.time() - start_time
        return (encryption_time, decryption_time)

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_length, plaintext_lengths)
        for et, dt in results:
            encryption_times.append(et)
            decryption_times.append(dt)

    plt.plot(plaintext_lengths, encryption_times, label='Encryption Time')
    plt.plot(plaintext_lengths, decryption_times, label='Decryption Time')
    plt.xlabel('Plaintext Length (characters)')
    plt.ylabel('Time (seconds)')
    plt.title('Encryption and Decryption Time vs. Plaintext Length')
    plt.legend()
    plt.show()

def test_comparison():
    """Compare performance of ncRNA, AES, and RSA"""
    plaintext_lengths = [50, 100, 200, 400, 800, 1600]
    ncrna_times = []
    aes_times = []
    rsa_times = []
    
    seed = "123456789"
    seed_sequence = "ACGUACGUACGUACGUACGUACGUACGUACGU"
    salt = b'salt_123'
    aes_key = get_random_bytes(32)
    rsa_key = RSA.generate(2048)
    rsa_cipher = PKCS1_OAEP.new(rsa_key)
    
    def test_ncrna(plaintext):
        start_time = time.time()
        encrypted_data, _, _ = encrypt(plaintext, seed, seed_sequence, salt)
        return time.time() - start_time
    
    def test_aes(plaintext):
        start_time = time.time()
        cipher = AES.new(aes_key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(pad(plaintext.encode(), AES.block_size))
        return time.time() - start_time
    
    def test_rsa(plaintext):
        start_time = time.time()
        block_size = 190
        blocks = [plaintext[i:i+block_size].encode() for i in range(0, len(plaintext), block_size)]
        for block in blocks:
            rsa_cipher.encrypt(block)
        return time.time() - start_time

    for length in plaintext_lengths:
        test_text = 'A' * length
        ncrna_times.append(test_ncrna(test_text))
        aes_times.append(test_aes(test_text))
        rsa_times.append(test_rsa(test_text))

    plt.figure(figsize=(10, 6))
    plt.plot(plaintext_lengths, ncrna_times, 'o-', label='ncRNA')
    plt.plot(plaintext_lengths, aes_times, 's-', label='AES')
    plt.plot(plaintext_lengths, rsa_times, '^-', label='RSA')
    plt.xlabel('Plaintext Length (characters)')
    plt.ylabel('Encryption Time (seconds)')
    plt.title('Encryption Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nPerformance Comparison Results:")
    print("Plaintext Length\tncRNA(s)\tAES(s)\t\tRSA(s)")
    print("-" * 50)
    for i, length in enumerate(plaintext_lengths):
        print(f"{length}\t\t{ncrna_times[i]:.6f}\t{aes_times[i]:.6f}\t{rsa_times[i]:.6f}")

def prepare_data_chunk(chunk):
    chunk_str = ''.join(chunk)
    return pad(chunk_str.encode(), AES.block_size)

def encode_plaintext_to_codons(plaintext):
    plaintext_bytes = plaintext.encode('utf-8')
    base64_bytes = base64.b64encode(plaintext_bytes)
    base64_str = base64_bytes.decode('ascii').rstrip('=')
    
    char_to_index = {char: idx for idx, char in enumerate(base64_chars)}
    
    codon_sequence = []
    for char in base64_str:
        try:
            idx = char_to_index[char]
            codon = codons[idx]
            codon_sequence.append(codon)
        except (KeyError, IndexError):
            continue
    
    return codon_sequence

def encrypt(plaintext, seed, seed_sequence, salt):
    try:
        codon_sequence = encode_plaintext_to_codons(plaintext)
        
        substitution_matrix = generate_codon_substitution_matrix(seed)
        
        substituted_sequence = substitute_codons(codon_sequence, substitution_matrix)
        
        structured_sequence, indices_order = apply_rna_secondary_structure(substituted_sequence)
        
        dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence, salt)
        
        data_to_encrypt = ''.join(structured_sequence)
        
        encrypted_data = cha_encrypt(data_to_encrypt, dynamic_key)
        
        encrypted_data_with_checksum = add_checksum(encrypted_data)
        
        return encrypted_data_with_checksum, substitution_matrix, indices_order
        
    except Exception as e:
        print(f"Encryption process error: {str(e)}")
        raise

if __name__ == "__main__":
    DEBUG = False
    
    num_threads = multiprocessing.cpu_count()
    plaintext = "Hello, World! This is a test of the encryption algorithm based on ncRNA."
    seed = "123456789"
    seed_sequence = "ACGUACGUACGUACGUACGUACGUACGUACGU"
    salt = b'salt_123'
    
    if DEBUG:
        start_time = time.time()
        encrypted_data_with_checksum, substitution_matrix, indices_order = encrypt(
            plaintext, seed, seed_sequence, salt
        )
        encryption_time = time.time() - start_time
        print(f"Encryption completed in {encryption_time:.6f} seconds")
        encrypted_data = encrypted_data_with_checksum[:-32]
        entropy = calculate_entropy(encrypted_data)
        print(f"Entropy of encrypted data: {entropy:.4f} bits/byte")
        plot_entropy_histogram(encrypted_data)

    print("\n=== Starting Performance Tests ===")
    print("1. ncRNA Encryption/Decryption Performance Test")
    test_performance()
    
    print("\n2. ncRNA, AES, and RSA Performance Comparison Test")
    test_comparison()

def chunked_encryption(plaintext, chunk_size=400):
    chunks = [plaintext[i:i+chunk_size] for i in range(0, len(plaintext), chunk_size)]
    encrypted_chunks = []
    for chunk in chunks:
        encrypted_chunk = encrypt_chunk(chunk)
        encrypted_chunks.append(encrypted_chunk)
    return combine_chunks(encrypted_chunks)

def optimized_nussinov(sequence):
    pairs = {}
    for i in range(len(sequence)):
        for j in range(i + 4, len(sequence)):
            if can_pair(sequence[i], sequence[j]):
                pairs[(i,j)] = True

def process_large_file(plaintext, chunk_size=1024):
    """Processes large files in chunks to reduce memory usage."""
    chunks = (plaintext[i:i+chunk_size] for i in range(0, len(plaintext), chunk_size))
    results = []
    
    for chunk in chunks:
        encrypted_chunk = encrypt(chunk, seed, seed_sequence, salt)
        results.append(encrypted_chunk)
        
    return combine_results(results)
