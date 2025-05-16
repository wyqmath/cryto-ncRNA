import hashlib  # Import hashlib module for generating hash values
import random   # Import random module for generating random numbers
import time     # Import time module for calculating time
from collections import Counter  # Import Counter for counting objects
from Crypto.Cipher import AES  # Import AES encryption module
from Crypto.Util.Padding import pad, unpad  # Import padding and unpadding functions
import base64   # Import base64 module for encoding
import math     # Import math module for mathematical operations
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256  # Import correct hash module
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes  # Add this line to import statement in file top
from Crypto.Cipher import ChaCha20  # Import ChaCha20 encryption module

# Define 64 codons
codons = np.array([a + b + c for a in 'ACGU' for b in 'ACGU' for c in 'ACGU'])

# Define Base64 character set (excluding '=')
base64_chars = np.array(list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'))

# Add can_pair function before nussinov_algorithm function
def can_pair(base1, base2):
    """Checks if two RNA bases can pair.

    Allowed pairs: AU, UA, GC, CG, GU, UG
    """
    pairs = {
        ('A', 'U'), ('U', 'A'),
        ('G', 'C'), ('C', 'G'),
        ('G', 'U'), ('U', 'G')
    }
    return (base1, base2) in pairs

# 1. Optimized codon generation using a generator instead of a precomputed array
def codon_generator():
    for a in 'ACGU':
        for b in 'ACGU':
            for c in 'ACGU':
                yield a + b + c

# 2. Optimized substitution matrix generation
def generate_codon_substitution_matrix(seed):
    rng = random.Random(seed)
    codons_list = list(codon_generator())
    shuffled = codons_list.copy()
    rng.shuffle(shuffled)
    # Uses a more memory-efficient dictionary comprehension
    return {k: v for k, v in zip(codons_list, shuffled)}

# Move process_chunk function to outside
def process_chunk(chunk_and_matrix):
    """Helper function to process codon chunks.

    Args:
        chunk_and_matrix: Tuple (chunk, substitution_matrix)
    Returns:
        list: List of substituted codons.
    """
    chunk, substitution_matrix = chunk_and_matrix
    return [substitution_matrix[codon] for codon in chunk]

# Modified substitute_codons function
def substitute_codons(codon_sequence, substitution_matrix):
    """Substitutes codons in a sequence using a substitution matrix.

    Args:
        codon_sequence: List of codons.
        substitution_matrix: Substitution matrix dictionary.
    Returns:
        list: List of substituted codons.
    """
    return [substitution_matrix[codon] for codon in codon_sequence]

# 3. Optimized memory usage for the linear_fold function
def linear_fold(sequence):
    """Performs RNA secondary structure prediction using the LinearFold algorithm.

    Args:
        sequence: RNA sequence string.
    Returns:
        str: Dot-bracket notation of the structure.
    """
    n = len(sequence)
    # Uses lists instead of numpy arrays
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

# Define inverse_rna_secondary_structure function
def inverse_rna_secondary_structure(codon_sequence, indices_order):
    """Reverses RNA secondary structure reordering.

    Args:
        codon_sequence: List of codons.
        indices_order: Original reordering indices.
    Returns:
        list: Restored codon sequence.
    """
    # Merge into a single sequence
    sequence = ''.join(codon_sequence)
    sequence_array = np.array(list(sequence))
    
    indices_order_array = np.array(indices_order)
    
    # Validate length match
    if len(indices_order_array) != len(sequence_array):
        # If lengths do not match, try to adjust sequence length
        min_len = min(len(indices_order_array), len(sequence_array))
        sequence_array = sequence_array[:min_len]
        indices_order_array = indices_order_array[:min_len]
        print(f"Warning: Sequence length adjusted to {min_len}")
    
    # Generate inverse indices
    inverse_order = np.argsort(indices_order_array)
    
    # Apply inverse reordering
    original_sequence_array = sequence_array[inverse_order]
    original_sequence = ''.join(original_sequence_array)
    
    # Ensure the result length is a multiple of 3
    if len(original_sequence) % 3 != 0:
        padding_length = 3 - (len(original_sequence) % 3)
        original_sequence = original_sequence + 'N' * padding_length
        print(f"Warning: Added {padding_length} padding characters to ensure sequence length is a multiple of 3")
    
    # Convert back to codon sequence
    original_codon_sequence = [original_sequence[i:i+3] for i in range(0, len(original_sequence), 3)]
    
    return original_codon_sequence

# 4. Optimized apply_rna_secondary_structure function
def apply_rna_secondary_structure(codon_sequence):
    """Applies RNA secondary structure reordering.

    Args:
        codon_sequence: List of codons.
    Returns:
        tuple: (Reordered codon sequence, index order).
    """
    base_sequence = ''.join(codon_sequence)
    structure = linear_fold(base_sequence)
    
    # Uses list comprehensions instead of numpy operations
    paired_indices = [i for i, c in enumerate(structure) if c in '()']
    unpaired_indices = [i for i, c in enumerate(structure) if c == '.']
    indices_order = paired_indices + unpaired_indices
    
    # Uses list operations instead of numpy reordering
    new_sequence = ''.join(base_sequence[i] for i in indices_order)
    new_codon_sequence = [new_sequence[i:i+3] for i in range(0, len(new_sequence), 3)]
    
    return new_codon_sequence, indices_order

# 5. Generate dynamic key from biological data
def generate_dynamic_key_from_biological_data(seed_sequence, salt, iterations=100000):
    valid_bases = set('ACGU')
    if not set(seed_sequence.upper()).issubset(valid_bases):
        raise ValueError("Seed sequence contains invalid bases.")
    dynamic_key = PBKDF2(seed_sequence, salt, dkLen=32, count=iterations, hmac_hash_module=SHA256)
    return dynamic_key

# 6. AES encryption
def aes_encrypt(data_sequence, key):
    data_str = ''.join(data_sequence)
    data_bytes = data_str.encode()
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(pad(data_bytes, AES.block_size))
    return cipher.nonce + tag + ciphertext

# 7. Add checksum
def add_checksum(encrypted_data):
    checksum = hashlib.sha256(encrypted_data).digest()
    return encrypted_data + checksum

# Stream cipher encryption
def cha_encrypt(data, key):
    """Encrypts data using ChaCha20.

    Args:
        data: Data to encrypt (bytes or str).
        key: Encryption key.
    Returns:
        bytes: nonce + encrypted data.
    """
    # Ensure input data is bytes type
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = data
        
    cipher = ChaCha20.new(key=key)
    ciphertext = cipher.nonce + cipher.encrypt(data_bytes)
    return ciphertext

# Move inverse_substitute_codons function before decrypt function
def inverse_substitute_codons(codon_sequence, substitution_matrix):
    """Reverses codon substitution, restoring substituted codons to original."""
    inverse_matrix = {v: k for k, v in substitution_matrix.items()}
    original_codon_sequence = []
    
    for i, codon in enumerate(codon_sequence):
        if codon not in inverse_matrix:
            raise ValueError(f"Cannot find inverse substitution for codon '{codon}'")
        original_codon_sequence.append(inverse_matrix[codon])
    
    return original_codon_sequence

# Modified decode_codons_to_plaintext function
def decode_codons_to_plaintext(codon_sequence):
    """Decodes a codon sequence back to a plaintext string."""
    try:
        # Combine codon sequence into a string
        codon_str = ''.join(codon_sequence)
        
        # Ensure codon string length is a multiple of 3
        if len(codon_str) % 3 != 0:
            padding_length = 3 - (len(codon_str) % 3)
            codon_str = codon_str + 'N' * padding_length
        
        # Split into a list of codons
        codons_list = [codon_str[i:i+3] for i in range(0, len(codon_str), 3)]
        
        # Convert codons to indices
        codon_indices = []
        for codon in codons_list:
            try:
                idx = np.where(codons == codon)[0][0]
                codon_indices.append(idx)
            except IndexError:
                print(f"Warning: Skipping invalid codon '{codon}'")
                continue
        
        # Convert to Base64 characters
        base64_str = ''.join(base64_chars[idx % 64] for idx in codon_indices)
        
        # Add Base64 padding
        padding_length = -len(base64_str) % 4
        base64_padded = base64_str + '=' * padding_length
        
        # Decode Base64
        try:
            plaintext_bytes = base64.b64decode(base64_padded)
            return plaintext_bytes.decode('utf-8')
        except Exception as e:
            print(f"Base64 decoding failed, trying other methods: {str(e)}")
            # Try direct decoding
            return base64_str
            
    except Exception as e:
        print(f"Decoding process error: {str(e)}")
        raise

# Then is decrypt function
def decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order):
    encrypted_data = verify_and_remove_checksum(encrypted_data_with_checksum)
    dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence, salt)
    decrypted_sequence = cha_decrypt(encrypted_data, dynamic_key)
    unfolded_sequence = inverse_rna_secondary_structure(decrypted_sequence, indices_order)
    original_codon_sequence = inverse_substitute_codons(unfolded_sequence, substitution_matrix)
    plaintext = decode_codons_to_plaintext(original_codon_sequence)
    return plaintext

# Checksum verification
def verify_and_remove_checksum(encrypted_data_with_checksum):
    encrypted_data = encrypted_data_with_checksum[:-32]
    checksum = encrypted_data_with_checksum[-32:]
    computed_checksum = hashlib.sha256(encrypted_data).digest()
    if checksum != computed_checksum:
        raise ValueError("Checksum does not match. Data may be corrupted.")
    return encrypted_data

# AES decryption replaced with stream cipher decryption
def cha_decrypt(encrypted_data, key):
    """Decrypts data using ChaCha20.

    Args:
        encrypted_data: Encrypted data (including nonce).
        key: Decryption key.
    Returns:
        list: Decrypted codon sequence.
    """
    try:
        nonce = encrypted_data[:8]  # ChaCha20 nonce is 8 bytes
        ciphertext = encrypted_data[8:]
        cipher = ChaCha20.new(key=key, nonce=nonce)
        decrypted_data = cipher.decrypt(ciphertext)
        
        # Try decoding with multiple encodings
        for encoding in ['utf-8', 'latin1', 'ascii']:
            try:
                data_str = decrypted_data.decode(encoding)
                # Validate if decoded data matches codon format
                if len(data_str) % 3 == 0 and all(c in 'ACGU' for c in data_str):
                    codon_sequence = [data_str[i:i+3] for i in range(0, len(data_str), 3)]
                    return codon_sequence
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, process as binary
        data_str = ''.join(chr(b) for b in decrypted_data)
        codon_sequence = [data_str[i:i+3] for i in range(0, len(data_str), 3)]
        return codon_sequence
        
    except Exception as e:
        print(f"Decryption process error: {str(e)}")
        raise

# Calculate entropy
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

# Plot entropy histogram
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

# Performance testing function
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
    
    # Initialize keys
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
        # RSA can only encrypt data of limited length at a time, needs to be processed in chunks
        block_size = 190  # Max encryption block size for RSA-2048
        blocks = [plaintext[i:i+block_size].encode() for i in range(0, len(plaintext), block_size)]
        for block in blocks:
            rsa_cipher.encrypt(block)
        return time.time() - start_time

    for length in plaintext_lengths:
        test_text = 'A' * length
        ncrna_times.append(test_ncrna(test_text))
        aes_times.append(test_aes(test_text))
        rsa_times.append(test_rsa(test_text))

    # Plot performance comparison graph
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

    # Print detailed results
    print("\nPerformance Comparison Results:")
    print("Plaintext Length\tncRNA(s)\tAES(s)\t\tRSA(s)")
    print("-" * 50)
    for i, length in enumerate(plaintext_lengths):
        print(f"{length}\t\t{ncrna_times[i]:.6f}\t{aes_times[i]:.6f}\t{rsa_times[i]:.6f}")

# New function: Prepare and pad data chunks
def prepare_data_chunk(chunk):
    """Prepares a data chunk for encryption.

    Args:
        chunk: List of codons.
    Returns:
        bytes: Prepared data chunk.
    """
    # Join codon list into a string
    chunk_str = ''.join(chunk)
    # Encode string to bytes and pad
    return pad(chunk_str.encode(), AES.block_size)

def encode_plaintext_to_codons(plaintext):
    """Encodes plaintext into a codon sequence.

    Args:
        plaintext: Plaintext string to encode.
    Returns:
        list: Codon sequence.
    """
    # Convert plaintext to base64
    plaintext_bytes = plaintext.encode('utf-8')
    base64_bytes = base64.b64encode(plaintext_bytes)
    base64_str = base64_bytes.decode('ascii').rstrip('=')
    
    # Create a mapping from base64 characters to indices
    char_to_index = {char: idx for idx, char in enumerate(base64_chars)}
    
    # Convert base64 characters to codons
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
        # 1. Encode plaintext to codons
        codon_sequence = encode_plaintext_to_codons(plaintext)
        
        # 2. Generate substitution matrix
        substitution_matrix = generate_codon_substitution_matrix(seed)
        
        # 3. Substitute codons
        substituted_sequence = substitute_codons(codon_sequence, substitution_matrix)
        
        # 4. Apply RNA secondary structure
        structured_sequence, indices_order = apply_rna_secondary_structure(substituted_sequence)
        
        # 5. Generate dynamic key
        dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence, salt)
        
        # 6. Prepare data for encryption
        data_to_encrypt = ''.join(structured_sequence)
        
        # 7. Encrypt using ChaCha20
        encrypted_data = cha_encrypt(data_to_encrypt, dynamic_key)
        
        # 8. Add checksum
        encrypted_data_with_checksum = add_checksum(encrypted_data)
        
        return encrypted_data_with_checksum, substitution_matrix, indices_order
        
    except Exception as e:
        print(f"Encryption process error: {str(e)}")
        raise

# Test code
if __name__ == "__main__":
    DEBUG = False  # Set to True to output debug information
    
    # Basic encryption test
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

    # Run performance tests
    print("\n=== Starting Performance Tests ===")
    print("1. ncRNA Encryption/Decryption Performance Test")
    test_performance()
    
    print("\n2. ncRNA, AES, and RSA Performance Comparison Test")
    test_comparison()

def chunked_encryption(plaintext, chunk_size=400):
    chunks = [plaintext[i:i+chunk_size] for i in range(0, len(plaintext), chunk_size)]
    encrypted_chunks = []
    for chunk in chunks:
        # Process each smaller chunk
        encrypted_chunk = encrypt_chunk(chunk) # encrypt_chunk is not defined
        encrypted_chunks.append(encrypted_chunk)
    return combine_chunks(encrypted_chunks) # combine_chunks is not defined

def optimized_nussinov(sequence):
    # Use sparse dynamic programming
    # Store only possible pairing positions
    pairs = {}
    for i in range(len(sequence)):
        for j in range(i + 4, len(sequence)):  # Minimum loop size is 4
            if can_pair(sequence[i], sequence[j]):
                pairs[(i,j)] = True
    # Compute only at possible pairing positions

# Process large files in chunks
def process_large_file(plaintext, chunk_size=1024):
    """Processes large files in chunks to reduce memory usage."""
    chunks = (plaintext[i:i+chunk_size] for i in range(0, len(plaintext), chunk_size))
    results = []
    
    for chunk in chunks:
        encrypted_chunk = encrypt(chunk, seed, seed_sequence, salt) # seed, seed_sequence, salt are not defined in this scope
        results.append(encrypted_chunk)
        
    return combine_results(results)
