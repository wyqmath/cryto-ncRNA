# Crypto-ncRNA: A Bio-Inspired Encryption Algorithm Based on Non-Coding RNA

**Crypto-ncRNA** is a research project developing a bio-inspired encryption algorithm based on the unique characteristics of non-coding RNA (ncRNA). It merges principles from biological sequences and processes with modern cryptographic techniques to explore the potential of ncRNA in information encryption and data protection. The project simulates the dynamic behavior of ncRNA, such as sequence transcription, structural folding, and the use of biological data for key generation, to create a novel encryption system. This system is designed to encrypt and decrypt various data types, including text and genetic sequences, with an emphasis on security and biological plausibility.

The algorithm has evolved to incorporate increasingly complex and secure mechanisms, drawing inspiration from biological phenomena to enhance its cryptographic strength.

### Background

Non-coding RNAs (ncRNAs) play crucial regulatory roles within biological systems. Beyond their involvement in gene expression regulation, ncRNAs exhibit highly complex sequence patterns and structural dynamics. The **crypto-ncRNA** project is inspired by these properties, aiming to simulate the dynamic behavior of ncRNA sequences to develop a unique approach to information encryption. By integrating these biological concepts with established cryptographic methods, the project seeks to create an encryption system that is both theoretically intriguing and practically robust.

### Core Concepts and Mechanisms

The crypto-ncRNA algorithm is built upon several key bio-inspired mechanisms:

#### 1. Plaintext to Codon Encoding

*   **Base64 and Codon Mapping:** Input plaintext is first encoded into a Base64 string. Each character in the Base64 string (excluding padding '=') is then mapped to a specific 3-nucleotide codon from a predefined list of 64 codons ('ACGU' combinations). This creates an initial RNA-like sequence.

#### 2. ncRNA-Inspired Sequence Transformation (Codon Substitution)

*   **Transcription Simulation and Substitution:** The core idea involves simulating RNA transcription. The codon sequence undergoes a substitution process, inspired by sequence variations.
*   **Codon Substitution Matrix:** A substitution matrix is generated based on a user-provided `seed`. This matrix maps each of the 64 possible codons to another unique codon in a shuffled manner. This significantly increases complexity compared to single base substitutions. The 64! (approximately 1.27 x 10⁸⁹) possible matrices make exhaustive search computationally infeasible.
    ```python
    # Conceptual: Generating a codon substitution matrix (from ncRNA3.5.py)
    # import random
    # def codon_generator():
    #     for a in 'ACGU':
    #         for b in 'ACGU':
    #             for c in 'ACGU':
    #                 yield a + b + c
    #
    # def generate_codon_substitution_matrix(seed):
    #     rng = random.Random(seed)
    #     codons_list = list(codon_generator())
    #     shuffled = codons_list.copy()
    #     rng.shuffle(shuffled)
    #     return {k: v for k, v in zip(codons_list, shuffled)}
    ```

#### 3. Non-Linear Transformation via RNA Secondary Structure Simulation

To introduce non-linearity, a critical property for strong cryptographic systems (providing confusion and diffusion), the algorithm simulates the folding of RNA into secondary structures.
*   **RNA Folding Principle:** RNA molecules fold into complex 2D structures (e.g., hairpins, stem-loops) based on their sequence. This folding is an inherently non-linear process.
*   **LinearFold Algorithm Integration:** The `linear_fold` algorithm is used to predict the RNA secondary structure from the substituted codon sequence (concatenated into a single string). The algorithm identifies base pairings. The sequence elements are then reordered based on these predicted pairings (paired indices followed by unpaired indices). This step significantly enhances resistance to linear cryptanalysis. The order of rearrangement (`indices_order`) is saved for the decryption process.
    ```python
    # Conceptual: Applying RNA secondary structure (from ncRNA3.5.py)
    # def linear_fold(sequence):
    #     n = len(sequence)
    #     dp = [0] * n # Simplified representation
    #     stack = []
    #     structure = ['.' for _ in range(n)]
    #     # ... (logic for pairing) ...
    #     return ''.join(structure)
    #
    # def apply_rna_secondary_structure(codon_sequence):
    #     base_sequence = ''.join(codon_sequence)
    #     structure = linear_fold(base_sequence)
    #     paired_indices = [i for i, c in enumerate(structure) if c in '()']
    #     unpaired_indices = [i for i, c in enumerate(structure) if c == '.']
    #     indices_order = paired_indices + unpaired_indices
    #     new_sequence = ''.join(base_sequence[i] for i in indices_order)
    #     new_codon_sequence = [new_sequence[i:i+3] for i in range(0, len(new_sequence), 3)]
    #     return new_codon_sequence, indices_order
    ```

#### 4. Dynamic Key Generation and Stream Encryption

The security of the encryption heavily relies on the cryptographic key.
*   **Key Derivation from Biological Sequence:** A dynamic key is generated from a user-provided `seed_sequence` (an RNA sequence) and a `salt` using the PBKDF2 algorithm with SHA256 as the hash function. This process generates a robust cryptographic key (e.g., 256-bit).
    ```python
    # Conceptual: Dynamic key generation (from ncRNA3.5.py)
    # from Crypto.Protocol.KDF import PBKDF2
    # from Crypto.Hash import SHA256
    # def generate_dynamic_key_from_biological_data(seed_sequence, salt, iterations=100000):
    #     valid_bases = set('ACGU')
    #     if not set(seed_sequence.upper()).issubset(valid_bases):
    #         raise ValueError("Seed sequence contains invalid bases.")
    #     dynamic_key = PBKDF2(seed_sequence, salt, dkLen=32, count=iterations, hmac_hash_module=SHA256)
    #     return dynamic_key
    ```
*   **Stream Encryption (ChaCha20):** The structurally transformed sequence is then encrypted using the ChaCha20 stream cipher with the generated dynamic key. ChaCha20 is a secure and efficient stream cipher. The output includes the nonce and the ciphertext.

#### 5. Integrity Check via Checksum

To ensure data integrity:
*   **SHA-256 Checksum:** A SHA-256 hash of the encrypted data (nonce + ciphertext) is calculated and appended to it. This checksum is used during decryption to verify that the data has not been corrupted or tampered with.
    ```python
    # Conceptual: Adding a checksum (from ncRNA3.5.py)
    # import hashlib
    # def add_checksum(encrypted_data):
    #     checksum = hashlib.sha256(encrypted_data).digest()
    #     return encrypted_data + checksum
    ```

### Overall Encryption Process (Flow based on `ncRNA3.5.py`)

The encryption process integrates these mechanisms:

1.  **Plaintext Encoding:** Input plaintext is encoded into a sequence of codons via Base64 encoding and subsequent mapping to a predefined codon set.
2.  **Substitution Matrix Generation:** A codon substitution matrix is generated using a `seed`.
3.  **Codon Substitution:** The encoded codon sequence undergoes substitution using the generated matrix.
4.  **Non-Linear Transformation (RNA Folding):** The substituted sequence is transformed non-linearly using the `linear_fold` algorithm to simulate RNA secondary structure. The reordering indices (`indices_order`) are saved.
5.  **Dynamic Key Generation:** A dynamic key is generated from a `biological_key_seed` (RNA sequence) and a `salt` using PBKDF2.
6.  **Stream Encryption:** The transformed sequence (concatenated) is encrypted using ChaCha20 with the dynamic key.
7.  **Checksum Addition:** A SHA-256 checksum is calculated from the encrypted data (nonce + ciphertext) and appended.

The output includes the final encrypted data (nonce + ciphertext + checksum), the `substitution_matrix` (or the seed to regenerate it), and the `indices_order`. The `seed`, `biological_key_seed`, and `salt` must be securely managed and available for decryption.

### Overall Decryption Process (Flow based on `ncRNA3.5.py`)

Decryption reverses the encryption steps, requiring the same seeds, salt, substitution matrix, and saved order information:

1.  **Verify and Remove Checksum:** The checksum is separated from the received data. The checksum of the remaining data (nonce + ciphertext) is computed and verified against the received checksum. If they match, the checksum is removed.
2.  **Dynamic Key Regeneration:** The dynamic key is regenerated using the original `biological_key_seed` and `salt`.
3.  **Stream Decryption:** The data (nonce + ciphertext) is decrypted using ChaCha20 with the regenerated dynamic key. The result is a string of nucleotides.
4.  **Inverse Non-Linear Transformation:** The non-linear transformation is reversed using the saved `indices_order` to restore the original substituted codon sequence.
5.  **Inverse Substitution:** The original codon substitution matrix is regenerated (if not provided directly) using the `seed`. The substitution is reversed to get the encoded codon sequence.
6.  **Plaintext Decoding:** The sequence of codons is decoded back to the original plaintext by reversing the codon-to-Base64 character mapping and then Base64 decoding.

### Security Analysis

The security of crypto-ncRNA leverages several factors:

*   **Substitution Matrix Complexity:** Using 64 codons for substitution results in 64! (approximately 1.27 x 10⁸⁹) possible matrices, making brute-force attacks on this component computationally infeasible.
*   **Non-Linear Transformation:** The RNA secondary structure folding step (using `linear_fold`) introduces non-linearity. This transformation enhances resistance against linear cryptanalysis and contributes to the algorithm's confusion and diffusion properties.
*   **Dynamic Key Security and Encryption Strength:**
    *   **Strong Key Derivation:** PBKDF2 is a standard key derivation function that helps protect against brute-force attacks on the `biological_key_seed`.
    *   **Secure Stream Cipher:** ChaCha20 is a well-vetted, secure stream cipher providing confidentiality.
    *   **High Entropy Seeds:** Using actual biological sequences as seeds for key generation, combined with PBKDF2, produces keys with a large keyspace.
*   **Integrity Check:** The SHA-256 checksum provides a strong guarantee of data integrity.
*   **Overall Security:** The combination of a large substitution space, non-linear transformations, strong key derivation, and a modern stream cipher aims to provide resilience against various cryptanalytic attacks:
    *   **Known-Plaintext Attacks:** The complexity of the combined transformations makes it difficult to deduce the key even if some plaintext-ciphertext pairs are known.
    *   **Frequency Analysis Attacks:** Codon-level substitution, followed by structural rearrangement and strong encryption, significantly obscures original character or symbol frequencies.
    *   **Chosen-Plaintext Attacks:** The strength of the ChaCha20 cipher and the dynamic key make it challenging for an attacker to gain significant information by choosing plaintexts to encrypt.

*Further mathematical proofs and detailed security analyses would be necessary to rigorously quantify the security levels against specific attack models.*

### Dependencies

To run this project, you will need:
- Python 3.x
- `pycryptodome`: For cryptographic functions like AES (used conceptually or in prior versions, current ChaCha20 also from this library), PBKDF2, SHA256.
- `numpy`: For numerical operations, particularly in codon and Base64 character array handling.
- `matplotlib`: For visualizing performance metrics (optional, used in testing functions).

Install dependencies using pip:
```bash
pip install pycryptodome numpy matplotlib
```

### Installation

1.  Clone the project repository:
    ```bash
    git clone https://github.com/JLU-WangXu/crypto-ncRNA.git
    cd crypto-ncRNA
    ```
2.  Install the necessary dependencies as shown above.
3.  Ensure you are using Python 3.x.

### Usage

The core functionalities are exposed through `encrypt` and `decrypt` functions within the `ncRNA3.5.py` script.

#### Encryption

To encrypt data, call the `encrypt` function, providing the plaintext, a seed for the substitution matrix, an RNA sequence as a seed for the dynamic key, and a salt.

```python
# Example from ncRNA3.5.py
from ncRNA3.5 import encrypt, generate_codon_substitution_matrix # Assuming ncRNA3.5.py is in the Python path or same directory

plaintext = "Hello, World! This is a test of the encryption algorithm based on ncRNA."
substitution_seed = "123456789"  # Seed for substitution matrix
biological_key_seed_rna = "ACGUACGUACGUACGUACGUACGUACGUACGU"  # RNA sequence for dynamic key
salt_bytes = b'salt_123'  # Salt for key derivation

# Encrypt the data
encrypted_data_package, sub_matrix, order_indices = encrypt(
    plaintext, 
    substitution_seed, 
    biological_key_seed_rna, 
    salt_bytes
)

print(f"Encrypted Data (first 50 bytes): {encrypted_data_package[:50]}...")
# The 'sub_matrix' and 'order_indices' are needed for decryption.
# In a real application, 'substitution_seed' would be used to regenerate 'sub_matrix' at decryption.
```

#### Decryption

To decrypt, use the `decrypt` function with the encrypted data package, the original seeds, salt, the substitution matrix (or seed to regenerate it), and the `indices_order`.

```python
# Example from ncRNA3.5.py (continued from encryption)
from ncRNA3.5 import decrypt 

# For decryption, you need the same seeds, salt, substitution matrix, and indices_order
# If you only stored the substitution_seed, regenerate the matrix:
# sub_matrix_for_decrypt = generate_codon_substitution_matrix(substitution_seed)

decrypted_plaintext = decrypt(
    encrypted_data_package,
    substitution_seed, # or pass sub_matrix directly if the decrypt function is modified to accept it
    biological_key_seed_rna,
    salt_bytes,
    sub_matrix, # Pass the generated substitution matrix
    order_indices 
)

print(f"Decrypted Data: {decrypted_plaintext}")

# Accuracy Verification
if plaintext == decrypted_plaintext:
    print("Accuracy: 100.0% - Decryption successful!")
else:
    print("Accuracy: 0.0% - Decryption failed!")
    print(f"Original:  '{plaintext}'")
    print(f"Decrypted: '{decrypted_plaintext}'")
```

Note: The `encrypt` function in `ncRNA3.5.py` returns `encrypted_data_with_checksum, substitution_matrix, indices_order`. The `decrypt` function expects these, along with the original seeds and salt.

### Testing

The `ncRNA3.5.py` script includes functions for performance testing (`test_performance`) and comparison with other algorithms like AES and RSA (`test_comparison`). These can be run by executing the script:

```bash
python ncRNA3.5.py
```

This will typically output:
*   Performance metrics (encryption/decryption time vs. plaintext length).
*   Comparison results against AES and RSA.
*   Entropy information and byte frequency distribution plots if `DEBUG` is enabled in the `if __name__ == "__main__":` block.

### Results Analysis

The output from tests or direct usage may include:

*   **Encryption Time:** Time taken for the encryption process.
*   **Decryption Time:** Time taken for the decryption process.
*   **Accuracy:** Verification that the decrypted data matches the original plaintext.
*   **Entropy:** A measure of the randomness of the encrypted data, which can be an indicator of cryptographic strength (calculated in the test code).
*   **Performance Comparison:** Benchmarks against standard ciphers like AES and RSA.

### License

This project is open-sourced under the MIT License. See the `LICENSE` file in the repository for more details.
