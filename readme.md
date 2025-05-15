# crypto-ncRNA: A Bio-Inspired Encryption Algorithm Based on Non-Coding RNA

**crypto-ncRNA** is a research project developing a bio-inspired encryption algorithm based on the unique characteristics of non-coding RNA (ncRNA). It merges principles from biological sequences and processes with modern cryptographic techniques to explore the potential of ncRNA in information encryption and data protection. The project simulates the dynamic behavior of ncRNA, such as sequence transcription, structural folding, and the use of biological data for key generation, to create a novel encryption system. This system is designed to encrypt and decrypt various data types, including text and genetic sequences, with an emphasis on security and biological plausibility.

The algorithm has evolved to incorporate increasingly complex and secure mechanisms, drawing inspiration from biological phenomena to enhance its cryptographic strength.

### Background

Non-coding RNAs (ncRNAs) play crucial regulatory roles within biological systems. Beyond their involvement in gene expression regulation, ncRNAs exhibit highly complex sequence patterns and structural dynamics. The **crypto-ncRNA** project is inspired by these properties, aiming to simulate the dynamic behavior of ncRNA sequences to develop a unique approach to information encryption. By integrating these biological concepts with established cryptographic methods, the project seeks to create an encryption system that is both theoretically intriguing and practically robust.

### Core Concepts and Mechanisms

The crypto-ncRNA algorithm is built upon several key bio-inspired mechanisms that have been progressively refined:

#### 1. ncRNA-Inspired Sequence Transformation

*   **Transcription Simulation and Substitution:** The core idea involves simulating RNA transcription. Data is first represented as a sequence analogous to RNA bases. A substitution process, inspired by base pairing rules and sequence variations, is applied.
    *   **Evolution from Base to Codon Substitution:** To significantly increase complexity and resistance to brute-force attacks, the substitution mechanism was enhanced from single bases (e.g., 'A', 'C', 'G', 'T', offering 4! = 24 substitution possibilities) to codons (triplets of nucleotides, e.g., 'AUG', 'CGA'). Using 4 bases, there are 4³ = 64 distinct codons. This expands the substitution matrix possibilities to 64!, making exhaustive search computationally infeasible.
    # Conceptual: Generating a codon substitution matrix
# import random
# def generate_codon_substitution_matrix(seed):
#     random.seed(seed)
#     bases = ['A', 'C', 'G', 'U'] # Or T
#     codons = [b1+b2+b3 for b1 in bases for b2 in bases for b3 in bases]
#     shuffled_codons = random.sample(codons, len(codons))
#     substitution_map = {codons[i]: shuffled_codons[i] for i in range(len(codons))}
#     return substitution_map

#### 2. Non-Linear Transformation via RNA Secondary Structure Simulation

To introduce non-linearity, a critical property for strong cryptographic systems (providing confusion and diffusion), the algorithm simulates the folding of RNA into secondary structures.
*   **RNA Folding Principle:** RNA molecules fold into complex 2D structures (e.g., hairpins, stem-loops) based on their sequence. This folding is an inherently non-linear process.
*   **Nussinov Algorithm Integration:** The Nussinov algorithm, a dynamic programming method for predicting RNA secondary structures by maximizing complementary base pairs, is employed to simulate this folding. The structural information derived (e.g., base pairing patterns represented in dot-bracket notation) is used to non-linearly transform the sequence, for instance, by reordering sequence elements based on predicted pairings. This step significantly enhances resistance to linear cryptanalysis. The order of rearrangement is saved for the decryption process.
    # Simplified Nussinov algorithm for RNA secondary structure prediction (as per project's v3)
# import random # Not needed for this specific Nussinov
# import hashlib # Not needed for this specific Nussinov

def nussinov_algorithm(sequence):
    n = len(sequence)
    dp = [[0]*n for _ in range(n)]

    # Define base pairing rules (canonical and G-U wobble)
    def can_pair(b1, b2):
        pairs = [('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C'), ('G', 'U'), ('U', 'G')]
        return (b1, b2) in pairs

    # Dynamic programming to fill the matrix, following the project's specific recurrence
    for k in range(1, n): # k is the length of the subsequence minus 1
        for i in range(n - k): # i is the start index of the subsequence
            j = i + k # j is the end index of the subsequence

            # Base score for the inner part dp[i+1][j-1]
            inner_score = 0
            if i + 1 <= j - 1:
                inner_score = dp[i+1][j-1]

            # 'pair_term_value' considers if sequence[i] and sequence[j] pair,
            # or if they are just ends of the segment.
            # This reflects the specific logic: `pair = dp[i+1][j-1] + 1` if can_pair else `dp[i+1][j-1]`
            if can_pair(sequence[i], sequence[j]):
                pair_term_value = inner_score + 1
            else:
                pair_term_value = inner_score

            # 'unpair_term_value' considers if sequence[i] or sequence[j] is unpaired.
            score_i_unpaired = dp[i+1][j] if i+1 <= j else 0
            score_j_unpaired = dp[i][j-1] if i <= j-1 else 0
            unpair_term_value = max(score_i_unpaired, score_j_unpaired)

            dp[i][j] = max(pair_term_value, unpair_term_value)

    # Generate secondary structure (dot-bracket notation) using traceback
    structure = ['.'] * n
    def traceback_structure(curr_i, curr_j):
        if curr_i < curr_j:
            # This traceback logic needs to correctly reverse the DP choices.
            # The prompt's v3 traceback:
            # if dp[curr_i][curr_j] == dp[curr_i+1][curr_j]: (if i is unpaired)
            #    traceback_structure(curr_i+1, curr_j)
            # elif dp[curr_i][curr_j] == dp[curr_i][curr_j-1]: (if j is unpaired)
            #    traceback_structure(curr_i, curr_j-1)
            # else: (if i,j pair or form ends of segment from inner_score)
            #    structure[curr_i] = '('
            #    structure[curr_j] = ')'
            #    traceback_structure(curr_i+1, curr_j-1)
            # This simplified traceback is used in the project.

            # Check if curr_j is unpaired
            if dp[curr_i][curr_j] == (dp[curr_i][curr_j-1] if curr_i <= curr_j-1 else 0):
                traceback_structure(curr_i, curr_j-1)
            # Check if curr_i is unpaired
            elif dp[curr_i][curr_j] == (dp[curr_i+1][curr_j] if curr_i+1 <= curr_j else 0):
                traceback_structure(curr_i+1, curr_j)
            # Check if curr_i and curr_j pair
            elif can_pair(sequence[curr_i], sequence[curr_j]) and \
                 dp[curr_i][curr_j] == ((dp[curr_i+1][curr_j-1] if curr_i+1 <= curr_j-1 else 0) + 1):
                structure[curr_i] = '('
                structure[curr_j] = ')'
                if curr_i + 1 <= curr_j - 1:
                     traceback_structure(curr_i+1, curr_j-1)
            # Fallback for the case where i,j don't pair but dp[i][j] was from inner_score
            elif curr_i + 1 <= curr_j - 1:
                 traceback_structure(curr_i+1, curr_j-1)


    traceback_structure(0, n-1)
    return ''.join(structure)

#### 3. Dynamic Key Generation

The security of the encryption heavily relies on the cryptographic key.
*   **Evolution of Key Generation:**
    *   Initially, keys might have been generated based on simpler seeds like timestamps or basic input attributes.
    *   To enhance security and unpredictability, the system evolved to use actual biological sequences (e.g., RNA sequences from public databases like NCBI or user-provided sequences) as seeds for key generation. These sequences, known for their inherent complexity and high entropy, are processed (e.g., via SHA-256 hashing) to produce robust cryptographic keys (e.g., 256-bit).
    import hashlib
# import datetime # For older version of key generation

# Evolved key generation using a biological sequence as a seed
def generate_dynamic_key_from_biological_data(seed_sequence):
    # Validate sequence (e.g., ensure it contains valid RNA/DNA bases)
    valid_bases = set('ACGU') # Assuming RNA, can be extended for DNA ('T')
    if not set(seed_sequence.upper()).issubset(valid_bases):
        raise ValueError("Seed sequence contains invalid RNA/DNA bases.")

    hash_object = hashlib.sha256(seed_sequence.encode())
    dynamic_key = hash_object.digest()  # Returns bytes, suitable for AES etc.
    # For XOR-based ciphers as in earlier versions, one might use:
    # dynamic_key_int = int(hash_object.hexdigest(), 16)
    return dynamic_key
*   **Application:** The generated dynamic key is used in a symmetric encryption step. Earlier versions might have used XOR operations, while later enhancements suggest using established ciphers like AES for greater security.

#### 4. Redundancy and Error Correction

To ensure data integrity and provide some resilience against minor corruptions or attacks:
*   **Initial Approach:** Simple addition of random redundant bits to the encrypted data.
*   **Enhanced Mechanism:** Inspired by biological error correction mechanisms (e.g., DNA repair pathways), more robust techniques like checksums or error correction codes (ECC) are incorporated. This helps verify data integrity upon decryption and can potentially correct minor errors, enhancing the reliability of the encrypted communication.
    # Conceptual: Adding a checksum or ECC
# def add_error_correction_mechanism(data):
#     # Example: calculate a checksum (e.g., CRC32) or apply a simple ECC
#     checksum = hashlib.md5(data.encode()).hexdigest() # Example checksum
#     return data + "CHECKSUM:" + checksum
# def verify_and_remove_error_correction(data_with_ecc):
#     # Split data and checksum, verify, then return data
#     pass

### Overall Encryption Process (Conceptual Flow)

The encryption process integrates these mechanisms:

1.  **Plaintext Encoding:** Input data (text, gene sequence) is encoded into a sequence of units suitable for biological simulation (e.g., codons).
2.  **Substitution:** The encoded sequence undergoes substitution using a dynamically generated codon substitution matrix (derived from a `substitution_seed`).
3.  **Non-Linear Transformation:** The substituted sequence is transformed non-linearly using RNA secondary structure simulation (e.g., guided by Nussinov algorithm results). The transformation order (`indices_order`) is saved for decryption.
4.  **Dynamic Key Encryption:** A dynamic key is generated from a high-entropy `biological_key_seed`. The transformed sequence is then encrypted using this key (e.g., with AES or a strong XOR-based cipher).
5.  **Error Correction/Redundancy:** An error correction code or checksum is added to the encrypted data.

The output includes the final encrypted data, and information necessary for decryption like the `indices_order`. The seeds (`substitution_seed`, `biological_key_seed`) must be securely managed and available for decryption.

### Overall Decryption Process (Conceptual Flow)

Decryption reverses the encryption steps, requiring the same seeds and saved order information:

1.  **Verify and Remove ECC:** The error correction code/checksum is verified and removed from the received data.
2.  **Dynamic Key Decryption:** The dynamic key is regenerated using the original `biological_key_seed`. The data is decrypted.
3.  **Inverse Non-Linear Transformation:** The non-linear transformation is reversed using the saved `indices_order`.
4.  **Inverse Substitution:** The original codon substitution matrix is regenerated using the `substitution_seed`. The substitution is reversed.
5.  **Plaintext Decoding:** The sequence of units is decoded back to the original plaintext.

### Security Analysis

The security of crypto-ncRNA aims to leverage several factors:

*   **Substitution Matrix Complexity:** Using 64 codons for substitution results in a 64! (approximately 1.27 x 10⁸⁹) possible matrices, making brute-force attacks on this specific component computationally infeasible.
*   **Non-Linear Transformation:** The RNA secondary structure folding step, especially when guided by algorithms like Nussinov, introduces significant non-linearity. Predicting RNA structure is computationally hard (NP-complete for some formulations without simplifications). This transformation enhances resistance against linear cryptanalysis and contributes to the algorithm's confusion and diffusion properties.
*   **Dynamic Key Security:**
    *   **High Entropy Seeds:** Using actual biological sequences (which are inherently complex and can be selected for high entropy) as seeds for key generation, combined with strong cryptographic hash functions (e.g., SHA-256), produces keys with a large keyspace (e.g., 2²⁵⁶).
    *   **Unpredictability:** This makes the keys difficult to predict or brute-force, assuming the seed biological sequence is kept secret or is sufficiently complex.
*   **Overall Security:** The combination of a large substitution space, non-linear transformations, and strong dynamic keys aims to provide resilience against various cryptanalytic attacks:
    *   **Known-Plaintext Attacks:** The complexity of the combined transformations makes it difficult to deduce the key even if some plaintext-ciphertext pairs are known.
    *   **Frequency Analysis Attacks:** Codon-level substitution, followed by structural rearrangement and encryption, significantly obscures original character or symbol frequencies.
    *   **Chosen-Plaintext Attacks:** The non-linearity and the strength of the dynamic key make it challenging for an attacker to gain significant information by choosing plaintexts to encrypt.

*Further mathematical proofs and detailed security analyses, as outlined in the project's research, would be necessary to rigorously quantify the security levels against specific attack models.*

### Dependencies

To run this project, you may need:
- Python 3.x
- `pycryptodome`: For cryptographic functions like AES (if used, as suggested in v2.0 improvements).
- `numpy`: For numerical operations, potentially in implementations of algorithms like Nussinov or data handling.
- `matplotlib`: For visualizing results (optional).

Install dependencies using pip (example):
```bash
pip install pycryptodome numpy matplotlib

Installation
Clone the project repository:
git clone https://github.com/JLU-WangXu/crypto-ncRNA.git
cd crypto-ncRNA

Install the necessary dependencies as shown above.
Ensure you are using Python 3.x.
Usage

The core functionalities are typically exposed through encrypt and decrypt functions within the project's modules.

Encryption

To encrypt data, you would call an encrypt function, providing the plaintext and necessary seeds/parameters.

Example based on the project's v3 structure
from your_encryption_module import encrypt # Assuming functions are organized
plaintext = "This is a secret message using ACGT patterns."
substitution_seed = "my_substitution_key_123" # Seed for substitution matrix
biological_key_seed_rna = "AUGGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUA" # RNA sequence for dynamic key
Encrypt the data
encrypted_data_package = encrypt(plaintext, substitution_seed, biological_key_seed_rna)
This function would return the ciphertext and any other data needed for decryption,
like 'indices_order' from the Nussinov folding.
e.g., encrypted_data_package = (ciphertext, indices_order, checksum_or_ecc_info)
print(f"Encrypted Data: {encrypted_data_package[0]}")
Decryption

To decrypt, use the decrypt function with the encrypted data package and the same seeds/parameters used during encryption.

from your_encryption_module import decrypt
Assuming encrypted_data_package = (ciphertext, indices_order, ...)
decrypted_plaintext = decrypt(encrypted_data_package[0],
substitution_seed,
biological_key_seed_rna,
encrypted_data_package[1]) # Pass indices_order
print(f"Decrypted Data: {decrypted_plaintext}")
Example (Illustrative Main Execution from v1, adapted)
if name == "main":
plaintext = "ACGTACGTACGTACGTACGT"
# In v3, seeds are more distinct:
substitution_seed_for_matrix = "a_secure_seed_for_matrix"
biological_key_seed_for_dyn_key = "UAGCGCUAGCAUGCAUGCUAGCAUGCUAGC"
print(f"Original Plaintext: {plaintext}")
# Encryption (assuming encrypt returns: ciphertext, indices_order, time)
encrypted_output, order_info, enc_time = encrypt(plaintext,
substitution_seed_for_matrix,
biological_key_seed_for_dyn_key)
print(f"Encrypted Output: {encrypted_output}")
print(f"Encryption Time: {enc_time:.6f} seconds")
# Decryption (assuming decrypt needs ciphertext, seeds, order_info)
decrypted_data, dec_time = decrypt(encrypted_output,
substitution_seed_for_matrix,
biological_key_seed_for_dyn_key,
order_info)
print(f"Decrypted Plaintext: {decrypted_data}")
print(f"Decryption Time: {dec_time:.6f} seconds")
# Accuracy Verification
if plaintext == decrypted_data:
print("Accuracy: 100.0% - Decryption successful!")
else:
print("Accuracy: 0.0% - Decryption failed!")
print(f"Original: '{plaintext}'")
print(f"Decrypted: '{decrypted_data}'")

Note: The exact function signatures and return values for encrypt and decrypt will depend on the project's consolidated implementation. The snippets above are illustrative.

Testing

The project likely includes a test script (e.g., test_algorithms.py as mentioned in the original v1.0 documentation) to evaluate the encryption and decryption processes on various datasets (e.g., text, gene data).

To run tests (example command):

python test_algorithms.py


This script would typically output performance metrics and correctness checks.

Results Analysis

The output from tests or direct usage may include:

Encryption Time: Time taken for the encryption process.
Decryption Time: Time taken for the decryption process.
Accuracy: Verification that the decrypted data matches the original plaintext.
Entropy: A measure of the randomness of the encrypted data, which can be an indicator of cryptographic strength (as mentioned in v1.0).
License

This project is open-sourced under the MIT License. See the LICENSE file in the repository for more details.
