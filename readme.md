# cryto-ncRNA: 基于非编码RNA（ncRNA）的加密算法项目

**cryto-ncRNA** 是一个利用非编码RNA（ncRNA）的生物启发式模型来实现加密算法的项目。本项目旨在通过模拟ncRNA的生物特性，研究其在信息加密、数据保护等领域的潜力。该项目通过结合现代密码学技术与生物序列处理，实现对文本、基因数据等的加密与解密。

## 目录

- [背景](#背景)
- [功能](#功能)
- [依赖](#依赖)
- [安装](#安装)
- [使用](#使用)
  - [加密](#加密)
  - [解密](#解密)
  - [示例](#示例)
- [测试](#测试)
- [结果分析](#结果分析)
- [许可证](#许可证)

## 背景

非编码RNA（ncRNA）在生物体内具有重要的调控作用。它们不仅在基因表达的调控中发挥作用，还展示了高度复杂的序列模式。**cryto-ncRNA**项目通过模拟这些生物序列的动态行为，开发了一种独特的信息加密方式。该项目结合生物序列和现代加密算法，旨在创建一个既具有理论意义又具有实际应用价值的加密系统。




<div align="center">
  <img src="https://github.com/JLU-WangXu/cryto-ncRNA/blob/main/pic/ncRNA.png" alt="ncRNA图示" width="400"/>
</div>



## 功能

- **基于ncRNA的加密算法**：利用非编码RNA的特性进行加密，数据被转换为模拟的RNA序列，并通过自定义加密过程实现信息保护。
  ```pyhton
  def generate_substitution_matrix(seed):
    random.seed(seed)
    bases = ['A', 'C', 'G', 'T']  # 碱基
    substitution = dict()
    shuffled_bases = random.sample(bases, len(bases))
    for i, base in enumerate(bases):
        substitution[base] = shuffled_bases[i]
    return substitution
  
  def transcribe_dna_to_rna(plaintext, substitution_matrix):
    transcribed = ''.join([substitution_matrix.get(char, char) for char in plaintext])
    return transcribed
  ```

- **动态密钥生成**：采用动态生成的密钥进行加密，密钥基于输入数据的特定属性或时间生成。
 ```pyhton
def generate_dynamic_key(seed=None):
    if seed is None:
        now = datetime.datetime.now()
        seed = int(now.strftime('%Y%m%d%H%M%S'))

    seed_str = str(seed)
    hash_object = hashlib.sha256(seed_str.encode())
    dynamic_key = int(hash_object.hexdigest(), 16) % (2**128)
    
    return dynamic_key

def apply_dynamic_key(data, key):
    key_bin = format(key, '0128b')
    data_bin = ''.join(format(ord(char), '08b') for char in data)  # 将数据转换为二进制格式
    
    # 使用异或操作将数据与密钥进行加密
    encrypted_data = ''.join('1' if data_bin[i] != key_bin[i % len(key_bin)] else '0' for i in range(len(data_bin)))
    
    # 将二进制数据转换回字符
    chars = [chr(int(encrypted_data[i:i+8], 2)) for i in range(0, len(encrypted_data), 8)]
    return ''.join(chars)

def reverse_dynamic_key(data, key):
    # 和apply_dynamic_key相同，异或操作是对称的，解密时调用相同逻辑
    return apply_dynamic_key(data, key)
 ```
  
- **冗余保护**：在加密数据中加入冗余位以增强其完整性和抗攻击性。
 ```pyhton
  def insert_redundancy(encrypted_data):
    redundancy = ''.join(random.choice('01') for _ in range(8))  # 8位随机冗余
    return encrypted_data + redundancy
 ```

- **加密与解密功能**：实现了完整的加密和解密过程，可以应用于文本、基因数据等多种数据类型。
  
 ```pyhton
def encrypt(plaintext, seed=None):
    start_time = time.time()  # 开始计时
    if seed is None:
        seed = "initial_seed_value"
    substitution_matrix = generate_substitution_matrix(seed)
    
    transcribed_data = transcribe_dna_to_rna(plaintext, substitution_matrix)
    print(f"转录后的数据：{transcribed_data}")
    
    spliced_data, original_order = split_and_splice(transcribed_data)
    print(f"剪接后的数据：{spliced_data}")
    
    dynamic_key = generate_dynamic_key(seed=int(seed))
    print(f"动态密钥：{dynamic_key}")
    
    encrypted_data = apply_dynamic_key(spliced_data, dynamic_key)
    print(f"应用动态密钥后的加密数据：{encrypted_data}")
    
    encrypted_with_redundancy = insert_redundancy(encrypted_data)
    print(f"加密后的数据（含冗余）：{encrypted_with_redundancy}")
    
    end_time = time.time()  # 结束计时
    encryption_time = end_time - start_time  # 计算加密时间
    print(f"加密时间：{encryption_time:.6f} 秒")
    
    return encrypted_with_redundancy, original_order, encryption_time

```

```pyhton
def decrypt(encrypted_with_redundancy, seed, original_order):
    start_time = time.time()  # 开始计时
    encrypted_data = encrypted_with_redundancy[:-8]  # 移除最后8位冗余数据
    print(f"移除冗余后的数据：{encrypted_data}")
    
    dynamic_key = generate_dynamic_key(seed=int(seed))  # 确保解密时使用相同的密钥生成逻辑
    print(f"动态密钥：{dynamic_key}")
    
    decrypted_spliced_data = reverse_dynamic_key(encrypted_data, dynamic_key)
    print(f"使用动态密钥解密后的数据：{decrypted_spliced_data}")
    
    decrypted_data = inverse_splice(decrypted_spliced_data, original_order)
    print(f"逆剪接后的数据：{decrypted_data}")
    
    inverse_substitution_matrix = {v: k for k, v in generate_substitution_matrix(seed).items()}
    decrypted_data = ''.join([inverse_substitution_matrix.get(char, char) for char in decrypted_data])
    print(f"逆转录后的解密数据：{decrypted_data}")
    
    end_time = time.time()  # 结束计时
    decryption_time = end_time - start_time  # 计算解密时间
    print(f"解密时间：{decryption_time:.6f} 秒")
    
    return decrypted_data, decryption_time




```
## 依赖

为了运行该项目，您需要以下依赖项：

- Python 3.x
- `pycryptodome`：用于加密和解密功能的库
- `numpy`：用于数据处理
- `matplotlib`：用于结果的可视化（可选）

通过以下命令安装依赖项：

```bash
pip install pycryptodome numpy matplotlib
```

## 安装

1. 克隆项目到本地：

    ```bash
    git clone https://github.com/JLU-WangXu/cryto-ncRNA.git
    cd cryto-ncRNA
    ```

2. 安装所有必要的依赖项（如上所示）：

    ```bash
    pip install pycryptodome numpy matplotlib
    ```

3. 确保你使用的是Python 3.x版本。

## 使用

### 加密

要进行加密，可以使用 `encrypt()` 函数，该函数接受文本或基因序列作为输入，并通过自定义加密算法返回加密后的数据。

```python
from encryption_algorithms import encrypt

# 输入数据
plaintext = "ACGTACGTACGT"

# 加密数据
encrypted_data, original_order, encryption_time = encrypt(plaintext, seed="123456789")

print(f"Encrypted Data: {encrypted_data}")
```
## 解密

要解密加密的数据，可以使用 `decrypt()` 函数，确保使用与加密时相同的动态密钥和顺序。

```python
from encryption_algorithms import decrypt

# 解密数据
decrypted_data, decryption_time = decrypt(encrypted_data, seed="123456789", original_order=original_order)

print(f"Decrypted Data: {decrypted_data}")
```

### 示例

可以通过 main 函数测试加密解密流程，并验证解密的准确性：

```python

if __name__ == "__main__":
    plaintext = "ACGTACGTACGT"

    # 加密
    encrypted_data, original_order, encryption_time = encrypt(plaintext, seed="123456789")
    
    # 解密
    decrypted_data, decryption_time = decrypt(encrypted_data, seed="123456789", original_order=original_order)
    
    # 准确性验证
    if plaintext == decrypted_data:
        accuracy = 100.0
    else:
        accuracy = 0.0
    print(f"Encryption Accuracy: {accuracy}%")
    print(f"Encryption Time: {encryption_time:.6f} seconds")
    print(f"Decryption Time: {decryption_time:.6f} seconds")
```

## 测试
可以使用 test_algorithms.py 文件中定义的函数对不同类型的数据集（文本、基因数据）进行加密和解密测试。通过运行此脚本，你可以获得加密时间、解密时间、熵值等结果。

运行以下命令启动测试：
```python
python test_algorithms.py
```

## 结果分析
算法的性能和准确性将在控制台输出，并包括以下信息：

加密时间：表示加密过程的运行时间。
解密时间：表示解密过程的运行时间。
准确性：解密后数据与原始数据的匹配度。
熵值：用于评估加密后数据的随机性。

## 许可证
此项目根据MIT许可证开源，详情请参阅 LICENSE 文件。

