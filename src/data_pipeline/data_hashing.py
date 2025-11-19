import hashlib
import pandas as pd
from typing import List


def dataframe_to_stable_bytes(df: List) -> bytes:
    """
    Combines a list of DataFrames into a single, cryptographically stable byte stream.
    This ensures the hash remains consistent across different environments and operating systems.
    """
    stable_string = df.to_csv(
        index=False, 
        encoding='utf-8' 
        # Note: For maximum safety, ensure column order is always sorted before this step.
    )
    return stable_string.encode('utf-8')

def generate_sha256_hash(data_stream:bytes):
    """
    Generates the SHA-256 hash of the given data.

    Args:
        data (str or bytes): The input data to be hashed. If a string,
                             it will be encoded to UTF-8 bytes.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash.
    """
    
    sha256_hash= hashlib.sha256(data_stream).hexdigest()
    return sha256_hash

   
