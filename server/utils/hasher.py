from io import FileIO

BUF_SIZE = 65536
MEMORY_VIEW = memoryview(bytearray(BUF_SIZE))

def compute_hash(f: FileIO, hasher) -> str:
    while bytes_read := f.readinto(MEMORY_VIEW):
        hasher.update(MEMORY_VIEW[:bytes_read])
    return hasher.hexdigest()
