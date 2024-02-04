def is_available() -> bool:
    return False

def device(idx: int):
    raise NotImplementedError('python_directml is not installed.')

def device_count() -> int:
    return 0

def device_name(idx: int):
    raise NotImplementedError('python_directml is not installed.')
