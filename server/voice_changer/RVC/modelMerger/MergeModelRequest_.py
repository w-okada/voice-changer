# from dataclasses import dataclass, field
# from typing import List
# from dataclasses_json import dataclass_json


# @dataclass_json
# @dataclass
# class MergeFile:
#     filename: str
#     strength: int


# @dataclass_json
# @dataclass
# class MergeModelRequest:
#     command: str = ""
#     slot: int = -1
#     defaultTune: int = 0
#     defaultIndexRatio: int = 1
#     defaultProtect: float = 0.5
#     files: List[MergeFile] = field(default_factory=lambda: [])
