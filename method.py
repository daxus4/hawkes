from typing import Any, List, Optional


class Method:
    @classmethod
    def from_conf(cls, name: str, conf: dict):
        return cls(
            name=name,
            submethods=conf.get(f'{name}_parameters', None)
        )
        
    def __init__(self, name: str, submethods: Optional[List[Any]] = None):
        self.name = name
        self.submethods = submethods

    def get_submethods_folders(self) -> List[str]:
        if not self.submethods:
            return [self.name]
        return [f'{self.name}_{submethod}' for submethod in self.submethods]
    
    def get_submethods_values(self) -> List[Any]:
        if not self.submethods:
            return [None]
        return [submethod for submethod in self.submethods]