# Python TDD Mockup

**Linter**: ruff | **Test**: pytest | **Config**: pyproject.toml

## Project Structure
```
package/
├── src/module/
│   ├── __init__.py
│   └── module.py
└── tests/
    ├── conftest.py
    └── test_module.py
```

## Implementation Mockup
```python
from typing import Optional
from dataclasses import dataclass

@dataclass
class ClassName:
    """[Class docstring]"""
    attr1: str
    attr2: Optional[int] = None

    def method1(self, arg1: str) -> ReturnType:
        """[Method description]"""
        pass
```

## Test Mockup
```python
import pytest
from module import ClassName

@pytest.fixture
def sample_fixture():
    """[Fixture description]"""
    pass

class TestClassName:
    def test_method1_returns_expected(self):
        """Test method1 returns expected result."""
        pass

    def test_method1_raises_on_invalid(self):
        """Test method1 raises ValueError on invalid input."""
        pass
```

## Workflow Commands
```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Lint
ruff check src/ tests/
ruff format src/ tests/
```
