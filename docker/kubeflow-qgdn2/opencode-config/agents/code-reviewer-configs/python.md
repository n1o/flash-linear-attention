# Python Code Review Config

## Linter Commands

Run these in order:

```bash
# Type checking
mypy --strict src/ 2>&1 || true

# Linting + formatting check
ruff check src/ tests/ 2>&1 || true
ruff format --check src/ tests/ 2>&1 || true

# Security scanning
bandit -r src/ 2>&1 || true

# Complexity analysis
radon cc src/ -a -s 2>&1 || true
```

## Python-Specific Edge Cases

### Type System
- `None` vs missing key vs empty value
- `Optional` fields that aren't checked
- Type narrowing not working as expected
- Mutable default arguments (`def foo(items=[])`)

### Async/Await
- Missing `await` on coroutines
- Blocking calls in async functions
- Unclosed async resources (aiohttp sessions, etc.)
- Race conditions with shared state

### Collections
- Modifying list/dict while iterating
- KeyError on dict access (use `.get()`)
- Empty sequence truthiness confusion
- Shallow vs deep copy issues

### Common Bugs
- `is` vs `==` for comparisons
- Integer caching surprises (`is` on large ints)
- String encoding issues (bytes vs str)
- `except:` catching too much (including SystemExit)
- Circular imports

### Resource Management
- Missing `with` for file operations
- Database connections not closed
- Thread pool exhaustion
- Memory leaks in long-running processes

## Test Coverage Gaps to Check
- Are exceptions properly tested?
- Are async code paths tested?
- Are edge cases from type hints covered?
