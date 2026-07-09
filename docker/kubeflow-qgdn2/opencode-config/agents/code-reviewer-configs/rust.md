# Rust Code Review Config

## Linter Commands

Run these in order:

```bash
# Compiler warnings
cargo build 2>&1 || true

# Clippy (comprehensive linting)
cargo clippy -- -D warnings 2>&1 || true

# Format check
cargo fmt --check 2>&1 || true

# Security audit
cargo audit 2>&1 || true

# Unsafe code detection
cargo geiger 2>&1 || true
```

## Rust-Specific Edge Cases

### Ownership & Borrowing
- Use after move
- Borrow checker workarounds that hide bugs
- Unnecessary clones (performance)
- Lifetime issues masked by `'static`

### Error Handling
- `unwrap()` / `expect()` in library code
- `?` propagating wrong error type
- Error context lost (use `anyhow` or `thiserror`)
- Panic in destructors

### Option & Result
- `unwrap()` on None/Err
- Using `if let` when `match` is clearer
- Not using combinators (`map`, `and_then`)
- `ok()` silently discarding errors

### Unsafe Code
- Incorrect pointer arithmetic
- Aliasing violations
- Uninitialized memory access
- Missing safety comments on `unsafe fn`
- FFI boundary issues

### Concurrency
- Deadlocks with multiple Mutex
- Poisoned mutex not handled
- `Arc<Mutex<>>` when `Arc<AtomicX>` suffices
- Send/Sync bounds missing or incorrect
- Channel disconnection not handled

### Common Bugs
- Integer overflow (debug panics, release wraps)
- Off-by-one in slice indexing
- `==` on floats
- String vs &str confusion
- Forgetting to handle all enum variants

### Async Rust
- Holding lock across `.await`
- Blocking in async context
- Future not `Send` when needed
- Cancellation safety issues
- `select!` fairness problems

### Performance
- Unnecessary allocations
- Large types on stack
- Missing `#[inline]` on hot paths
- Box<dyn Trait> when generics work

## Test Coverage Gaps to Check
- Are `Result::Err` paths tested?
- Are panic conditions documented and tested?
- Are unsafe blocks minimized and justified?
