# Rust TDD Mockup

**Linter**: clippy | **Test**: cargo test | **Config**: Cargo.toml

## Project Structure
```
crate/
├── src/
│   ├── lib.rs
│   └── module.rs
├── tests/
│   └── integration_test.rs
└── Cargo.toml
```

## Implementation Mockup
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModuleError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub struct ClassName {
    attr1: String,
    attr2: Option<i32>,
}

impl ClassName {
    pub fn new(attr1: String, attr2: Option<i32>) -> Result<Self, ModuleError> {
        if attr1.is_empty() {
            return Err(ModuleError::InvalidInput("attr1 cannot be empty".into()));
        }
        Ok(Self { attr1, attr2 })
    }

    pub fn method1(&self, arg1: &str) -> Result<ReturnType, ModuleError> {
        // [Implementation]
        Ok(result)
    }
}
```

## Test Mockup
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid_input() {
        let result = ClassName::new("value".to_string(), Some(1));
        assert!(result.is_ok());
    }

    #[test]
    fn test_new_empty_attr1_fails() {
        let result = ClassName::new("".to_string(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_method1_returns_expected() {
        let instance = ClassName::new("value".to_string(), None).unwrap();
        let result = instance.method1("arg").unwrap();
        assert_eq!(result, expected);
    }
}
```

## Workflow Commands
```bash
# Run tests
cargo test

# Run with coverage (requires cargo-tarpaulin)
cargo tarpaulin --out Html

# Lint
cargo clippy -- -D warnings
cargo fmt --check
```
