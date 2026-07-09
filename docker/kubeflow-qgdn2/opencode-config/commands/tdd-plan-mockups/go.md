# Go TDD Mockup

**Linter**: golangci-lint | **Test**: go test | **Config**: go.mod

## Project Structure
```
package/
├── module.go
├── module_test.go
└── go.mod
```

## Implementation Mockup
```go
package module

import "errors"

// ClassName represents [description]
type ClassName struct {
    Attr1 string
    Attr2 int
}

// NewClassName creates a new ClassName instance
func NewClassName(attr1 string, attr2 int) (*ClassName, error) {
    if attr1 == "" {
        return nil, errors.New("attr1 cannot be empty")
    }
    return &ClassName{Attr1: attr1, Attr2: attr2}, nil
}

// Method1 does [description]
func (c *ClassName) Method1(arg1 string) (ReturnType, error) {
    // [Implementation]
    return result, nil
}
```

## Test Mockup
```go
package module

import "testing"

func TestNewClassName(t *testing.T) {
    tests := []struct {
        name    string
        attr1   string
        attr2   int
        wantErr bool
    }{
        {"valid input", "value", 1, false},
        {"empty attr1", "", 1, true},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            _, err := NewClassName(tt.attr1, tt.attr2)
            if (err != nil) != tt.wantErr {
                t.Errorf("NewClassName() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}

func TestClassName_Method1(t *testing.T) {
    c, _ := NewClassName("value", 1)

    got, err := c.Method1("arg")
    if err != nil {
        t.Fatalf("Method1() error = %v", err)
    }
    if got != expected {
        t.Errorf("Method1() = %v, want %v", got, expected)
    }
}
```

## Workflow Commands
```bash
# Run tests
go test ./...

# Run with coverage
go test ./... -coverprofile=coverage.out
go tool cover -html=coverage.out

# Lint
golangci-lint run
```
