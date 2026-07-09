# Go Code Review Config

## Linter Commands

Run these in order:

```bash
# Vet (built-in static analysis)
go vet ./... 2>&1 || true

# Comprehensive linting
golangci-lint run ./... 2>&1 || true

# Race detector (if tests exist)
go test -race ./... 2>&1 || true

# Security scanning
gosec ./... 2>&1 || true
```

## Go-Specific Edge Cases

### Error Handling
- Ignored errors (`_ = someFunc()`)
- Errors not wrapped with context
- Panic instead of returning error
- Deferred function errors ignored
- Error shadowing in nested scopes

### Nil Handling
- Nil pointer dereference
- Nil slice/map access (read ok, write panics for map)
- Nil interface vs nil concrete type
- Nil function receivers

### Concurrency
- Goroutine leaks (no way to stop)
- Channel deadlocks
- Race conditions on shared data
- Missing mutex locks
- Copying mutex values
- Context cancellation not propagated

### Slices & Maps
- Slice append capacity gotcha
- Slice aliasing (sharing underlying array)
- Map iteration order non-deterministic
- Concurrent map read/write (use sync.Map)

### Common Bugs
- String concatenation in loops (use strings.Builder)
- defer in loop (deferred until function returns)
- Range loop variable capture in goroutine
- Short variable declaration shadowing
- Unintended variable shadowing with `:=`

### Resource Management
- HTTP response body not closed
- File descriptors not closed
- Database connections not returned to pool
- Context timeout not respected

### Interface Satisfaction
- Interface implemented by pointer vs value receiver
- Empty interface hiding type bugs
- Type assertion panics

## Test Coverage Gaps to Check
- Are error paths tested?
- Are concurrent operations tested with `-race`?
- Are context cancellations handled?
