# TypeScript Code Review Config

## Linter Commands

Run these in order:

```bash
# Type checking
npx tsc --noEmit 2>&1 || true

# Linting
npx eslint src/ tests/ --ext .ts,.tsx 2>&1 || true

# Formatting check
npx prettier --check "src/**/*.ts" "tests/**/*.ts" 2>&1 || true
```

## TypeScript-Specific Edge Cases

### Type System
- `any` types hiding bugs
- Type assertions (`as`) bypassing checks
- `undefined` vs `null` handling
- Optional chaining (`?.`) returning undefined
- Non-null assertion (`!`) used incorrectly
- Union types not narrowed properly

### Async/Promises
- Unhandled promise rejections
- Missing `await` on async functions
- `Promise.all` failing fast (use `Promise.allSettled`?)
- Floating promises (not awaited or returned)

### Arrays & Objects
- Array index access returning `undefined`
- Object spread shallow copy issues
- `in` operator vs `hasOwnProperty`
- Prototype pollution risks

### Common Bugs
- `==` vs `===` comparisons
- Truthy/falsy confusion (`0`, `""`, `false`)
- `typeof null === 'object'`
- Array `.sort()` mutates in place
- `parseInt` radix parameter missing
- Event listener memory leaks

### React-Specific (if applicable)
- Missing dependency arrays in hooks
- Stale closures in useEffect/useCallback
- Keys in lists not unique/stable
- State updates not batched properly
- Memory leaks from subscriptions

### Node.js-Specific (if applicable)
- Callback error handling
- Stream backpressure
- Event emitter memory leaks
- Unhandled `error` events crashing process

## Test Coverage Gaps to Check
- Are error boundaries tested?
- Are loading/error states covered?
- Are race conditions tested?
