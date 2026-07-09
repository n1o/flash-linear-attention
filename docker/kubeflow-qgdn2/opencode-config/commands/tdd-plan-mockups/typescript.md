# TypeScript TDD Mockup

**Linter**: eslint | **Test**: vitest/jest | **Config**: package.json, tsconfig.json

## Project Structure
```
package/
├── src/
│   ├── module.ts
│   └── index.ts
└── tests/
    └── module.test.ts
```

## Implementation Mockup
```typescript
export interface IClassName {
  attr1: string;
  attr2?: number;
}

export class ClassName implements IClassName {
  constructor(
    public attr1: string,
    public attr2?: number
  ) {}

  method1(arg1: string): ReturnType {
    // [Implementation]
  }
}
```

## Test Mockup
```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { ClassName } from '../src/module';

describe('ClassName', () => {
  let instance: ClassName;

  beforeEach(() => {
    instance = new ClassName('value');
  });

  it('should return expected result from method1', () => {
    expect(instance.method1('arg')).toBe(expected);
  });

  it('should throw on invalid input', () => {
    expect(() => instance.method1('')).toThrow();
  });
});
```

## Workflow Commands
```bash
# Run tests
npm test
# or
pnpm vitest run

# Run with coverage
pnpm vitest run --coverage

# Lint
pnpm eslint src/ tests/
pnpm tsc --noEmit
```
