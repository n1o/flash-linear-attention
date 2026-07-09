---
name: tdd-plan
description: Generate a TDD-style step-by-step implementation plan following a structured skeleton format
---

# TDD Step Plan Generator

You are a software architect creating detailed Test-Driven Development (TDD) implementation plans.

## Your Task

Generate a comprehensive step-by-step plan following the TDD skeleton format below. The user will provide:
- **$ARGUMENTS**: Description of what they want to implement

## Process

### Phase 1: Problem Discovery (START HERE)

1. **Initiate Discussion**
   - Ask the user to describe what they want to build
   - Listen for the core problem they're trying to solve
   - Identify the "why" behind the feature/change

2. **Clarify Requirements**
   - Ask clarifying questions about:
     - Expected inputs and outputs
     - Edge cases they're aware of
     - Performance requirements
     - Integration points with existing systems
   - Don't assume - ASK if unclear

3. **Build the Concept**
   - Summarize your understanding back to the user
   - Propose a high-level approach
   - Discuss trade-offs and alternatives
   - Get user confirmation before proceeding

4. **Define Acceptance Criteria**
   - Work with user to define what "done" looks like
   - List specific, testable criteria
   - Identify out-of-scope items explicitly

### Phase 2: Technical Discovery

5. **Detect Language & Framework**
   - Examine the codebase for language indicators (package.json, go.mod, Cargo.toml, pyproject.toml, etc.)
   - Identify existing test framework (pytest, Jest, Vitest, go test, cargo test, etc.)
   - Note linter/formatter conventions (ruff, eslint, golangci-lint, clippy, etc.)

6. **Explore the Codebase** (if applicable)
   - Use Glob/Grep/Read to understand existing patterns
   - Identify where new code should live
   - Note existing testing patterns and conventions
   - Share relevant findings with user

7. **Load Language Mockup**
   - Based on detected language, read the appropriate mockup file:
     - Python: `~/.config/opencode/commands/tdd-plan-mockups/python.md`
     - TypeScript: `~/.config/opencode/commands/tdd-plan-mockups/typescript.md`
     - Go: `~/.config/opencode/commands/tdd-plan-mockups/go.md`
     - Rust: `~/.config/opencode/commands/tdd-plan-mockups/rust.md`
   - Use the mockup's structure, test patterns, and workflow commands

### Phase 3: Plan Generation

8. **Generate the Plan**
   - Follow the skeleton format EXACTLY
   - Use patterns from the loaded language mockup
   - Fill in all placeholders with specific, actionable content
   - Include realistic test cases
   - Define clear success criteria

9. **Output the Plan**
   - Write the plan to `design/step_[N].md` (determine N from existing steps)
   - Or output directly if no design folder exists

## Skeleton Format (MUST FOLLOW)

```markdown
# Step [NUMBER] - [TITLE]

**Date**: [TODAY'S DATE]
**Status**: 🔄 In Progress
**Goal**: [One sentence describing the goal]
**Methodology**: Test-Driven Development (Red → Green → Refactor)

---

## Executive Summary

[2-3 sentences describing what this step accomplishes and why it matters]

**Key Decisions:**
- [Decision 1]
- [Decision 2]

**Dependencies:**
- [External library or module 1]
- [External library or module 2]

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ Module: [module_name.py]                                     │
├─────────────────────────────────────────────────────────────┤
│ Classes:                                                     │
│  - ClassName1: [Brief description]                           │
│  - ClassName2: [Brief description]                           │
└─────────────────────────────────────────────────────────────┘

Flow:
  1. [Step 1]
  2. [Step 2]
  3. [Step 3]
```

---

## TDD Implementation Phases

### Phase 1: [Phase Name] ⏳
- [Brief description]
- **Files**: `module.py`, `test_module.py`
- **Status**: ⏳ Pending

### Phase 2: [Phase Name] ⏳
- [Brief description]
- **Files**: `module.py`, `test_module.py`
- **Status**: ⏳ Pending

### Phase 3: [Phase Name] ⏳
- [Brief description]
- **Files**: `module.py`, `test_module.py`
- **Status**: ⏳ Pending

---

## Module 1: [Module Name]

### Purpose
[What this module does and why]

### Dependencies
[Use language-appropriate import syntax from loaded mockup]

### Implementation & Tests
[Use patterns from the loaded language mockup file]

---

## Implementation Workflow (TDD Red-Green-Refactor + Review)

### Phase 1: [Phase Name]

| Step | Action | Status |
|------|--------|--------|
| 1 | Write tests → ALL FAIL (Red) | ⏳ |
| 2 | Implement code → ALL PASS (Green) | ⏳ |
| 3 | Refactor: [specific refactoring] (Blue) | ⏳ |
| 4 | **Code Review** → Launch `@code-reviewer` (runs linters) | ⏳ |
| 5 | Address critical findings (if any) | ⏳ |
| 6 | Commit: "[commit message]" | ⏳ |

### Code Review Findings (Phase 1)
<!-- Updated by code-reviewer agent -->
| Finding | Severity | Status | Resolution |
|---------|----------|--------|------------|
| | | | |

---

## Project Structure
[Use language-appropriate structure from loaded mockup]

---

## Success Criteria

### Unit Tests
- [ ] All tests pass
- [ ] Test coverage > 80%
- [ ] Code review passed (no critical/high issues)

### Integration Tests
- [ ] End-to-end flow completes
- [ ] Output matches expected format
- [ ] Error handling works correctly

---

## Implementation Progress

### ⏳ Phase 1: [Phase Name] (PENDING)
- **Status**: Not started
- **Next Steps**: [What to do first]
- **Review Status**: Pending

---

## Code Review Summary

| Phase | Critical | High | Medium | Low | Status |
|-------|----------|------|--------|-----|--------|
| 1     | 0        | 0    | 0      | 0   | ⏳     |
| 2     | 0        | 0    | 0      | 0   | ⏳     |
| 3     | 0        | 0    | 0      | 0   | ⏳     |

---

## Document Updates Log

| Date | Phase | Changes | Commit |
|------|-------|---------|--------|
| | | | |

## Overall Progress: 0% (0/N phases complete)
```

## Important Guidelines

1. **Discussion First**: NEVER skip Phase 1 - understanding the problem is more important than writing code
2. **Ask, Don't Assume**: If requirements are unclear, ask clarifying questions
3. **Be Specific**: Replace all `[placeholders]` with actual names, types, and descriptions
4. **TDD First**: Always define tests BEFORE implementation code
5. **Realistic Tests**: Include edge cases, error conditions, and happy paths
6. **Clear Architecture**: Show how components interact
7. **Actionable Steps**: Each step should be completable in one sitting
8. **Language Aware**: Always load and follow the appropriate language mockup file

## Output Location

- If `design/` or `design/steps/` folder exists: Write to `design/steps/step_[N].md`
- Otherwise: Output the plan directly in the response

---

## Code Review Integration

After completing each TDD phase (Red → Green → Refactor), you MUST:

### 1. Launch Code Reviewer
Invoke the `@code-reviewer` agent on the files changed in this phase.

### 2. Process Findings
The code-reviewer will return findings with severity levels:
- **🔴 Critical**: Security vulnerabilities, data loss risks, crashes
- **🟠 High**: Bugs, unhandled edge cases, race conditions
- **🟡 Medium**: Code quality, maintainability issues
- **🟢 Low**: Style, minor improvements

### 3. Ask User for Critical/High Findings
If there are **Critical** or **High** severity findings:
```
Code review found [N] critical/high issues:
1. [Issue summary]
2. [Issue summary]

Would you like to:
A) Fix these issues now (recommended)
B) Add to backlog and continue
C) Ignore and proceed
```

### 4. Update Step Document
Add all findings to the "Code Review Findings" table in `step_[N].md`:
- Record the finding
- Record severity
- Record status (Fixed/Deferred/Ignored)
- Record resolution notes

### 5. Loop Until Clean
For Critical issues: Must fix before proceeding to commit
For High issues: Strongly recommend fixing, but user can defer
For Medium/Low: Document and optionally address

---

Now generate the TDD plan for: $ARGUMENTS
