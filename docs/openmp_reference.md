# OpenMP C Coding Style Guide

## 1. General Formatting

- Indentation: 4 spaces
- No tabs
- K&R brace style
- Maximum line length: 100 characters

## 2. Naming Conventions

| Element        | Convention        |
|---------------|-------------------|
| File names    | omp_\<topic\>.c     |
| Functions     | snake_case()      |
| Variables     | snake_case        |
| Macros        | UPPER_SNAKE_CASE  |

## 3. OpenMP-Specific Rules

- Prefer explicit data scoping
- Use `default(none)` in non-introductory examples
- Always document whether output order is deterministic
- Never rely on implicit synchronization without explaining it

## 4. Compilation Hygiene

- Always compile with warnings enabled
- Treat warnings as correctness indicators
- Avoid undefined or implementation-dependent behavior unless explicitly discussed

## 5. Pedagogical Priority

Code clarity and conceptual correctness take precedence over micro-optimizations.
