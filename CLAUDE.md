# CLAUDE.md

## Test Generation Policy
Whenever you create or modify a feature, bugfix, or any functional code:
1. Generate corresponding tests in the `backend/tests/` directory
2. Tests should cover happy path, edge cases, and failure cases
3. For NBA model code specifically, include tests for:
   - Data validation (player stats, game data, odds)
   - Model prediction accuracy assertions
   - API response structure
4. Run the tests after generating them and fix any failures before finishing
5. Do not consider a task complete until tests pass