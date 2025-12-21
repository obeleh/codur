# Trial-Error Loop

A coding agent can run a CLI command and observe the result. If the command returns an error, the agent should treat that error as part of the task to solve.

Workflow:
1. Run the CLI command.
2. If it fails, inspect the error output.
3. Make targeted file edits to resolve the error.
4. Run the same CLI command again to verify the fix.
5. Repeat this loop until the command succeeds.

Hints for effective iteration:
- If you keep seeing the same error, add focused logging or assertions around the failing path to surface more context.
- Reduce the scope: isolate the smallest input or code path that reproduces the error.
- Change one thing at a time so you can attribute improvements or regressions.
- Revert speculative changes quickly if they don’t move the error.
- Keep notes on attempted fixes and outcomes to avoid cycling.
