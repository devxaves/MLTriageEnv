# Implementation Task List

This file tracks the work needed to make the OpenEnv submission compliant.

## Status

- [x] Audit spec gaps
- [x] Fix OpenEnv API contract
- [x] Harden baseline inference
- [x] Align packaging and deployment
- [x] Refresh documentation
- [x] Validate and fix tests
- [x] Finalize task list file

## Notes

- The environment must expose the standard `step()` / `reset()` / `state()` API.
- The baseline must be reproducible, structured, and use the OpenAI client.
- The repo must run cleanly in Docker and on Hugging Face Spaces.
