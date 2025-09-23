# Logs Directory

All application logs should be saved in this directory.

## Naming Convention

Use the following format for log files:
- `YYYY-MM-DD_module_name.log` for daily logs
- `YYYY-MM-DD_HH-MM-SS_event.log` for specific events

## Examples

- `2024-01-15_data_processing.log`
- `2024-01-15_14-30-00_error_trace.log`
- `2024-01-15_api_requests.log`

## Important Notes

- All logs are gitignored by default
- Use structured logging (JSON format preferred)
- Rotate logs daily or when they exceed 100MB
- Never log sensitive information (passwords, tokens, PII)