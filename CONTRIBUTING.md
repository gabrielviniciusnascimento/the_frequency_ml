# Contributing to The Frequency ML

## How to contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Code style

- Python 3.13+
- Follow PEP 8
- Add docstrings to all functions
- Include logging (not print statements)
- Add checkpointing to new scripts (skip if output exists)

## Data

- Do not commit raw NHANES data (too large, publicly available)
- Do not commit personal data
- OHHR data is CC BY 4.0 — cite correctly

## Reporting issues

Open an issue with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
