# Contributing to Deep Research Agent

Thank you for your interest in contributing to the Deep Research Agent project!

## How to Contribute

### Reporting Issues

- Check if the issue already exists
- Use the issue template when creating new issues
- Include reproduction steps and environment details

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with clear commit messages
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Research_Agent.git
cd Research_Agent

# Install dependencies
pip install -r requirements.txt
cd frontend && pnpm install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run tests
pytest

# Start development servers
python server.py  # Backend (port 8000)
cd frontend && pnpm dev  # Frontend (port 3000)
```

### Code Style

- **Python**: Follow PEP 8 guidelines
- **TypeScript**: Use ESLint configuration
- **Commits**: Use conventional commit messages
  - `feat: add new feature`
  - `fix: resolve bug`
  - `docs: update documentation`
  - `refactor: improve code structure`

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Test both backend and frontend changes

### Documentation

- Update README.md for major changes
- Add docstrings to Python functions
- Comment complex logic
- Update API documentation

## Areas for Contribution

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations
- 🧪 Test coverage

## Questions?

Feel free to open an issue for discussion before starting major work.

---

Thank you for contributing! 🎉
