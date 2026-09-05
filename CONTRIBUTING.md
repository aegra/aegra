# Contributing to Aegra

Thank you for your interest in contributing to Aegra! This guide will help you get started.

## 🚀 Quick Start

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/aegra.git
cd aegra
```

### 2. Set Up Development Environment

**Option 1: Using Make (Recommended)**
```bash
make dev-install     # Installs dependencies + git hooks
```

**Option 2: Using uv directly**
```bash
uv sync --all-packages
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

Git hooks will run before every commit and check:
- ✅ Code formatting (Ruff)
- ✅ Linting issues
- ✅ Type checking (mypy)
- ✅ Commit message format
- ✅ Security issues (Bandit)

## 📝 Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/) for clear and structured commit history.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, build, etc.)
- `ci`: CI/CD changes
- `build`: Build system changes

### Scope (Optional)

Use scope to specify which part of the codebase is affected:
- `api`, `auth`, `graph`, `db`, `middleware`, `tests`, `docs`, `ci`

### Examples

```bash
# Good commits ✅
git commit -m "feat: add user authentication endpoint"
git commit -m "feat(api): implement rate limiting for threads"
git commit -m "fix: resolve database connection pool exhaustion"
git commit -m "fix(auth): handle expired JWT tokens correctly"
git commit -m "docs: update API documentation for assistants"
git commit -m "refactor(graph): simplify state management logic"
git commit -m "test(e2e): add integration tests for streaming"
git commit -m "chore: upgrade langchain to v0.3.1"

# Bad commits ❌
git commit -m "fixed stuff"
git commit -m "WIP"
git commit -m "Updated files"
git commit -m "Fix bug"
```

### Breaking Changes

If your change breaks backward compatibility, add `BREAKING CHANGE:` in the footer:

```bash
git commit -m "feat(api): redesign authentication flow

BREAKING CHANGE: The authentication endpoint now requires a different payload structure."
```

## 🔍 Code Quality Standards

### Before Committing

Run these commands to ensure your code meets our standards:

```bash
# Format and fix auto-fixable issues
make format

# Check for remaining issues
make lint

# Run type checking
make type-check

# Run tests
make test

# Or run all checks at once
make ci-check
```

### Pre-commit Hooks

The pre-commit hooks will automatically run when you commit. If they fail:

1. **Review the errors** - The hooks will show what needs to be fixed
2. **Fix the issues** - Most formatting issues are auto-fixed
3. **Stage the changes** - `git add .`
4. **Commit again** - `git commit -m "your message"`

### Bypassing Hooks (Not Recommended)

Only in emergencies:
```bash
git commit --no-verify -m "emergency fix"
```

⚠️ **Warning**: CI will still check your code, so bypassing hooks locally doesn't help!

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
uv run --package aegra-api pytest libs/aegra-api/tests/e2e/test_assistants/test_assistant_graph.py

# Run specific test
uv run --package aegra-api pytest libs/aegra-api/tests/e2e/test_assistants/test_assistant_graph.py::test_create_assistant
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Place end-to-end tests in `tests/e2e/`
- Use descriptive test names: `test_should_return_error_when_invalid_input`
- Aim for 80%+ code coverage

## 🔌 API Compatibility

Aegra is a drop-in replacement for LangSmith Deployments. Every request field the LangGraph SDK can send must either change behavior or return `422`. Never accept a field and ignore it.

- Declare every SDK field on the request model, even ones Aegra does not implement yet; reject unsupported values with a clear error.
- Extend the drift tests when you add a route or field, so a new SDK field fails CI instead of being dropped.
- The only fields allowed to be inert are ones that configure a system outside Aegra (LangSmith-side routing). List each in `docs/feature-support.mdx` with the reason.
- Follow the public LangGraph Platform names and defaults for config keys, env vars and enum values.

## 🔒 Security

- Never commit secrets, API keys, or credentials
- Use environment variables for sensitive data
- Run `make security` to check for security issues
- Report security vulnerabilities privately to the maintainers

## 📋 Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Write clean, readable code
- Follow existing code style
- Add tests for new features
- Update documentation if needed

### 3. Commit Your Changes

```bash
git add .
git commit -m "feat: add amazing feature"
```

The pre-commit hooks will run automatically.

### 4. Push to Your Fork

```bash
git push origin feat/your-feature-name
```

### 5. Open a Pull Request

- Use a clear, descriptive title following Conventional Commits format
- Describe what changes you made and why
- Reference any related issues
- Ensure all CI checks pass

### PR Title Format

Your PR title must follow Conventional Commits:

```
feat: add user profile endpoint
fix(auth): resolve token expiration bug
docs: update installation guide
```

### CI Checks

Your PR must pass:
- ✅ Code formatting (Ruff)
- ✅ Linting (Ruff)
- ✅ Type checking (mypy)
- ✅ Security checks (Bandit)
- ✅ Tests (pytest)
- ✅ Conventional Commits validation

## 🛠️ Development Workflow

### Daily Development

```bash
# 1. Pull latest changes
git pull origin main

# 2. Create feature branch
git checkout -b feat/my-feature

# 3. Make changes and test
make format
make test

# 4. Commit (hooks run automatically)
git commit -m "feat: add my feature"

# 5. Push
git push origin feat/my-feature

# 6. Open PR on GitHub
```

### Keeping Your Fork Updated

```bash
# Add upstream remote (once)
git remote add upstream https://github.com/aegra/aegra.git

# Update your fork
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## 📚 Code Style Guidelines

### Python

- Follow PEP 8 (enforced by Ruff)
- Use type hints for all functions
- Keep functions small and focused
- Use descriptive variable names
- Avoid comments unless necessary (code should be self-documenting)

### Documentation

- Update README.md for user-facing changes
- Add docstrings for public APIs
- Use Google-style docstrings
- Keep documentation up-to-date

## 🐛 Reporting Bugs

### Before Reporting

1. Check existing issues
2. Try the latest version
3. Reproduce the bug consistently

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. ...
2. ...

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11]
- Aegra version: [e.g., 0.1.0]

**Additional context**
Any other relevant information.
```

## 💡 Feature Requests

We welcome feature requests! Please:
1. Check if it's already requested
2. Describe the use case
3. Explain why it would be valuable
4. Consider contributing the implementation

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards others

## 📞 Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Discord**: [Join our community] (if available)

## 🎉 Recognition

Contributors will be:
- Listed in our README
- Mentioned in release notes
- Part of our growing community

Thank you for contributing to Aegra! 🚀
