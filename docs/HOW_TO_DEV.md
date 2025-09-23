# HOW TO DEV

This is the standard development workflow. Follow these steps for every project.

## Initial Setup
1. Fill in PROJECT_GOAL.md
2. Fill in PLAN.md with features
3. Brainstorm with AI to refine
4. Update README.md

## Development Cycle
For each feature:

1. **Plan**: Review ARCHITECTURE.md, plan approach
2. **Build**: Write feature and tests together
3. **Test**: Run pytest continuously
4. **Document**: Update PLAN.md, ARCHITECTURE.md, TESTS.md
5. **Track**: Create build_status file

## AI Commands
```
"Start working on [feature] from PLAN.md"
"Review my code against CONVENTIONS.md"
"Help debug this error: [error]"
"Update documentation for completed feature"
```

## Before Marking Done
- [ ] Tests pass
- [ ] Docs updated
- [ ] PLAN.md updated
- [ ] Build status created