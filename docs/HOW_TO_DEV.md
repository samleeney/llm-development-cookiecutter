# HOW TO DEV

This is the standard development workflow. Follow these steps for every project.

## Initial Setup
1. Fill in the "FILL IN HERE" section in PROJECT_GOAL.md with your project description
2. Fill in the "FILL IN HERE" section in PLAN.md with your features list
3. Ask AI to "merge and optimize PROJECT_GOAL.md and PLAN.md based on my input" - AI will structure your content properly
4. Review and refine the merged content with AI
5. Update README.md with final project description

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