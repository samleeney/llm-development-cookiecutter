# PROJECT GOAL

<!-- PERMANENT INSTRUCTIONS - DO NOT REMOVE THIS SECTION -->
## How to Use This Document

This is the north star document for your project. It defines:
- What you're building and why
- Who will use it and how
- What success looks like
- Technical constraints and choices

Update this document when project scope changes, but keep it focused and concise. This should be the first document anyone reads to understand your project.

---

<!-- EXAMPLE CONTENT - REPLACE EVERYTHING BELOW WITH YOUR PROJECT SPECIFICS -->

## Overview
**Project**: DataViz Dashboard
**Purpose**: Interactive web dashboard for real-time data visualization

## Problem
Analysts spend hours manually creating reports from CSV files, with no ability to explore data interactively or share insights easily.

## Solution
A web-based dashboard that auto-imports CSV files, provides interactive charts, and allows one-click sharing of filtered views.

## Users
- Primary: Business analysts in small to medium companies
- Secondary: Data scientists needing quick exploratory tools

## Key Features
1. Drag-and-drop CSV import with automatic type detection
2. Interactive charts with zoom, filter, and drill-down
3. Shareable dashboard links with preserved filters
4. Export to PDF/PNG for presentations

## Success Criteria
- [ ] Process 100MB CSV files in under 5 seconds
- [ ] Support 10+ simultaneous users
- [ ] 90% of users can create first chart within 2 minutes
- [ ] Zero data loss during 30-day testing period

## Technical Stack
- Language: Python 3.11
- Framework: FastAPI + React
- Key libraries: Pandas, Plotly, PostgreSQL
- Deployment: Docker on AWS EC2

## Timeline
- Week 1-2: Core data ingestion
- Week 3-4: Basic visualization
- Week 5-6: Interactivity and sharing
- Week 7-8: Testing and deployment