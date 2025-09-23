# Results Directory

All outputs, reports, and results should be saved in this directory.

## Naming Convention

**REQUIRED FORMAT**: `YYYY-MM-DD_HH-MM-SS_description.ext`

## Examples

- `2024-01-15_14-30-00_model_predictions.csv`
- `2024-01-15_09-45-30_analysis_report.pdf`
- `2024-01-15_16-20-15_confusion_matrix.png`
- `2024-01-15_11-00-00_experiment_results.json`

## File Types

- `.csv` - Tabular data results
- `.json` - Structured data outputs
- `.pdf` - Reports and documents
- `.png`/`.jpg` - Charts, plots, visualizations
- `.html` - Interactive reports
- `.pickle`/`.pkl` - Serialized Python objects

## Important Notes

- All results are gitignored by default
- Always use the timestamp format to prevent overwrites
- Group related results in subdirectories if needed (e.g., `results/experiment_001/`)
- Include metadata in the files when possible (parameters, version, etc.)