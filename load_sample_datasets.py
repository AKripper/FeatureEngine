from pathlib import Path

BASE_DIR = Path(__file__).parent

SAMPLE_DATASETS = {
    "Titanic": BASE_DIR / "sample_datasets" / "titanic.csv",
    "Housing": BASE_DIR / "sample_datasets" / "housing.csv",
}