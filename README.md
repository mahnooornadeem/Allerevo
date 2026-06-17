# Evolutionary Divergence & Food Protein Allergenicity

A prototype linking **evolutionary distance from human homologs** to **food protein allergenicity**, using a curated dataset, an interactive dashboard, and a machine-learning classifier.

Based on the research proposal *"Role of evolutionary divergence from human homologs in food protein allergenicity using bioinformatics and machine learning."*

---

## Hypothesis

Proteins that have diverged further from their human homologs (lower sequence identity) tend to be more allergenic — a more "foreign" sequence draws a stronger immune response.

## Dataset

- **794 food proteins** (535 plant, 259 animal) across **105 protein families**.
- Compiled from **AllergenOnline (FARRP)** and the **WHO/IUIS Allergen Nomenclature**.
- Human homologs identified by **BLAST** against **UniProt**; **245** proteins had a detectable human homolog.
- **Label:** "allergen" = confirmed by biological test (basophil release / SPT positive); "non-allergen" = IgE-binding reported only. (This matches the `S_AR` annotation.)

## Files

| File | What it is |
|------|------------|
| `allergen_clean.csv` | The cleaned 794-protein dataset (model input). |
| `train_model.py` | Trains Random Forest + SVM with 5-fold cross-validation. |
| `allergen_model.pkl` | The trained Random Forest, saved after running the script. |
| `allerevo.html` | Self-contained interactive dashboard (open in any browser). |

## How to run the model

```bash
pip install pandas scikit-learn joblib
python train_model.py
```

## Results (5-fold cross-validation)

On the 245 proteins with an identifiable human homolog:

- **Random Forest: ~76% accuracy** (majority-class baseline ~60%).
- **Percent identity to the human homolog is the most important feature** — direct support for the divergence hypothesis.

## Honest limitations

- Early-stage **prototype**, not a clinical or diagnostic tool, and not a replacement for AllergenOnline / IUIS / validated predictors.
- The label distinguishes biologically-confirmed allergens from IgE-binding-only proteins, not allergen vs. true negative control.
- Modest dataset; protein length is also influential and needs checking for confounding.
- Next steps: add epitope and physicochemical features, tune the SVM, and investigate the 549 proteins with no detectable human homolog (the most-diverged case).

## Data sources

AllergenOnline (FARRP) · WHO/IUIS Allergen Nomenclature · UniProt · BLAST · Pfam
