# Food protein allergenicity and evolutionary divergence from human homologs

This is a prototype I built to test the central idea from my research proposal: that food proteins which have diverged further from their closest human homolog tend to be more allergenic. It pulls together my curated dataset, a simple interactive dashboard for exploring it, and a first machine-learning model.

## The idea

If a protein looks very different from anything in our own bodies, the immune system has more reason to treat it as "foreign." So I wanted to check whether lower sequence identity to a human homolog actually lines up with higher allergenicity across a real dataset.

## My dataset

I compiled 794 food proteins (535 plant, 259 animal) across 105 protein families, using AllergenOnline and the WHO/IUIS Allergen Nomenclature. For each one I ran BLAST against human proteins in UniProt to find the closest human homolog and record the percent identity and E-value. 245 of the 794 had a detectable human homolog; the rest had none, which is interesting in itself.

For the label, I treated a protein as an allergen if it had biological confirmation (basophil activation or a positive skin prick test), and as a non-allergen if only IgE binding was reported without that confirmation.

## What's in here

- `allergen_clean.csv` — my dataset, cleaned up for analysis.
- `train_model.py` — trains a Random Forest and an SVM, with 5-fold cross-validation.
- `allergen_model.pkl` — the saved model after I run the script.
- `allerevo.html` — a dashboard I made to browse the data and see the trend (opens in a browser).

## Running it

```
pip install pandas scikit-learn joblib
python train_model.py
```

## What I found so far

On the 245 proteins that have a human homolog, the Random Forest got around 76% accuracy with 5-fold cross-validation (the baseline from just guessing the majority class is about 60%). The most important feature was percent identity to the human homolog — which is what my hypothesis predicts.

Two things I'm still cautious about: protein length also came out as influential, and I want to check whether that's biological or just a quirk of my dataset. And the SVM did worse than the Random Forest, so it probably needs better feature scaling and the extra features I planned (epitope and physicochemical properties).

## What this is and isn't

This is an early prototype to demonstrate the direction, not a finished tool. It doesn't diagnose allergy and it isn't a replacement for AllergenOnline, IUIS, or established predictors. The dataset is modest and the label is "confirmed allergen vs IgE-binding-only," not a true negative control.

## Where I'd take it next

Add the epitope and physicochemical features from my proposal, tune the SVM, look properly at the 549 proteins with no human homolog (the most-diverged case), and bring in the phylogenetic distance measures.
