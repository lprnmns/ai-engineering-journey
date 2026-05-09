# Titanic Deck + Ticket + Fare Report

## Hypothesis

The current best model may be missing Titanic-specific group and location signals.

This experiment adds:

```text
Deck = first letter of Cabin
TicketGroupSize = number of passengers sharing the same Ticket
FarePerPerson = Fare / TicketGroupSize
```

## Why These Features?

### Deck

Cabin location may proxy ship location and passenger class.

### TicketGroupSize

Passengers sharing a ticket may have travelled together.

### FarePerPerson

Fare can represent a group ticket price, so dividing by ticket group size may create a cleaner economic signal.

## Model

- Model: HistGradientBoostingClassifier
- Feature set: plus_title + Deck + TicketGroupSize + FarePerPerson
- Metric: accuracy
- Validation: 5-fold StratifiedKFold

## Results

- Fold 1: 0.86034
- Fold 2: 0.84831
- Fold 3: 0.80337
- Fold 4: 0.83146
- Fold 5: 0.83708

- CV mean: 0.83611
- CV std: 0.01913

## Baseline Reference

exp_003 plus_title hgb: CV 0.83725, public LB 0.77272

## Submission

- Path: `competitions/titanic/submissions/deck_ticket_fare_hgb_submission.csv`

## Competition Interpretation

This experiment checks whether domain-specific Titanic features move us closer to the estimated 12 additional correct predictions needed for 0.80 public LB.

Important:

TicketGroupSize is computed from train+test features only, not target labels. This is targetless feature engineering, not label leakage.
