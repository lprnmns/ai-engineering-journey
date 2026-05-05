# Titanic EDA Report

## Dataset Shapes

- Train shape: 891 rows, 12 columns
- Test shape: 418 rows, 11 columns
- Submission shape: 418 rows, 2 columns

## Target

Target column: Survived

Target distribution:

Survived
0    549
1    342

Survival rate: 38.38%

## Columns

Train columns:

PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

Numeric columns:

PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare

Categorical columns:

Name, Sex, Ticket, Cabin, Embarked

## Missing Values — Train

Cabin          687
Age            177
Embarked         2
PassengerId      0
Name             0
Pclass           0
Survived         0
Sex              0
Parch            0
SibSp            0
Fare             0
Ticket           0

## Missing Values — Test

Cabin          327
Age             86
Fare             1
Name             0
Pclass           0
PassengerId      0
Sex              0
Parch            0
SibSp            0
Ticket           0
Embarked         0

## Train/Test Feature Alignment

Columns present in train features but missing in test:

[]

Columns present in test but not train features:

[]

## Initial Competition Thinking

1. Survived is the target and exists only in train.csv.
2. test.csv does not contain Survived, so it must be predicted.
3. PassengerId is required for submission but should not be treated as a meaningful predictive signal by default.
4. Cabin has many missing values and needs careful handling.
5. Age has missing values and will need imputation.
6. Embarked has a small number of missing values in train.
7. Fare has missing values in test.
8. Sex, Pclass, Age, Fare, SibSp, Parch, Embarked are likely useful baseline features.
9. Name, Ticket, Cabin may contain useful engineered features later.
10. First baseline should be simple and reproducible before advanced feature engineering.

## Next Step

Build a preprocessing pipeline:

- impute missing Age
- impute missing Fare
- impute missing Embarked
- encode categorical columns
- create first baseline model
