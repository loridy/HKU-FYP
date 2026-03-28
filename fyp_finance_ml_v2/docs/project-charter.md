# Project Charter

## Core research question
Which feature families are useful for stock selection once they are judged not only by ML classification metrics, but also by signal-quality and portfolio-performance metrics?

## Safe completion rules
- keep horizons to 1D and 3D
- keep models to logistic regression, random forest, and one optional booster
- keep portfolio logic simple: top-K equal-weight
- use time-aware splits only
- make every stage export CSV tables and figures
