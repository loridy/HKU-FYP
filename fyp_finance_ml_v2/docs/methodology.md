# Methodology (Code Version)

This code version is designed around one principle:

**Models stay simple; feature families provide depth; financial metrics provide breadth.**

The pipeline runs in the following order:
1. load or generate panel data
2. engineer feature families
3. create 1D and 3D forward labels
4. split by time into train / validation / test
5. fit simple classifiers
6. tune classification threshold on validation only
7. evaluate:
   - ML metrics
   - signal metrics
   - portfolio metrics
   - benchmark-relative metrics
8. build a four-panel summary figure on one page
