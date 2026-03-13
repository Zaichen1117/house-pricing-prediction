
---
library_name: sklearn
tags:
- Dutch house price prediction
- tabular-regression
license: NA
datasets:
- data/data_train_processed.parquet
- data/data_val_processed.parquet
model: 
- 'ridge regression'
metrics:
- 'r-squared'
features:
- ['log(Area)', 'log(Room)', 'log(AR-ratio)', 'Manhattan_distance', 'Region', 'Location']
target:
- log(Price)
additional_information: 
- Dataset is bounded to samples with price < 2M
- The Dataset contains only houses from Amsterdam
- IMPORTANT: this model is not production ready
---
