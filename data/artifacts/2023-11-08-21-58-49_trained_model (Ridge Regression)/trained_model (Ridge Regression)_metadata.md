
---
library_name: sklearn
tags:
- Amsterdam house pricing prediction
- tabular-regression
license: NA
datasets:
- data/data_train_processed.parquet
- data/data_val_processed.parquet
model: 
- 'Ridge regression'
metrics:
- 'r-squared'
features:
- ['Region_1', 'Location_1', 'Location_2', 'Location_3', 'Location_5', 'Location_6', 'Location_7', 'Location_8', 'Location_9', 'log(Area)', 'log(Room)', 'log(AR-ratio)', 'Manhattan_distance']
target:
- log(Price)
additional_information: 
- Dataset is bounded to samples with price < 2M
- The Dataset contains only houses from Amsterdam
- IMPORTANT: this model is not production ready
---
