# Machine Learning Workflow: A regression Case Study

**Project:** [Amsterdam House Price Prediction](https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction) 

![project-img](/photo_amsterdam.jpeg)
Photo by <a href=" https://unsplash.com/@adrienolichon">Adrien Olichon</a> on <a href="https://unsplash.com/photos/QRtym77B6xk">Unsplash</a>


**Objective:** Predict House Pricing and Illustrate Machine Learning Workflow

**Framework:** 
- The problem is framed as a regression problem since the target has continuous numerical values.
- I used a supervised learning approach 
- The model will be trained offline in one batch
- The primary performance metric will be r2, combined with Root Mean Square Error (RSME) and Maximum Absolute Error (MAE) 
- We aim for a model with an r2 of at least 0.7.

**How would we solve the problem manually:**
- One can derive the price of a given house by comparing the price with other similar houses, e.g., on [pararius.nl](https://www.pararius.nl).

**List assumptions coming from research questions:**
- the bigger the area of a house, the higher the cost of a house
- the bigger the number of rooms, the higher the cost of a house
- a closer location to the city centre would lead to higher housing costs.
