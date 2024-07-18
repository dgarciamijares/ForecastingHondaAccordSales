# Honda Accord Sales Forecasting

This project aims to predict the monthly sales of the Honda Accord in the United States using linear regression and economic indicators.

## Project Overview

The data spans from January 2014 to November 2023 and includes various economic indicators and Google search query volumes.

## Objectives

- Accurately forecast future sales to help Honda align production with customer demand.
- Analyze the impact of various economic indicators on sales performance.
- Explore the influence of seasonality on sales figures.

## Methodology

1. **Data Preparation:**
   - Split data into training (2014-2018) and testing (2019-2023) sets.
   - Consider key variables: Unemployment, AccordQueries, CPIAll, CPIEnergy, MilesTraveled.

2. **Model Development:**
   - Initial linear regression model with all five independent variables.
   - Variable selection based on significance, VIF, and model performance metrics.

3. **Seasonality Analysis:**
   - Incorporated MonthFactor variable to capture monthly variations in sales.
   - Final model includes MonthFactor and significant economic indicators.

4. **Additional Features:**
   - Added Consumer Confidence Index (CCI) to test its predictive value.

## Results

- Best-performing model explained ~74.8% of variance in the training data.
- Identified significant monthly variations in sales.
- Overfitting observed in testing set performance.

## Technologies Used

- Python (Pandas, Statsmodels, Matplotlib)
- Jupyter Notebook

## Conclusion

The project demonstrates the application of linear regression in forecasting sales, highlighting the importance of considering both economic indicators and seasonal trends.

## Visualizations

[Include links or images of your visualizations]

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/honda-accord-sales-forecasting.git
