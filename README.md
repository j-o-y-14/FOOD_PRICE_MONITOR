#  Global Food Price Monitor (GFPM)

**Tagline:** *Tracking the World’s Food Costs Through the Lens of Inflation.*

---
🔗 Links



Website: https://food-forecast-nexus.lovable.app

Full Article: https://food-price-monitor.hashnode.dev/global-food-price-monitor-predicting-food-prices-with-data-science

Streamlit : https://foodpricemonitor.streamlit.app/
##  Overview

The **Global Food Price Monitor (GFPM)** is a data-driven analytics system designed to **track, analyze, and forecast global staple food prices** in relation to **inflation trends**.  
Rising food costs and global inflation have become major concerns for households, policymakers, and international organizations.  
This project aims to bridge that gap by providing an accessible platform that visualizes how inflation influences food affordability worldwide.

---

##  Problem Statement

Food prices around the world have been rising unevenly due to inflation, global conflicts, supply chain disruptions, and climate events.  
However, there is **no centralized system** that continuously monitors and analyzes how inflation affects staple food prices across countries and time periods.  

**Goal:**  
Build a system that collects, cleans, analyzes, and visualizes the relationship between **inflation** and **food price trends**, while providing **predictive insights** for the future.

---

##  Research Questions

1. How have global staple food prices (wheat, rice, maize, oils, sugar) evolved over the past decade?  
2. What is the relationship between inflation rates and food price changes across countries?  
3. Which regions are most vulnerable to inflation-driven food price spikes?  
4. How do global shocks (e.g., COVID-19, Ukraine war) affect food prices relative to inflation?  
5. Can future food prices be predicted based on inflation and historical data?

---

##  System Design

### **1. Data Collection**
- Import global datasets from FAO, IMF, and World Bank.
- Collect historical inflation rates, CPI, and staple food price indexes.  
- Tools: `Python`, `pandas`, `requests`

---

### **2. Data Cleaning & Preprocessing**
- Handle missing values and inconsistent units.
- Normalize currencies and merge datasets by country and time period.  
- Convert to a time-series structure.  
- Tools: `pandas`, `numpy`

---

### **3. Exploratory Data Analysis (EDA)**
Performed detailed exploratory analysis to uncover relationships between inflation and food prices.

- **Heatmap:** Checked multicollinearity between inflation, CPI, and food price variables.  
- **Histograms:** Examined distribution of food prices, inflation rates, and consumer indices.  
- **Boxplots:** Identified outliers and abnormal spikes in price or inflation data.  
- **Pairplots:** Explored linearity and correlation between economic indicators.  
- **Time Series Decomposition:** Extracted trend, seasonality, and residual components for each staple food category and region.  


Tools: `matplotlib`, `seaborn`, `Tableau`

---

### **4. Predictive Modeling**

#### 🔹 Time Series Models
- **ARIMA / SARIMA / Prophet** for price forecasting.
- Predict future food prices based on inflation trends.

#### 🔹 Machine Learning Models
- **Linear Regression, XGBoost, Random Forest** to predict food prices using inflation, GDP, and other economic factors.



---

###  **Deep Learning Models**

To enhance forecasting accuracy, deep learning models were incorporated to capture complex, non-linear relationships between inflation, food prices, and external economic factors.

#### **1. LSTM (Long Short-Term Memory)**
- Captures sequential dependencies in food price and inflation data.
- Learns long-term temporal patterns.
- Frameworks: `TensorFlow`, `Keras`.



#### **Evaluation Metrics**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score

#### **Model Comparison**
| Model | Type | Strength |
|--------|------|-----------|
| ARIMA | Statistical | Simple, interpretable |
| Prophet | Time Series | Seasonality, trend analysis |
| Random Forest | Machine Learning | Non-linear relationships |
| LSTM | Deep Learning | Sequence learning |


---

### **5. Visualization & Dashboard**
- Interactive dashboard for exploring trends and forecasts.
- Visuals: Line charts, heatmaps, global maps, and correlation plots.  


---

### **6. Reporting & Insights**
- Auto-generate insights and recommendations based on trends.
- Export results as reports or presentations.  
- Tools: `Python`, `ReportLab`, `PowerPoint`

---

##  System Architecture
Data Sources (FAO, IMF, World Bank)

↓

Data Collection (Python)

↓

Data Cleaning & Preprocessing

↓

Exploratory Data Analysis (EDA)

↓

Modeling (Time Series + ML + Deep Learning)

↓

Visualization (Tableau)

↓

Deployment(Streamlit)

##  Target Audience

The **Global Food Price Monitor (GFPM)** serves a variety of stakeholders who rely on accurate, timely, and actionable food price insights:

1. **Policymakers & Government Agencies**  
   - **Who:** Ministries of Agriculture, Finance, Trade; Central Banks; National Statistics Bureaus  
   - **Why:** Food prices affect inflation, subsidies, and social protection policies. Early warnings help design interventions like tariff adjustments, food aid, and strategic reserves.  
   - **Needs:** Clear dashboards and forecasts to guide policy and budget decisions.

2. **International Organizations & NGOs**  
   - **Who:** FAO, WFP, World Bank, IMF, UN agencies, humanitarian NGOs  
   - **Why:** Price monitoring is critical for identifying crisis zones and managing food aid, market interventions, and funding decisions.  
   - **Needs:** Comparative global and regional trends, inflation-adjusted insights, and anomaly alerts.

3. **Agribusinesses & Supply Chain Stakeholders**  
   - **Who:** Food importers/exporters, wholesalers, retailers, logistics companies  
   - **Why:** Price trends influence procurement, inventory, and profit margins. Forecasting aids in cost anticipation, contract negotiation, and logistics optimization.  
   - **Needs:** Commodity-specific short- and medium-term forecasts and market signals.

4. **Financial & Economic Analysts**  
   - **Who:** Banks, investment firms, economists, research institutions  
   - **Why:** Food prices affect CPI, interest rates, and investment decisions. Analysts monitor commodities to anticipate market risks and macroeconomic shifts.  
   - **Needs:** Historical and forecast data to build economic models, investment reports, or inflation outlooks.

5. **Researchers & Data Scientists**  
   - **Who:** Academics, students, open-data enthusiasts, policy researchers  
   - **Why:** Global food price datasets allow modeling seasonality, volatility, and causal relationships; useful for predictive modeling and policy testing.  
   - **Needs:** Clean, structured data; reproducible pipelines; open APIs or notebooks.




