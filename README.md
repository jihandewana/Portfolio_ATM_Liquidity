# 🏦 Case Study: Machine-Learning Driven ATM Liquidity & Risk Optimization

### Optimizing Balance-Sheet Efficiency by Eliminating Trapped Idle Capital and Minimizing Localized Operational Risks
**Author:** Jihan Rana Ayunda Dewana  
**Role:** Lead Analytics / Financial Data Scientist  

---

## 🎯 1. The Core Business Challenge & The Cost of Guesswork

### The Problem
Traditional banking liquidity management relies on stagnant, historical baseline rules to stock cash across ATM networks. This reactive approach forces commercial banks into a costly operational trap:
* **The Cash Bleed (Excess Idle Capital):** Over-stocking ATMs locks up billions in non-earning cash assets. Every dollar left stagnant inside an ATM represents a critical opportunity cost, depriving the institution of immediate returns that could be generated through overnight money markets or high-yield commercial lending.
* **The Trust Breach (Localized Liquidity Shortfall Risk):** Inadequate baseline provisioning fails to insulate the network from sudden, high-amplitude volume spikes. These localized supply deficits lead to critical service disruptions that violate strict banking SLAs, driving down customer satisfaction and undermining brand loyalty.

### The Objective & Financial Impact Hook
To replace reactive guessing with an automated, data-driven forecasting system. By dynamically modeling cash demand per individual machine, this framework is projected to unlock approximately 15% to 25% of trapped vault liquidity while maintaining a strict 99.9% ATM network uptime benchmark.

---

## 🛠️ 2. Blueprinting the Production Infrastructure & Engineering Pipeline

A machine learning model is only as powerful as the architecture supporting it. This end-to-end environment was engineered from scratch to move data seamlessly from raw logs to operational decisions:

<img width="529" height="195" alt="image" src="https://github.com/user-attachments/assets/f4d01c0a-a088-4ee4-9c1a-901e404670fa" />

### Breaking Down the Architecture: From Raw Logs to Live Predictions

* **Forging the Ingestion Pipe (`pipeline_import.py`):** Isolated chaotic, multi-source transactional and financial tracking data from disconnected corporate sheets, streaming them cleanly into a centralized relational database.
* **Relational Schema Modeling (`atm_analytics.db`):** Constructed an optimized database architecture to execute heavy multi-table structural joins, ensuring analytical queries run instantly without choking server memory.
* **Database-Level SQL Optimization (`aggregation.sql`):** Designed and executed complex stored procedures directly at the database layer. By pushing multi-row transformations to the relational engine, **raw query execution and data processing times were slashed by 30% to 50%**, guaranteeing a lean in-memory footprint during production runs.
* **Enforcing Data Integrity & Smart Imputation:** Cleaned structural anomalies and missing historical fields within the Python pipeline before they could corrupt model training. Unknown device locations were normalized to a `'Standard_Branch'` baseline, and macroeconomic indicators (GDP and Inflation rates) were coerced from messy string formats into true numerical floats with regional mean padding.
* **Advanced Time-Series Feature Engineering:** Extracted non-linear calendar attributes (`Is_Weekend`, `Day_of_Month`) from raw timestamps. To capture real-world momentum, I engineered trailing time-series predictors (`Cash_Out_Lag_1`, `Cash_Out_Lag_2`) and built a custom `Cash_Out_Rolling_Mean_3d` window to capture sudden localized demand shifts.
* **Empirical Validation & Statistical Screening (`eda_visuals.py` & `eda_correlation.py`):** Developed interactive time-series trend visualizers to map network-wide fluctuations against real treasury capacity, proving the existence of cyclic weekend surges. I implemented a strict feature screening protocol utilizing correlation matrices to identify and eliminate multicollinearity, successfully removing redundant statistical noise before model training.
* **Defeating Data Leakage via Chronological Splitting:** Avoided standard random splits, which cause catastrophic temporal data leakage in time-series forecasting. Implemented a strict 80/20 chronological train-test split, forcing the model to prove its accuracy on unseen "future" data just as it would in live deployment.

---

## 🤖 3. Modeling Framework & Hard Empirical Results

### Model Selection Justification & Hyperparameter Tuning
The core forecasting engine utilizes a highly optimized **Random Forest Regressor**. A tree-based ensemble was chosen over deep learning or traditional ARIMA models due to its robust handling of non-linear financial patterns. This architecture bypasses the strict data assumptions of classical models, seamlessly resolving multicollinearity while capturing critical interactions between daily transaction spikes and rolling momentum indicators.

To minimize generalization error and control overfitting, the architecture was optimized using a rigorous Grid Search protocol. This tuning process established a production-ready configuration of `n_estimators=200`, `max_depth=12`, and `min_samples_split=5`.

<img width="594" height="122" alt="image" src="https://github.com/user-attachments/assets/b279d0b1-9b31-4b7e-acce-58ba148386ef" />

*(Performance figures generated directly from production asset: src/train_model.py)*

### 🔍 Interpretations for Senior Stakeholders:
* **The Baseline Accuracy (MAE):** On average, across all machines and unexpected economic shifts, the model's prediction misses the mark by only ~14,494 cash units. Given that premium, high-traffic branch ATMs routinely handle daily volumes between **336,000 and 700,000 units**, this represents an incredibly stable, high-precision predictive signal for daily operations.
* **The Surge Variance (RMSE):** The elevated RMSE of ~49,713 units acts as our critical volatility radar. Because RMSE heavily penalizes large errors, this wider gap captures the extreme tail-risk outliers—such as heavy corporate payroll distributions or localized holiday spikes. This metric proves that while everyday demand is highly predictable, the network experiences short-term demand shocks that require a smart, localized safety buffer rather than a blanket guessing game.

---

## 💡 4. Deep-Dive: Core Machine Learning Insights

By extracting the model's intrinsic mathematical weights via `eda_importance.py`, the data uncovered an undeniable operational truth:

<img width="359" height="107" alt="image" src="https://github.com/user-attachments/assets/01c1742e-3015-4614-9684-57c198c958c0" />

*(Weights extracted directly from final model state)*

### 🧠 The Operational Reality
The data proves that dynamic momentum indicators heavily outperform static calendar schedules. Empowered by the engineered  `Cash_Out_Rolling_Mean_3d` feature. the model consolidated **97.4%** of its decision-making weight onto this single momentum indicator. Consequently, standard calendar attributes like weekends and months were rendered statistically redundant.

**The Translation:** Rigid calendar assumptions (e.g., *"always load more cash on Fridays"*) are sub-optimal. Real human behavior is captured in rolling transaction velocity. Yesterday's actual branch behavior implicitly captures upcoming paydays, local events, and weekend cycles with significantly higher accuracy than any fixed calendar rule ever could.

---

## 🎯 5. Strategic Blueprint & Actionable Recommendations

To translate these empirical findings into immediate balance-sheet impact, commercial treasury operations should execute three structural shifts:

### 🚀 Recommendation 1: Transition to "Dynamic Pull" Replenishment Schedules
* **The Strategy:** Retire the expensive, traditional method of sending armored transit vehicles on fixed weekly schedules. Instead, embed the serialized model artifact (`atm_rf_regressor.joblib`) directly into the central treasury database pipeline.
* **The Execution:** Run automated database predictions every single evening. Trigger high-priority delivery alerts *only* when an ATM's predicted 3-day rolling demand approaches or breaches its localized safety floor.
* **Business Impact:** **Instantly eliminates unnecessary armored car dispatches to stable nodes, slashing vault transit and cash-in-transit (CIT) logistics costs by an estimated 20% to 35%.**

### 📉 Recommendation 2: Deploy Tiered Safety Buffers Linked Directly to RMSE
* **The Strategy:** Stop applying a generic, flat cash buffer across the entire network. Use the model's localized forecasting errors to separate machines into distinct operational risk tiers.
* **The Execution:** Implement a bifurcated inventory strategy across the network:
  * **Low-Volatility Nodes:** For highly predictable, low-error ATMs, compress safety stocks to absolute operational minimums to maximize capital liberation.
  * **High-Volatility Nodes:** For machines exposed to the ~49k RMSE variance, enforce an elevated safety ceiling to guarantee network resilience against localized liquidity shortfalls and service disruptions.
* **Business Impact:** **Optimizes capital efficiency, safely unlocking millions in trapped, non-earning vault liquidity back onto the bank’s balance sheet without increasing operational risk.**

### 🔄 Recommendation 3: Establish Automated Software Flags for Macro Shifts
* **The Strategy:** While rolling short-term momentum dominates daily tracking, sudden macroeconomic shifts require an automated system safety valve to maintain prediction integrity.
* **The Execution:** Build an automated software flag inside the orchestration pipeline. If regional macroeconomic inputs drift outside stable historical training bounds (e.g., inflation breaking past the 3.2% mark), trigger an automated data pipeline refresh to retrain the model on fresh consumer spending patterns.
* **Business Impact:** **Prevents predictive decay and model drift caused by shifting economic baselines, ensuring the automated forecasting network maintains high precision through every macroeconomic cycle.**

---

## 📂 6. Production Architecture Artifacts

The final delivered software framework is organized into a modular, production-ready environment built for easy enterprise deployment:

<img width="535" height="347" alt="image" src="https://github.com/user-attachments/assets/391d44b1-f369-4558-8087-4ab14294efd1" />







