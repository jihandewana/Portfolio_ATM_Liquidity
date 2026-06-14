-- ====================================================================
-- PROJECT      : Liquidity Risk Analysis & ATM Cash Demand Forecasting
-- AUTHOR       : Jihan Rana Ayunda Dewana
-- ROLE         : Data Scientist / Analytics Professional
-- DIALECT      : SQLite (Timeline-Agnostic Resilient Schema)
-- DESCRIPTION  : Uses a LEFT JOIN framework to preserve transaction volumes 
--                across mismatched operational timelines (2018 vs 2022).
-- ====================================================================

DROP TABLE IF EXISTS aggregated_atm_features;

CREATE TABLE aggregated_atm_features (
    Transaction_Date TEXT,
    ATM_ID TEXT,
    Location_Type TEXT,
    Cash_Demand_Next_Day REAL,
    Total_Cash_Out REAL,
    Total_Transactions INTEGER,
    Debt_Equity_Ratio REAL,
    Financial_Risk_Label TEXT,
    GDP_Growth_Rate REAL,
    Inflation_Rate REAL
);

INSERT INTO aggregated_atm_features
SELECT 
    t.Date AS Transaction_Date,
    t.ATM_ID AS ATM_ID,
    COALESCE(m.Location_Type, 'Unknown') AS Location_Type,
    m.Cash_Demand_Next_Day,
    SUM(t.Amount) AS Total_Cash_Out,
    SUM(t.Number_of_Trxs) AS Total_Transactions,
    r.Debt_Equity_Ratio,
    r.Financial_Risk_Label,
    r.GDP_Growth_Rate,
    r.Inflation_Rate
FROM atm_transactions t
LEFT JOIN atm_master m 
    -- Cleanly match numeric IDs even if the master table uses the 'ATM_' text prefix
    ON CAST(t.ATM_ID AS TEXT) = CAST(REPLACE(m.ATM_ID, 'ATM_', '') AS INT)
LEFT JOIN corporate_risk r 
    -- Match macro factors to transaction dates if available
    ON t.Date = r.Date
GROUP BY 
    t.Date, 
    t.ATM_ID
ORDER BY 
    Transaction_Date ASC, 
    t.ATM_ID ASC;