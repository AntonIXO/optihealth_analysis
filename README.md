# OptiHealth Analysis Engine

This repository contains the backend data analysis engine for OptiHealth, a personalized health tracking and insights platform. This service operates as an asynchronous worker that processes user data to generate personalized health insights based on statistical analysis and machine learning.

## Project Architecture

The analysis engine is designed as a robust, scalable, and asynchronous system that processes user data in the background. The core components are:

1.  **Job Queue (Supabase)**: A table in the Supabase PostgreSQL database (`public.analysis_jobs`) acts as a job queue. A cron job (`pg_cron`) populates this queue with analysis tasks for each active user.

2.  **Python Worker (`main_worker.py`)**: A standalone Python service that continuously polls the `analysis_jobs` table for pending jobs.

3.  **Analysis Modules**: A collection of Python scripts located in the `analysis/` directory, each responsible for a specific type of data analysis.

4.  **Configuration (`analysis_config.yaml`)**: A YAML file that defines which analyses to run and with what parameters. This allows for easy enabling, disabling, and configuration of analyses without changing the core application logic.

### Workflow

The typical workflow for an analysis job is as follows:

1.  A `pg_cron` job schedules an analysis for a user by inserting a `pending` job into the `analysis_jobs` table.
2.  The Python worker (`main_worker.py`) fetches the pending job and updates its status to `in_progress`.
3.  The worker fetches the necessary health data for the user from the database using the functions in `data_loader.py`.
4.  The worker reads `analysis_config.yaml` to determine which analysis functions to execute.
5.  It dynamically loads and runs the appropriate functions from the `analysis/` modules, passing the user's data and configuration parameters.
6.  The analysis functions return structured JSON objects representing the generated insights.
7.  The worker stores these insights back into the `insights` table in the database.
8.  The job's status is updated to `completed`.

### 🚀 Quickstart & Local Development

**Option 1: Using Docker (Recommended)**
```bash
cp .env.docker .env
docker-compose up --build
```

**Option 2: Local Python Environment**
```bash
# Install dependencies using uv
uv sync

# Run the worker
python main_worker.py
```

## 🧪 Testing
To ensure the analysis modules are working correctly, run the test suite:
```bash
python test_runner.py
```

## Analysis Modules

The `analysis/` directory contains the core data science logic of the platform. Each module is responsible for a different layer of analysis:

-   `goals.py`: Analyzes the user's data against their defined personal goals. It calculates "streaks" of consecutive days where a goal was met, providing motivational feedback.
-   `correlation.py`: Performs correlation analysis (e.g., Pearson, Spearman) to find statistical relationships between different health metrics (e.g., "How does sleep duration affect resting heart rate?").
-   `comparative.py`: Compares metric distributions across different conditions or events using statistical tests (e.g., t-test) to answer questions like, "Is my heart rate lower on days I meditate?".
-   `clustering.py`: Uses unsupervised machine learning algorithms (e.g., K-Means) to discover hidden patterns by grouping days into distinct archetypes (e.g., "High-Stress Workdays," "Active Recovery Days").
-   `feature_importance.py`: Employs regression models (e.g., XGBoost) to identify and rank the lifestyle factors that most significantly predict a key health outcome, such as sleep score.
-   `forecasting.py`: Utilizes time-series models like Prophet to forecast future trends in key health metrics, helping users anticipate changes.

# **⚙️ Managing the Analysis Configuration (analysis\_config.yaml)**

The analysis\_config.yaml file is the central nervous system of the OptiHealth Insight Engine. It provides a simple, declarative way to define, enable, disable, and tweak every analysis the backend worker can perform without changing any Python code. Understanding this file is key to extending the platform's capabilities.

## **Core Concepts**

The entire configuration is a list under a single top-level key: analyses. Each item in this list is an **Analysis Block**, which represents one specific insight to be generated (e.g., correlating sleep vs. heart rate).

The worker processes this file sequentially, running each analysis block where enabled: true.

### **Anatomy of an Analysis Block**

Each block consists of several key-value pairs that tell the engine what to do and how to do it.

\# Example of a generic analysis block  
\- name: UniqueAnalysisName              \# A unique, human-readable name for this specific analysis.  
  module: python\_module\_name           \# The name of the .py file in the /analysis directory (without .py).  
  enabled: true                        \# Set to \`true\` to run, \`false\` to disable.  
  function: function\_to\_call           \# The specific function to execute within the module.  
  parameters:                          \# A dictionary of settings passed directly to the function.  
    param\_1: value\_1  
    param\_2: value\_2

## **Module-Specific Parameters**

The parameters section is the most important part of the configuration, as it changes depending on which module you are using. Below is a detailed guide for each available module.

### **module: correlation**

Finds statistical relationships between two numeric metrics over time.

| Parameter | Type | Description |
| :---- | :---- | :---- |
| metric\_a | string | The name of the first metric (must match metric\_definitions). |
| metric\_b | string | The name of the second metric. |
| method | string | The correlation method to use. Can be 'pearson' (for linear relationships) or 'spearman' (for monotonic relationships). |
| min\_data\_points | integer | The minimum number of days with data for both metrics required to run the analysis. |
| significance\_threshold | float | The absolute value of the correlation coefficient (e.g., 0.4) that must be met to generate an insight. |

### **module: comparative**

Compares a numeric metric on days following a specific event versus all other days.

| Parameter | Type | Description |
| :---- | :---- | :---- |
| event\_name | string | The name of the event to look for (e.g., "Running"). |
| metric | string | The numeric metric to compare between the two groups of days. |
| analysis\_type | string | The statistical test to use. Can be 'ttest' or 'mannwhitneyu' (recommended default). |
| time\_window\_days | integer | How many days *after* the event to include in the "event group" (e.g., 1 for the day immediately after). |
| min\_group\_size | integer | The minimum number of data points required in both the "event group" and the "control group". |
| significance\_threshold | float | The p-value (e.g., 0.05) below which the difference is considered statistically significant. |

### **module: clustering**

Discovers hidden patterns by grouping similar days together into "Day Types".

| Parameter | Type | Description |
| :---- | :---- | :---- |
| metrics\_to\_cluster | list of strings | A list of metric names that define the "character" of a day. |
| outcome\_metric | string | A key health metric (e.g., hrv\_rmssd) used to evaluate and describe the clusters. **Must not** be in metrics\_to\_cluster. |
| n\_clusters | integer | The number of distinct "Day Types" to find (typically 3 or 4). |
| min\_days | integer | The minimum number of complete days of data required to run the clustering analysis. |

### **module: feature\_importance**

Uses a machine learning model to find the most powerful predictors for a target health outcome.

| Parameter | Type | Description |
| :---- | :---- | :---- |
| target\_metric | string | The outcome metric you want to understand (e.g., sleep\_score). |
| feature\_metrics | list of strings | A list of all potential factors that could influence the target metric. |
| min\_days | integer | The minimum number of complete days of data required to train the model. |
| top\_n\_features | integer | The number of top predictors to report in the final insight (e.g., 3). |

### **module: forecasting**

Analyzes trends, projects future values, and detects anomalies for a single metric.

| Parameter | Type | Description |
| :---- | :---- | :---- |
| metric\_to\_forecast | string | The metric to analyze and forecast. |
| days\_to\_forecast | integer | How many days into the future to project the trend (e.g., 14). |
| min\_days | integer | The minimum number of historical data points required to train the forecast model. |
| desired\_trend | string | The direction of a "positive" trend. Can be 'increasing' (for metrics like HRV) or 'decreasing' (for resting HR). |
| anomaly\_sensitivity | float | The confidence interval for anomaly detection (e.g., 0.99). A higher value means fewer, more extreme anomalies will be flagged. |

### **module: goals**

Tracks user-defined goals and generates motivational insights about streaks.

| Parameter | Type | Description |
| :---- | :---- | :---- |
| min\_streak\_for\_insight | integer | The minimum number of consecutive days a user must meet their goal for an insight to be generated. |

## **How to Add a New Analysis**

Adding a new insight is as simple as adding a new block to the analyses list.

1. **Copy & Paste**: Find an existing block that is similar to what you want to achieve and copy it.  
2. **Give it a Unique Name**: Change the name to something descriptive and unique (e.g., HRV\_vs\_DeepSleep).  
3. **Configure the Module**: Set the module, function, and all required parameters for the analysis you want to run. Refer to the guide above for the correct parameters.  
4. **Enable the Analysis**: Set enabled: true.  
5. **Test**: The worker will automatically pick up and run your new analysis on its next cycle. Check the logs to see if it produces the expected insight.

### **Best Practices**

* **Use Comments**: Add comments (\#) in the YAML file to explain why an analysis was added or what it's intended to show.  
* **Exact Metric Names**: Ensure all metric names in the parameters match the metric\_name in your metric\_definitions table exactly.  
* **Test in Isolation**: When creating a new analysis, it can be helpful to set enabled: false for all other blocks to isolate the output and logs for the one you're testing.
