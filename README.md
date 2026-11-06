# Codveda Data Analytics Internship

Projects completed for the **Data Analytics internship at Codveda Technology**, covering two tasks at each of the three levels. Everything is written in Python and uses publicly available datasets.

## Projects

| Level | Task | Dataset | Script |
|---|---|---|---|
| 1 (Basic) | Exploratory Data Analysis | Diabetes progression, 442 patients | [`level1_eda_and_visualization.py`](level1_basic/level1_eda_and_visualization.py) |
| 1 (Basic) | Basic Data Visualization | Same | Same file |
| 2 (Intermediate) | Regression Analysis | Diabetes progression | [`level2_regression_and_clustering.py`](level2_intermediate/level2_regression_and_clustering.py) |
| 2 (Intermediate) | Clustering (K-Means) | UCI Wine, 178 samples, 13 features | Same file |
| 3 (Advanced) | Predictive Modeling (Classification) | Wisconsin Breast Cancer, 569 tumours | [`level3_classification.py`](level3_advanced/level3_classification.py) |
| 3 (Advanced) | NLP Sentiment Analysis | NLTK movie reviews, 2,000 reviews | [`level3_nlp_sentiment_analysis.py`](level3_advanced/level3_nlp_sentiment_analysis.py) |

## Results at a glance

### Level 1: Exploring the data
- No missing values or duplicates in the dataset.
- BMI (r = 0.59) and log-triglycerides (r = 0.57) correlate most strongly with disease progression.
- Average progression rises with BMI category: Normal 109, Overweight 165, Obese 213.
- Total cholesterol and LDL are almost duplicates (r = 0.90), which matters for modelling.



![BMI category](level1_basic/figures/viz_bar_bmi_category.png)




![Correlation heatmap](level1_basic/figures/eda_correlation_heatmap.png)



### Level 2: Regression and clustering
- **Regression** (80/20 train/test split): each extra BMI unit is associated with about 10.8 more points of progression. BMI alone gives a test R² of 0.23 (RMSE 63.7). Using all 10 features raises R² to 0.45.
- **K-Means** on standardised data: the elbow method and silhouette score both select k = 3. The clusters closely match the true wine cultivars (Adjusted Rand Index 0.90), even though the labels were never used for fitting.



![Regression](level2_intermediate/figures/regression_fit_and_residuals.png)




![K-Means clusters](level2_intermediate/figures/kmeans_cluster_scatter.png)



### Level 3: Classification
Positive class = malignant. Three models were trained on scaled features and tuned with 5-fold grid search (scored on F1). Test set: 114 tumours.

| Model (tuned) | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| Decision Tree | 0.930 | 1.000 | 0.810 | 0.895 | 0.943 |
| **Random Forest** | **0.974** | **1.000** | 0.929 | **0.963** | 0.995 |

Grid search did not improve test scores for Logistic Regression or Random Forest. With a small test set, these differences are within noise. In a medical setting recall matters most, because a missed malignant tumour is the costly error.



![Model comparison](level3_advanced/figures/model_comparison_tuned.png)




![ROC curves](level3_advanced/figures/roc_curves.png)



## How to run

    pip install -r requirements.txt
    python level1_basic/level1_eda_and_visualization.py
    python level2_intermediate/level2_regression_and_clustering.py
    python level3_advanced/level3_classification.py
    python level3_advanced/level3_nlp_sentiment_analysis.py

Each script saves its charts to a `figures/` folder next to it, along with CSV tables and a `*_results.txt` log. The NLP script needs internet access for its NLTK data.

## Tools
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, NLTK, TextBlob, WordCloud.

## Notes
- The datasets are classic teaching datasets bundled with scikit-learn or NLTK. In the diabetes data, sex is coded 1/2 with no documented meaning, so it appears as "Group 1/2".
- The NLP script was written for an environment with internet access; run it once in Colab to generate its outputs.
- Correlations describe patterns in the data and do not prove causes.

## Author
**Oluwaferanmi Adebanjo**
Internship: Data Analytics at [Codveda Technology](https://www.codveda.com)

#CodvedaJourney #CodvedaProjects
