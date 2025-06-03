## 4.A Summary

### <span style="color: #e74c3c;">**Continuous Features Analysis Summary**</span>

This analysis examines 18 continuous features in the student dataset to understand their distributions, relationships, and predictive power for identifying student withdrawal risk.

### <span style="color: #2E86AB;">**1. Feature Distributions**</span>

The histograms reveal several important patterns:
- **Academic grades** (qualification and admission grades) follow roughly normal distributions around 120-130
- **Age at enrollment** is heavily right-skewed with most students aged 17-25
- **Curricular units** (credited, enrolled, approved) show highly skewed distributions with many zero values
- **Economic indicators** (unemployment, inflation, GDP) show varied patterns across different time periods


### <span style="color: #2E86AB;">**2. Multicollinearity Issues**</span>

**Multicollinearity** occurs when features are highly correlated with each other, meaning they provide similar information to the model.

**High Correlations (>0.8):**
- 1st and 2nd semester grades: 0.84
- 1st and 2nd semester credited units: 0.94  
- 1st and 2nd semester enrolled units: 0.94
- 1st and 2nd semester approved units: 0.90

**VIF Analysis Results:**
**VIF (Variance Inflation Factor)** measures how much a feature's variance increases due to correlation with other features. Values >5 indicate problematic multicollinearity.

Several features show problematic multicollinearity:
- Enrolled units (1st sem): VIF = 23.49 (HIGH)
- Credited units (1st sem): VIF = 15.57 (HIGH)  
- Enrolled units (2nd sem): VIF = 16.42 (HIGH)
- Other academic performance metrics: VIF = 5-12 (MODERATE-HIGH)

### <span style="color: #2E86AB;">**3. Predictive Power**</span>

**Point-biserial correlation** measures the relationship between a continuous variable and a binary variable (in this case, withdrawal vs continuation). Values range from -1 to +1, with stronger correlations indicating better predictive power.

**Strongest predictors** (correlation with target):
- 2nd semester grades: 0.57
- 2nd semester approved units: 0.57
- 1st semester grades: 0.48
- 1st semester approved units: 0.48

**Moderate predictors:**
- Age at enrollment: -0.25 (negative correlation - older students more likely to drop out)
- Units enrolled/evaluated: 0.12-0.16

**Weak predictors:**
- Economic indicators: -0.03 to 0.05
- Previous qualifications: 0.08-0.10

### <span style="color: #e74c3c;">**Implications for Machine Learning Models**</span>

### <span style="color: #2E86AB;">**k-Nearest Neighbours (k-NN)**</span>
- **Distance calculation impact**: Multicollinear features will dominate distance measurements, reducing model effectiveness
- **Scaling required**: **Scaling** transforms features to similar ranges (e.g., 0-1) so no single feature dominates distance calculations due to its scale. Features have vastly different scales (grades 0-190 vs units 0-26)
- **Curse of dimensionality**: The **curse of dimensionality** means that as the number of features increases, data points become increasingly sparse and distant from each other, making similarity measures less meaningful. 18 features may be too many without dimensionality reduction

### <span style="color: #2E86AB;">**Logistic Regression**</span>
- **Coefficient instability**: **Coefficient instability** occurs when small changes in data cause large changes in model coefficients, making the model unreliable. High multicollinearity will make coefficients unreliable and difficult to interpret
- **Convergence issues**: Redundant features may cause numerical instability during fitting
- **Feature redundancy**: Similar information captured multiple times reduces model efficiency

### <span style="color: #e74c3c;">**Recommended Actions**</span>

### <span style="color: #2E86AB;">**1. Feature Selection**</span>
- **Remove redundant features**: Keep only one semester's academic metrics (2nd semester shows higher correlation)
- **Drop weak predictors**: Consider removing economic indicators and previous qualification grades
- **Priority features**: Focus on 2nd semester grades, approved units, age, and enrollment patterns

### <span style="color: #2E86AB;">**2. Feature Engineering**</span>
- **Combine related features**: Create composite academic performance scores
- **Feature scaling**: Apply StandardScaler or MinMaxScaler for k-NN
- **Handle skewness**: Consider log transformation for highly skewed unit counts

### <span style="color: #2E86AB;">**3. Dimensionality Reduction**</span>
- **Principal Component Analysis (PCA)**: **PCA** creates new features that are combinations of original features, capturing the most important patterns while reducing the total number of features. Reduce academic performance features to key components
- **Correlation-based filtering**: Remove features with correlation >0.8
- **Univariate selection**: **Univariate selection** evaluates each feature individually against the target variable and keeps only the most statistically significant ones. Keep only features with significant correlation to target (p<0.05)

### <span style="color: #2E86AB;">**4. Model-Specific Preparations**</span>
- **For k-NN**: Mandatory feature scaling, consider feature selection to reduce noise
- **For Logistic Regression**: Address multicollinearity first, then apply **regularisation (L1/L2)** if needed. **Regularisation** adds a penalty term to the model to prevent overfitting: L1 (Lasso) can automatically remove unimportant features by setting their coefficients to zero, whilst L2 (Ridge) shrinks coefficients towards zero to reduce their impact.

This analysis reveals that academic performance metrics are the strongest predictors, but careful feature selection is essential to avoid multicollinearity issues that could harm both models' performance.