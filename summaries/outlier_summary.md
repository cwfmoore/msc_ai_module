## 6.A Summary

### <span style="color: #e74c3c;">**Outlier Analysis Summary**</span>

This analysis examines extreme values in the continuous features to understand their nature, frequency, and potential impact on our machine learning models for predicting student withdrawal.

### <span style="color: #2E86AB;">**1. What Are Outliers and Why Do They Matter?**</span>

**Outliers** are data points that fall significantly outside the normal range of values for a feature. They can arise from:
- **Data entry errors** (typos, measurement mistakes)
- **Legitimate extreme cases** (mature students, exceptional performance)
- **Missing or incomplete data** (zeros from non-attendance)

**Why outlier analysis matters:**
- Outliers can **skew model performance** and lead to poor predictions
- Some outliers contain **valuable information** (e.g., failing students are important for dropout prediction)
- Different models are affected differently by extreme values
- Proper handling improves **model robustness** and **generalisation**

### <span style="color: #2E86AB;">**2. Key Findings from Our Analysis**</span>

**High Outlier Features (>10%):**
- **Age at enrollment**: 10.0% outliers - likely mature students (>25 years)
- **1st semester grades**: 16.4% outliers - mostly zeros from non-attending students
- **2nd semester grades**: 15.8% outliers - zeros from early dropouts
- **Credited/enrolled units**: 9-13% outliers - zeros from incomplete coursework

**Academic Performance Pattern:**
Most outliers in academic metrics are **zeros**, representing students who:
- Didn't attend classes
- Failed to complete assessments
- Dropped out early in the semester

**Economic Indicators:**
Very clean data with <1% outliers - no action needed.

### <span style="color: #2E86AB;">**3. Understanding the Context**</span>

**Important insight**: Many "outliers" in our dataset are **legitimate and informative values** rather than errors. Zero grades and zero credited units are strong indicators of student struggle and potential withdrawal - exactly what we want to predict.

**The IQR Method**: We used the **Interquartile Range (IQR) method** to detect outliers, where values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are flagged as outliers. This is a standard statistical approach that identifies the most extreme 5-10% of values.

### <span style="color: #e74c3c;">**Impact on Machine Learning Models**</span>

### <span style="color: #2E86AB;">**k-Nearest Neighbours (k-NN)**</span>

**Distance calculation sensitivity**: k-NN calculates distances between data points to find similar cases. Outliers can:
- **Dominate distance calculations** if features aren't scaled properly
- **Create isolated points** that don't have meaningful neighbours
- **Skew similarity measurements** leading to poor classification

**Specific issues in our data:**
- Large age outliers (>50) could overwhelm other features in distance calculations
- Zero academic performance creates clusters of failing students (which might actually be useful)
- **Feature scaling becomes critical** to prevent grade outliers (0-200 range) from dominating

### <span style="color: #2E86AB;">**Logistic Regression**</span>

**Coefficient estimation problems**: Logistic regression finds the best linear boundary between classes. Outliers can:
- **Pull the decision boundary** towards extreme values
- **Inflate coefficient estimates** making the model unstable
- **Reduce model interpretability** by giving extreme values too much influence

**Leverage and influence**: **Leverage** measures how far a data point is from others in feature space, whilst **influence** measures how much a single point affects the model. High-leverage outliers can have disproportionate influence on the final model.

### <span style="color: #e74c3c;">**Recommended Actions**</span>

### <span style="color: #2E86AB;">**1. Keep Informative Outliers**</span>
**DO NOT remove academic performance outliers** - they're crucial for prediction:
- Zero grades indicate struggling students
- Zero units show non-engagement
- These are **strong predictors** of withdrawal risk

### <span style="color: #2E86AB;">**2. Address Problematic Outliers**</span>
**Age at enrollment**: Consider capping extreme ages (>60) as they might represent data errors or very unusual cases that could skew the model.

**Previous qualification grades**: Very low scores (<50) might indicate data entry errors and could be investigated.

### <span style="color: #2E86AB;">**3. Feature Engineering Opportunities**</span>
Instead of removing outliers, create **binary indicator features**:
- `has_zero_grade_sem1`: Student received zero grade in first semester
- `has_zero_units_sem1`: Student completed zero units in first semester  
- `mature_student`: Student aged >25 at enrollment
- `low_prior_qualification`: Previous qualification grade <100

### <span style="color: #2E86AB;">**4. Model-Specific Preparations**</span>

**For k-NN:**
- **Mandatory feature scaling** to prevent outliers from dominating distance calculations
- Consider **robust scaling** (median and IQR-based) instead of standard scaling
- Keep academic outliers as they represent meaningful patterns

**For Logistic Regression:**
- **Feature scaling** recommended but less critical than for k-NN
- Consider **robust regression techniques** if outliers prove problematic
- **Regularisation** (L1/L2) can help reduce outlier influence automatically

### <span style="color: #2E86AB;">**5. Validation Strategy**</span>
**Monitor model performance** with and without outlier treatment to ensure we're not removing valuable predictive information. Academic performance outliers likely **improve** rather than harm prediction accuracy.

### <span style="color: #e74c3c;">**Next Steps**</span>

With outlier analysis complete, proceed to:
1. **Feature selection** - remove redundant and weak predictors identified earlier
2. **Feature scaling** - prepare data for both models  
3. **Train/test split** - ensure proper data separation
4. **Baseline model training** - test both approaches with cleaned data

This analysis confirms that most "outliers" in our dataset represent legitimate academic struggles - exactly the patterns our models need to learn for accurate dropout prediction.