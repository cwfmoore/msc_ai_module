## 5.A Summary

### <span style="color: #e74c3c;">**Categorical Features Analysis Summary**</span>

This analysis examines 18 categorical features in the student dataset to understand their diversity, balance, and predictive value for identifying student withdrawal risk.

### <span style="color: #2E86AB;">**1. Feature Cardinality**</span>

**Cardinality** refers to the number of unique categories within each feature. The analysis reveals:
- **Binary features** (8 features): Simple yes/no or male/female categories with 2 values each
- **Low cardinality** (2 features): Marital status (6), application order (8) 
- **Medium cardinality** (4 features): Course (17), previous qualification (17), application mode (18), nationality (21)
- **High cardinality** (4 features): Parents' qualifications and occupations (29-46 categories)

### <span style="color: #2E86AB;">**2. Class Imbalance Issues**</span>

**Class imbalance** occurs when one category dominates a feature, making it less useful for prediction.

**Severely imbalanced features** (>95% in one category):
- Nationality: 97.51% Portuguese students
- Educational special needs: 98.85% have no special needs
- International status: 97.51% are domestic students

These features provide little variation and minimal predictive value.

### <span style="color: #2E86AB;">**3. Statistical Significance**</span>

**Chi-square tests** measure whether categorical features are statistically associated with the target variable. A p-value <0.05 indicates significant association.

**Highly significant predictors** (p<0.001):
- Tuition fees up to date: χ² = 811.93
- Application mode: χ² = 399.12
- Scholarship holder: χ² = 265.10
- Course type: χ² = 298.27

**Non-significant predictors** (p>0.05):
- International status
- Nationality  
- Educational special needs

### <span style="color: #2E86AB;">**4. Information Content**</span>

**Mutual information** measures how much information each feature provides about the target variable. Higher scores indicate more predictive power.

**Most informative features:**
- Tuition fees up to date: 0.085
- Scholarship holder: 0.047
- Course type: 0.033
- Application mode: 0.029

**Uninformative features** (MI = 0.00):
- Nationality, daytime/evening attendance, displaced status, international status

### <span style="color: #e74c3c;">**Implications for Machine Learning Models**</span>

### <span style="color: #2E86AB;">**k-Nearest Neighbours (k-NN)**</span>
- **Encoding requirements**: **One-hot encoding** creates binary dummy variables for each category (e.g., "Course_Design"=1, "Course_Nursing"=0). High cardinality features create many new dimensions
- **Curse of dimensionality**: Parents' occupations (32-46 categories) would create 100+ new features after encoding
- **Distance distortion**: Irrelevant categories can dominate distance calculations

### <span style="color: #2E86AB;">**Logistic Regression**</span>
- **Coefficient interpretation**: Each category gets its own coefficient after encoding
- **Overfitting risk**: High cardinality features create many parameters that may not generalise well
- **Multicollinearity**: **Label encoding** assigns numbers to categories (e.g., Course_1, Course_2), but this implies false ordering relationships

### <span style="color: #e74c3c;">**Recommended Actions**</span>

### <span style="color: #2E86AB;">**1. Remove Uninformative Features**</span>
- **Drop severely imbalanced**: Nationality, educational special needs, international status
- **Remove zero-information**: Features with MI score = 0.00

### <span style="color: #2E86AB;">**2. Handle High Cardinality**</span>
- **Grouping strategy**: Combine rare categories in parents' occupations into "Other" category
- **Target encoding**: **Target encoding** replaces categories with their average target value, reducing dimensions whilst preserving predictive power
- **Feature selection**: Keep only top categories by frequency

### <span style="color: #2E86AB;">**3. Encoding Strategy**</span>
- **For k-NN**: Use one-hot encoding for low cardinality features (<10 categories)
- **For Logistic Regression**: One-hot encoding with **regularisation** to handle multiple coefficients. **Regularisation** prevents overfitting by penalising large coefficients
- **Alternative**: Label encoding for ordinal features (application order, qualifications)

### <span style="color: #2E86AB;">**4. Priority Features**</span>
**Keep these high-value features:**
- Tuition fees up to date
- Scholarship holder  
- Course type
- Application mode
- Gender
- Debtor status

**Consider removing:**
- Parents' detailed occupations/qualifications (too high cardinality)
- Nationality, international status (too imbalanced)
- Educational special needs (no variation)

This analysis shows that financial and course-related factors are the strongest categorical predictors, whilst demographic details provide limited additional value and risk model complexity.