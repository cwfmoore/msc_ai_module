## 3.A Summary

### <span style="color: #e74c3c;">**Target Variable Transformation**</span>

### <span style="color: #2E86AB;">**Binary Classification Approach**</span>

**Transformation**: Combined "Graduate" and "Enrolled" into single "Continuation" class (1), keeping "Dropout" as "Withdrawn" (0).

**Rationale**: From an institutional perspective, both graduates and currently enrolled students represent successful outcomes. The critical distinction is identifying students at risk of withdrawal for early intervention.

### <span style="color: #2E86AB;">**Algorithmic Benefits**</span>

**Logistic Regression**:
- Natural binary classification design
- **Clear probability interpretation**: Outputs values between 0-1 representing the probability a student will continue (e.g., 0.75 = 75% chance of continuation, 0.25 = 25% chance of withdrawal)
- Avoids multi-class complexity

**k-Nearest Neighbours**:
- Eliminates voting ties between three classes
- Cleaner decision boundaries in feature space
- Simplified majority voting mechanism

### <span style="color: #2E86AB;">**Class Distribution Results**</span>

| **Original (3-class)** | **Binary (2-class)** |
|------------------------|----------------------|
| Graduate: 2,209 (49.9%) | Continuation: 3,003 (67.9%) |
| Enrolled: 794 (17.9%) | Withdrawn: 1,421 (32.1%) |
| Dropout: 1,421 (32.1%) | |

**Outcome**: Manageable 68:32 class ratio suitable for both algorithms without requiring complex balancing techniques.

### <span style="color: #2E86AB;">**Practical Impact**</span>
This transformation aligns the machine learning task with institutional goals: identifying students who need support to continue their studies, regardless of their current academic standing.