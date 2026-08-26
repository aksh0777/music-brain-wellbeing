from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(numerical_cols, categorical_cols):
    """
    Constructs a scikit-learn ColumnTransformer for preprocessing.
    
    Parameters:
    -----------
    numerical_cols : list
        List of numerical feature names.
    categorical_cols : list
        List of categorical feature names.
        
    Returns:
    --------
    preprocessor : ColumnTransformer
        Configured ColumnTransformer with preprocessing steps.
    """
    # Define preprocessing for numerical columns: imputer + scaler
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Define preprocessing for categorical columns: imputer + one-hot encoder
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])
    
    return preprocessor
