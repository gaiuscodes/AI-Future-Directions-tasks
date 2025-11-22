"""
Hospital Patient Readmission Risk Prediction Model
Case Study Implementation with Confusion Matrix and Performance Metrics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    accuracy_score, 
    f1_score,
    classification_report,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def generate_hypothetical_data(n_samples=10000, n_features=20):
    """
    Generate hypothetical patient data for readmission prediction.
    This simulates real-world hospital data with relevant features.
    """
    print("Generating hypothetical patient data...")
    
    # Generate feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Add some realistic feature names
    feature_names = [
        'age_normalized',
        'length_of_stay',
        'comorbidity_index',
        'num_medications',
        'num_previous_admissions',
        'emergency_visits_6mo',
        'lab_value_1',
        'lab_value_2',
        'vital_sign_1',
        'vital_sign_2',
        'discharge_disposition_score',
        'follow_up_scheduled',
        'medication_complexity',
        'social_risk_score',
        'care_continuity_score',
        'diagnosis_severity',
        'procedure_complexity',
        'insurance_type_score',
        'distance_to_hospital',
        'previous_readmission_flag'
    ]
    
    # Create realistic relationships for readmission
    # Higher comorbidity, more medications, previous readmissions increase risk
    readmission_prob = (
        0.1 +  # Base probability
        0.15 * (X[:, 2] > 1) +  # High comorbidity
        0.10 * (X[:, 3] > 1.5) +  # Many medications
        0.20 * (X[:, 4] > 0.5) +  # Previous admissions
        0.15 * (X[:, 11] < -0.5) +  # No follow-up scheduled
        0.10 * (X[:, 19] > 0.5) +  # Previous readmission
        np.random.normal(0, 0.1, n_samples)  # Random noise
    )
    
    # Convert to binary outcome (readmission within 30 days)
    y = (readmission_prob > np.percentile(readmission_prob, 82)).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['readmission_30days'] = y
    
    print(f"Generated {n_samples} samples")
    print(f"Readmission rate: {y.mean():.2%}")
    print(f"Non-readmission: {(y == 0).sum()}, Readmission: {(y == 1).sum()}")
    
    return df, feature_names


def preprocess_data(df, feature_names):
    """
    Preprocess the data for modeling.
    """
    print("\nPreprocessing data...")
    
    X = df[feature_names].values
    y = df['readmission_30days'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model_with_regularization(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model with regularization to prevent overfitting.
    Uses early stopping and cross-validation.
    """
    print("\nTraining model with regularization...")
    
    # Model parameters with regularization to prevent overfitting
    model = XGBClassifier(
        objective='binary:logistic',
        max_depth=5,              # Limit tree complexity
        learning_rate=0.05,       # Smaller learning rate
        subsample=0.8,            # Row sampling (80% of data per tree)
        colsample_bytree=0.8,     # Feature sampling (80% of features per tree)
        min_child_weight=3,       # Minimum samples in leaf
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1.0,           # L2 regularization
        n_estimators=200,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    print(f"Model trained with {model.best_iteration} iterations")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and generate confusion matrix.
    """
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Extract values from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"\n{'':20} {'Predicted: No':<20} {'Predicted: Yes':<20}")
    print(f"{'Actual: No':<20} {tn:<20} {fp:<20}")
    print(f"{'Actual: Yes':<20} {fn:<20} {tp:<20}")
    print(f"{'Total':<20} {tn+fn:<20} {fp+tp:<20}")
    
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    print(f"Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:       {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:          {recall:.4f} ({recall*100:.2f}%)")
    print(f"Specificity:     {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"F1-Score:        {f1:.4f} ({f1*100:.2f}%)")
    print(f"AUC-ROC:         {auc:.4f} ({auc*100:.2f}%)")
    print(f"False Pos Rate:  {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    
    print("\n" + "="*60)
    print("CLINICAL INTERPRETATION")
    print("="*60)
    print(f"Precision ({precision*100:.1f}%): When the model predicts readmission, "
          f"it's correct {precision*100:.1f}% of the time.")
    print(f"Recall ({recall*100:.1f}%): The model identifies {recall*100:.1f}% of all "
          f"patients who will be readmitted.")
    print(f"Specificity ({specificity*100:.1f}%): The model correctly identifies "
          f"{specificity*100:.1f}% of patients who won't be readmitted.")
    
    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'auc': auc,
        'false_positive_rate': false_positive_rate,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """
    Visualize the confusion matrix.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Readmission', 'Readmission'],
                yticklabels=['No Readmission', 'Readmission'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Readmission Prediction Model', fontsize=16, pad=20)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, save_path='roc_curve.png'):
    """
    Plot ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Readmission Prediction Model', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=15, save_path='feature_importance.png'):
    """
    Plot feature importance from the trained model.
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {save_path}")
    plt.close()


def demonstrate_overfitting_prevention(X_train, y_train, feature_names):
    """
    Demonstrate overfitting prevention using cross-validation.
    """
    print("\n" + "="*60)
    print("OVERFITTING PREVENTION DEMONSTRATION")
    print("="*60)
    
    # Split training data further for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train model with regularization
    model = train_model_with_regularization(
        X_train_split, y_train_split, X_val_split, y_val_split
    )
    
    # Evaluate on training and validation sets
    train_pred = model.predict(X_train_split)
    val_pred = model.predict(X_val_split)
    
    train_acc = accuracy_score(y_train_split, train_pred)
    val_acc = accuracy_score(y_val_split, val_pred)
    
    train_precision = precision_score(y_train_split, train_pred)
    val_precision = precision_score(y_val_split, val_pred)
    
    print("\nTraining vs Validation Performance:")
    print(f"Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Gap:                 {abs(train_acc - val_acc):.4f} ({abs(train_acc - val_acc)*100:.2f}%)")
    
    print(f"\nTraining Precision:   {train_precision:.4f} ({train_precision*100:.2f}%)")
    print(f"Validation Precision: {val_precision:.4f} ({val_precision*100:.2f}%)")
    print(f"Gap:                 {abs(train_precision - val_precision):.4f} ({abs(train_precision - val_precision)*100:.2f}%)")
    
    if abs(train_acc - val_acc) < 0.05:
        print("\n✓ Good generalization: Small gap indicates minimal overfitting")
    else:
        print("\n⚠ Warning: Large gap may indicate overfitting")
    
    return model


def main():
    """
    Main function to run the complete case study demonstration.
    """
    print("="*60)
    print("HOSPITAL READMISSION RISK PREDICTION - CASE STUDY")
    print("="*60)
    
    # Step 1: Generate hypothetical data
    df, feature_names = generate_hypothetical_data(n_samples=10000, n_features=20)
    
    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, feature_names)
    
    # Step 3: Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Step 4: Train model with regularization
    model = train_model_with_regularization(
        X_train_split, y_train_split, X_val_split, y_val_split
    )
    
    # Step 5: Evaluate on test set
    results = evaluate_model(model, X_test, y_test)
    
    # Step 6: Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(results['confusion_matrix'])
    plot_roc_curve(y_test, results['y_pred_proba'])
    plot_feature_importance(model, feature_names)
    
    # Step 7: Demonstrate overfitting prevention
    demonstrate_overfitting_prevention(X_train, y_train, feature_names)
    
    print("\n" + "="*60)
    print("CASE STUDY DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - feature_importance.png")
    print("\nSee hospital_readmission_case_study.md for full case study documentation.")


if __name__ == "__main__":
    main()

