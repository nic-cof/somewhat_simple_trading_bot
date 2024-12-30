import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import TypeVar, Union

EstimatorType = TypeVar('EstimatorType', bound=Union[BaseEstimator, ClassifierMixin])
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """
    A class for comprehensive model evaluation and performance analysis.

    This class provides methods to:
    - Evaluate model performance with multiple metrics
    - Analyze predictions in detail
    - Generate performance reports
    - Compare multiple models
    - Create visualization of results
    """

    def __init__(self, output_dir: Optional[str] = "model_evaluation"):
        """
        Initialize the ModelEvaluator.

        Args:
            output_dir: Directory for storing evaluation results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Store evaluation history
        self.evaluation_history: List[Dict] = []

        # Default evaluation metrics
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': lambda y, yp: precision_score(y, yp, average='weighted'),
            'recall': lambda y, yp: recall_score(y, yp, average='weighted'),
            'f1': lambda y, yp: f1_score(y, yp, average='weighted')
        }

    def evaluate_performance(self,
                             model: EstimatorType,
                             x: pd.DataFrame,
                             y: pd.Series,
                             dataset_name: str = "Unknown") -> Dict:
        """
        Evaluate model performance comprehensively.

        Args:
            model: Trained model instance
            x: Feature DataFrame
            y: Target Series
            dataset_name: Name of the dataset being evaluated

        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Get predictions
            y_pred = model.predict(x)
            y_prob = model.predict_proba(x) if hasattr(model, 'predict_proba') else None

            # Calculate basic metrics with three classes
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision_macro': precision_score(y, y_pred, average='macro'),
                'recall_macro': recall_score(y, y_pred, average='macro'),
                'f1_macro': f1_score(y, y_pred, average='macro'),
                'precision_per_class': precision_score(y, y_pred, average=None).tolist(),
                'recall_per_class': recall_score(y, y_pred, average=None).tolist()
            }

            # Calculate ROC AUC for multi-class if probabilities available
            if y_prob is not None:
                # Calculate ROC AUC for each class using one-vs-rest approach
                metrics['roc_auc'] = roc_auc_score(y, y_prob, multi_class='ovr')

                # Calculate per-class ROC AUC
                metrics['roc_auc_per_class'] = {
                    'down': roc_auc_score(y == 0, y_prob[:, 0]),
                    'stable': roc_auc_score(y == 1, y_prob[:, 1]),
                    'up': roc_auc_score(y == 2, y_prob[:, 2])
                }

            # Get confusion matrix
            cm = confusion_matrix(y, y_pred)

            # Get detailed classification report
            class_report = classification_report(y, y_pred,
                                                 target_names=['down', 'stable', 'up'],
                                                 output_dict=True)

            # Store results
            evaluation_result = {
                'timestamp': datetime.now().isoformat(),
                'dataset_name': dataset_name,
                'metrics': metrics,
                'confusion_matrix': cm,
                'classification_report': class_report,
                'predictions': {
                    'y_true': y,
                    'y_pred': y_pred,
                    'y_prob': y_prob
                }
            }

            return evaluation_result

        except Exception as e:
            self.logger.error(f"Error during performance evaluation: {str(e)}")
            raise

    def analyze_predictions(self,
                            evaluation_result: Dict,
                            confidence_threshold: float = 0.8) -> Dict:
        """
        Perform detailed analysis of model predictions.

        Args:
            evaluation_result: Result dictionary from evaluate_performance
            confidence_threshold: Threshold for high-confidence predictions

        Returns:
            Dictionary containing prediction analysis
        """
        try:
            predictions = evaluation_result['predictions']
            y_true = predictions['y_true']
            y_pred = predictions['y_pred']
            y_prob = predictions['y_prob']

            analysis = {
                'overall_metrics': evaluation_result['metrics'],
                'per_class_performance': evaluation_result['classification_report'],
                'confusion_matrix': evaluation_result['confusion_matrix']
            }

            # Analyze prediction confidence if probabilities available
            if y_prob is not None:
                confidence_scores = np.max(y_prob, axis=1)
                high_conf_mask = confidence_scores >= confidence_threshold

                # High confidence predictions analysis
                if high_conf_mask.any():
                    high_conf_acc = accuracy_score(
                        y_true[high_conf_mask],
                        y_pred[high_conf_mask]
                    )

                    analysis['confidence_analysis'] = {
                        'high_confidence_predictions': sum(high_conf_mask),
                        'high_confidence_accuracy': high_conf_acc,
                        'average_confidence': confidence_scores.mean(),
                        'confidence_distribution': np.histogram(confidence_scores, bins=10)
                    }

            # Analyze error patterns
            error_mask = y_pred != y_true
            if error_mask.any():
                analysis['error_analysis'] = {
                    'total_errors': sum(error_mask),
                    'error_rate': sum(error_mask) / len(y_true),
                    'error_distribution': pd.crosstab(
                        y_true[error_mask],
                        y_pred[error_mask],
                        normalize='index'
                    )
                }

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing predictions: {str(e)}")
            raise

    def generate_metrics(self,
                         evaluation_result: Dict,
                         include_plots: bool = True) -> Dict:
        """
        Generate comprehensive metrics and visualizations.

        Args:
            evaluation_result: Result dictionary from evaluate_performance
            include_plots: Whether to generate and save plots

        Returns:
            Dictionary containing metrics and plot paths
        """
        try:
            metrics = evaluation_result['metrics'].copy()

            # Add additional derived metrics
            cm = evaluation_result['confusion_matrix']
            metrics['balanced_accuracy'] = np.mean(np.diag(cm) / np.sum(cm, axis=1))

            if include_plots:
                # Generate and save plots
                plot_paths = self.generate_evaluation_plots(evaluation_result)
                metrics['plot_paths'] = plot_paths

            return metrics

        except Exception as e:
            self.logger.error(f"Error generating metrics: {str(e)}")
            raise

    def compare_models(self,
                       models: Dict[str, EstimatorType],
                       x: pd.DataFrame,
                       y: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.

        Args:
            models: Dictionary of model name to model instance mappings
            x: Feature DataFrame
            y: Target Series

        Returns:
            DataFrame containing comparison results
        """
        try:
            comparison_results = []

            for model_name, model in models.items():
                # Evaluate model
                evaluation = self.evaluate_performance(model, x, y, model_name)
                metrics = evaluation['metrics']

                # Add model name and store results
                metrics['model_name'] = model_name
                comparison_results.append(metrics)

            # Create comparison DataFrame
            comparison_df = pd.DataFrame(comparison_results)

            # Set model name as index
            comparison_df.set_index('model_name', inplace=True)

            # Save comparison results
            self.save_comparison_results(comparison_df)

            return comparison_df

        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            raise

    def generate_evaluation_plots(self, evaluation_result: Dict) -> Dict[str, str]:
        """Generate and save evaluation plots."""
        plot_paths = {}

        try:
            # Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                evaluation_result['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues'
            )
            plt.title('Confusion Matrix')
            plot_path = self.output_dir / f"confusion_matrix_{datetime.now():%Y%m%d_%H%M%S}.png"
            plt.savefig(plot_path)
            plt.close()
            plot_paths['confusion_matrix'] = str(plot_path)

            # ROC Curve (if probabilities available)
            if evaluation_result['predictions']['y_prob'] is not None:
                plt.figure(figsize=(10, 8))
                for i in range(evaluation_result['predictions']['y_prob'].shape[1]):
                    y_true_binary = (evaluation_result['predictions']['y_true'] == i).astype(int)
                    y_prob_class = evaluation_result['predictions']['y_prob'][:, i]

                    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
                    plt.plot(fpr, tpr, label=f'Class {i}')

                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curves')
                plt.legend()

                plot_path = self.output_dir / f"roc_curves_{datetime.now():%Y%m%d_%H%M%S}.png"
                plt.savefig(plot_path)
                plt.close()
                plot_paths['roc_curves'] = str(plot_path)

            return plot_paths

        except Exception as e:
            self.logger.error(f"Error generating evaluation plots: {str(e)}")
            return plot_paths

    def save_comparison_results(self, comparison_df: pd.DataFrame) -> None:
        """Save model comparison results."""
        try:
            # Save to CSV
            output_path = self.output_dir / f"model_comparison_{datetime.now():%Y%m%d_%H%M%S}.csv"
            comparison_df.to_csv(output_path)
            self.logger.info(f"Comparison results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error saving comparison results: {str(e)}")

    def log_evaluation_results(self, evaluation_result: Dict) -> None:
        """Log evaluation results."""
        self.logger.info("\nModel Evaluation Results:")
        self.logger.info(f"Dataset: {evaluation_result['dataset_name']}")
        self.logger.info("\nMetrics:")
        for metric, value in evaluation_result['metrics'].items():
            self.logger.info(f"{metric}: {value:.4f}")

        self.logger.info("\nClassification Report:")
        for class_name, metrics in evaluation_result['classification_report'].items():
            if isinstance(metrics, dict):
                self.logger.info(f"\nClass {class_name}:")
                for metric_name, metric_value in metrics.items():
                    self.logger.info(f"{metric_name}: {metric_value:.4f}")
