# Importing necessary libraries
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class MultivariateStatisticalAnalysisTemplate(ABC):
    """
    Abstract base class for performing multivariate statistical analysis.
    """

    def render(self, df: pd.DataFrame) -> None:
        """
        Calls methods for heatmap correlation analysis and pairplot feature analysis.
        """
        self.heatmap_correlation_analysis(df)
        self.pairplot_feature_analysis(df)

    @abstractmethod
    def heatmap_correlation_analysis(self, df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def pairplot_feature_analysis(self, df: pd.DataFrame) -> None:
        pass


class SimpleMultivariateStatisticalAnalysis(MultivariateStatisticalAnalysisTemplate):
    """
    Concrete implementation of multivariate analysis with support for
    both numeric and categorical features.
    """

    def heatmap_correlation_analysis(self, df: pd.DataFrame) -> None:
        """
        Generates a hybrid correlation heatmap:
        - Numeric features use Pearson correlation.
        - Categorical features are label-encoded for visualization.
        """
        if df.empty:
            print("Dataset is empty. No heatmap can be generated.")
            return

        df_copy = df.copy()
        # Encode categorical features for correlation visualization
        for col in df_copy.select_dtypes(include=['object', 'category']).columns:
            df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))

        corr_matrix = df_copy.corr(numeric_only=True)

        if corr_matrix.empty:
            print("No valid features for correlation heatmap.")
            return

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Hybrid Feature Correlation Heatmap (Numeric + Encoded Categorical)")
        plt.show()

    def pairplot_feature_analysis(self, df: pd.DataFrame) -> None:
        """
        Generates hybrid pairwise plots:
        - Pairplot for numeric-numeric relationships.
        - Boxplots/stripplots for categorical-numeric relationships.
        """
        if df.empty:
            print("Dataset is empty. No plots can be generated.")
            return

        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(numeric_cols) < 1 and len(categorical_cols) < 1:
            print("No valid features for pairplot analysis.")
            return

        # 1️⃣ Pairplot for numeric features
        if len(numeric_cols) > 1:
            sns.pairplot(df[numeric_cols])
            plt.suptitle("Pairwise Relationships (Numeric Features)", y=1.02)
            plt.show()

        # 2️⃣ Boxplots/stripplots for categorical–numeric combinations
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                plt.figure(figsize=(10, 5))
                sns.boxplot(x=cat_col, y=num_col, data=df)
                sns.stripplot(x=cat_col, y=num_col, data=df, color='black', size=3, jitter=True, alpha=0.6)
                plt.title(f"{num_col} distribution across {cat_col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
