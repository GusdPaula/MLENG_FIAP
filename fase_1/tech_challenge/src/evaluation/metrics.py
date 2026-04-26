"""Métricas técnicas e de negócio para avaliação de modelos de churn."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class TelcoMetrics:
    """
    Calcula metricas tecnicas e de negocio para o modelo de churn.

    Metricas Tecnicas:
    - F1-Score: Media harmonica de precisao e recall
    - AUC-ROC: Area sob a curva ROC (sensibilidade vs especificidade)
    - PR-AUC: Area sob a curva Precision-Recall
    - Recall: Taxa de verdadeiros positivos (sensibilidade)
    - Precision: Taxa de predicoes positivas corretas

    Metricas de Negocio:
    - Churn Cost Avoided: Receita economizada ao reter clientes
    - False Positive Cost: Custo de campanhas desnecessarias
    - Net Benefit: Lucro liquido da intervencao
    """

    # Parâmetros de negócio (configuráveis)
    CUSTOMER_LTV = 2080  # Monthly_Charges X Tenure = $65 X 32 meses
    RETENTION_COST = 50  # Custo de retenção por cliente ($)
    FALSE_POSITIVE_COST = 20  # Custo de campanha ineficaz ($)

    @staticmethod
    def calcular_metricas_tecnicas(y_true, y_pred, y_pred_proba):
        """
        Calcula todas as métricas técnicas.

        Args:
            y_true: Labels verdadeiros (0/1)
            y_pred: Predições (0/1)
            y_pred_proba: Probabilidades (0-1)

        Returns:
            dict com todas as métricas
        """
        return {
            # Probabilísticas
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "pr_auc": average_precision_score(y_true, y_pred_proba),
            # Baseadas em threshold
            "f1_score": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            # Acurácia
            "accuracy": np.mean(y_true == y_pred),
        }

    @staticmethod
    def calcular_metricas_negocio(y_true, y_pred):
        """
        Calcula métricas de negócio.

        Args:
            y_true: Labels verdadeiros (0/1)
            y_pred: Predições (0/1)

        Returns:
            dict com métricas de negócio
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Clientes retidos com sucesso
        churn_avoided = tp * TelcoMetrics.CUSTOMER_LTV

        # Custo das campanhas ineficazes (false positives)
        fp_cost = fp * TelcoMetrics.FALSE_POSITIVE_COST

        # Custo das campanhas bem-sucedidas
        retention_spent = tp * TelcoMetrics.RETENTION_COST

        # Lucro líquido
        net_benefit = churn_avoided - fp_cost - retention_spent

        # ROI simplificado
        roi = (
            (net_benefit / (fp_cost + retention_spent)) * 100
            if (fp_cost + retention_spent) > 0
            else 0
        )

        return {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "churn_avoided_revenue": float(churn_avoided),
            "false_positive_cost": float(fp_cost),
            "retention_cost": float(retention_spent),
            "net_benefit": float(net_benefit),
            "roi_percent": float(roi),
        }

    @staticmethod
    def todas_as_metricas(y_true, y_pred, y_pred_proba):
        """Combina métricas técnicas e de negócio."""
        return {
            **TelcoMetrics.calcular_metricas_tecnicas(y_true, y_pred, y_pred_proba),
            **TelcoMetrics.calcular_metricas_negocio(y_true, y_pred),
        }

    @staticmethod
    def relatorio_completo(y_true, y_pred):
        """Gera relatório textual completo."""
        print("\n" + "=" * 60)
        print("RELATÓRIO DE CLASSIFICAÇÃO - TELCO CHURN")
        print("=" * 60 + "\n")
        print(classification_report(y_true, y_pred, target_names=["Não Churn", "Churn"]))
        print("=" * 60)
