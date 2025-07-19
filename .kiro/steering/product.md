# Product Overview

HSMDO (Hierarchical Shifted Multiplicative Dirichlet-Multinomial) Customer Segmentation System

A customer analytics tool that implements the HSMDO statistical model for customer lifetime value prediction and behavioral segmentation. The system processes transactional data to identify customer segments like "Champions", "Loyal at Risk", "Sleeping/Lost", and "Stable Customers" based on purchase frequency patterns and future purchase probability.

## Key Features
- Bayesian customer lifetime value modeling using STAN
- Automated customer segmentation based on lambda (purchase rate) and PZF (probability of zero future purchases)
- Russian language support for segment names and comments
- CSV data processing and export capabilities

## Customer Segments
- **Чемпионы (Champions)**: High lambda, low PZF - most valuable customers
- **Лояльные в зоне риска (Loyal at Risk)**: High lambda, high PZF - valuable but at risk
- **Спящие/Потерянные (Sleeping/Lost)**: PZF > 90% - likely churned customers
- **Стабильные покупатели (Stable Customers)**: Low PZF - reliable customers
- **Обычные покупатели (Regular Customers)**: Default segment for others