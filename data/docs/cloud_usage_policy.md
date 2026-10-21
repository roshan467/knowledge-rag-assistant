# Cloud Usage Policy (Acme Corp, synthetic data)

The approved cloud providers are AWS and Azure. Every cloud resource must be tagged with an owner and a cost centre. Public storage buckets are not allowed.

Changes to production environments are made only through the CI/CD pipeline after peer review. Data classified as confidential must be encrypted at rest and in transit.

Monthly budget alerts are sent to the resource owner at 80 percent of the approved budget. Unused development resources must be shut down every evening and at weekends.
