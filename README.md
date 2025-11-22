# Fold-in-MRGSRec

The implementation for the paper "Efficient Incorporation of New Interactions in Graph Recommenders via Folding-In"

Abstract: Graph-based recommender systems have emerged as a powerful paradigm for personalized recommendations. However, their reliance on full model retraining to incorporate new users or new interactions creates scalability barriers. The task becomes infeasible in real-life recommender systems due to excessive time and resource costs involved. To address this limitation, we propose a fast and efficient method for updating graph-based recommender models \emph{without full model retraining} on new data. Instead of changing all weights, we modify only small share of user representations who have new interactions. Our approach achieves a remarkable speedup of 700x over conventional model retraining approaches, drastically reducing computational overhead while maintaining the accuracy of the recommendations. Furthermore, we integrate our method into a multi-representation architecture that combines graph and sequential-based methods to capture different user and item representations. Extensive experiments on diverse datasets demonstrate that our approach achieves state-of-the-art recommendation accuracy while maintaining the efficiency of incremental updates, outperforming existing methods in both speed and quality.

To reproduce experiments start the script:

python3 ./modeling/train.py --params ./configs/<config_name>.json

We utilize the following versions of libraries:

numpy==1.26.4

pandas==2.0.3

torch==2.5.1+cu124

sklearn==1.6.0

scipy==1.11.4
