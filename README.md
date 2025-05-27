# ICL-structured-data

Mary Letey 2025

Code reproducing figures and experiments for  

**_Solvable Model of In-Context Pretrain-Test Task Alignment_**

We study how task structure during pretraining shapes inductive bias in in-context learning (ICL) of regression tasks. We present a fully solvable model of linear regression performed by linear attention, and derive exact asymptotics for the average test error as a function of pretraining and testing task structure, as well as pretraining task diversity, context length, and number of contexts. We show that the generalization error decomposes into two components: a pretraining bias term, describing the model’s inductive bias from pretraining, and a pretrain-test task interaction term. Our analysis identifies a precise tradeoff between specialization and generalization in ICL, particularly evident in realistic task distributions that exhibit non-isotropic structure. We also show that longer test-time contexts reliably improve accuracy, consistent with empirical observations.


_Mary Letey, Yue M. Lu, Cengiz Pehlevan_
