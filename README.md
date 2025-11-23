This repo is focused on creating and fine-tuning a specialized model to predict Cp distribution on airfoil where the flow is turbulent. The main project, which predicts all Cp and aerodynamics coefficient , is located on 'airfoil-ml' repository.

XFOIL prediction for turbulent flow is a significant simplification of real-world physics. The primary goal of this fine-tuned model is to have better prediction for such this flow.

Step 1 : the airfoil-ml is change so the label are: Cp_ps and Cp_ss
Step 2 : model is load and the CFD dataset is used to fine tune it
Step 3 : prediction for unseen data (not in both dataset)

![Cp comparison between airfoil-ml , Fine-tuned airfoil-ml , ground truth CFD data](media/cp_comparison.png)