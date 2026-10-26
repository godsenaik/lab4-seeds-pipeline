\# Seeds ML CI Pipeline



\*\*Student Name:\*\* Hemant Talwar  

\*\*Roll Number:\*\* 25P0620005  

\*\*Course:\*\* MSc Artificial Intelligence, Goa Business School  

\*\*Academic Batch:\*\* 2025–26



\## 1. Project Overview



This project implements a Continuous Integration (CI) pipeline for a machine-learning classification task using the UCI Seeds dataset. The pipeline automatically validates the dataset, trains and evaluates a baseline and candidate model, checks model quality, runs application tests, and publishes a downloadable package only when all checks pass.



\## 2. Dataset and Task



\- \*\*Dataset:\*\* UCI Seeds

\- \*\*Task:\*\* Multiclass classification

\- \*\*Dataset size:\*\* 210 rows

\- \*\*Input features:\*\* 7 numerical features describing wheat seeds

\- \*\*Target:\*\* Wheat variety, represented by classes 1, 2, and 3

\- \*\*Missing values:\*\* 0

\- \*\*Dataset size on disk:\*\* Approximately 9.3 KB



The dataset is suitable for this project because it is small, tabular, and fast to process on a CPU.



\## 3. Models and Evaluation



The baseline model is a `DummyClassifier` using the most-frequent-class strategy. The candidate model is a `RandomForestClassifier` with 100 estimators and a fixed random state of 42, inside a preprocessing pipeline using `StandardScaler`.



The dataset is split reproducibly using a fixed random state of 42, with 25% reserved for testing. The preprocessing pipeline is fitted on the training data.



\*\*Evaluation metric:\*\* Macro F1-score. It evaluates classification performance across all classes and gives each class equal weight.



\*\*Quality gate:\*\* The candidate model must achieve a Macro F1-score at least 0.05 higher than the baseline score. This improvement margin is fixed for the experiment.



The successful run reported a baseline Macro F1-score of 0.1619 and a candidate score of 0.9021. The required score was 0.2119, so the quality gate passed.



\## 4. Automated CI Pipeline



The workflow is defined in `.github/workflows/ci.yml` and runs on pushes to the `main` branch. It can also be started manually.



The pipeline performs these steps:



1\. Checks out the repository.

2\. Sets up Python 3.12.

3\. Installs the dependencies.

4\. Validates the dataset.

5\. Trains and evaluates the models, enforcing the quality gate.

6\. Runs application tests with pytest.

7\. Packages and uploads the model and supporting files only if the previous steps succeed.



The tests check that the model loads, a valid sample produces an appropriately shaped prediction, and a missing feature is rejected clearly.



\## 5. Failure and Recovery Experiments



\### Experiment A: Model quality-gate failure



The candidate model was temporarily replaced with a DummyClassifier, making it perform no better than the baseline. The quality gate failed because the required improvement margin was not achieved. The pipeline stopped before running the tests or publishing an artifact.



The original Random Forest model was restored without weakening the quality gate.



\### Experiment B: Application test failure



The model-loading test was temporarily changed to assert that the loaded model was `None`. The test failed as expected, preventing package publication.



The assertion was corrected, and the local test suite subsequently passed all three tests.



\### Final recovery



The Random Forest model and corrected test were restored. The final successful

