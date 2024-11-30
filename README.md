# Diabetes Prediction
CSE 158 Assignment 2

## Question #1: Explanatory Analysis 
  The diabetes_prediction_dataset.csv file contains 100,000 entries of medical and demographic data of patients, indicating whether diabetes is positive, and labeled 1, or negative, labeled 0. The data contains various features such as gender, age, hypertension, heart disease, smoking history, BMI (Body Mass Index), HbA1c (Hemoglobin A1c) levels, and blood glucose levels -- where hypertension and heart disease are indicated present with a 1-label and 0-labeled otherwise. 
  
  These features give insight into how exactly diabetes was diagnosed as features like a higher BMI, HbA1c or blood glucose, for example, are linked closely to a higher risk of diabetes. HbA1c levels, the measure of an individual's average blood sugar level over the past 2-3 months, are considered particularly alarming as an indicator of diabetes if the patient shows more than 6.5%. On the other hand, another factor that suggests a risk for diabetes is a higher BMI, ranging from 18.5-24.9 as normal, 25-29.9 as overweight, and anything past 30 as obese. Interestingly, although these features are linked closely to patients being high-risk, only 36.8% of individuals diagnosed with diabetes exhibit an HbA1c percentage greater than 6.5%, and approximately 11% of individuals diagnosed have BMI of greater than 25, suggesting that other factors could also be playing a role in determining the diagnoses. Additionally, the data indicates that individuals with an age greater than 53 are more closely linked to diabetes, though the average age of the patients listed in the dataset is ~41.


## Question #4:
1. Describe literature related to the problem you are studying. 

2. If you are using an existing dataset, where did it come from and how was it used? 
- The information in our dataset was sourced from Electronic Health Records (EHRs). EHRs are digitized records of patient data that contain large amounts of information, including medical history, diagnoses, treatment plans, and outcomes. EHRs are maintained as standard practice by healthcare providers and a very reliable source of information for a task like ours. The diabetes prediction parameters found in our dataset are tracked and measured by healthcare professionals, making it an optimal source to develop an accurate prediction model. 

3. What other similar datasets have been studied in the past and how? What are the state-of-the-art methods currently employed to study this type
of data? 
- [https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2023.1016381/full
](https://www.researchgate.net/publication/328766758_Predicting_Diabetes_Mellitus_With_Machine_Learning_Techniques)

4. Are the conclusions from existing work similar to or different from your own findings?
