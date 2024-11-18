import streamlit as st
import pandas as pd
from midtermCheckpoint import midtermPage
from final import final

def main():
    st.title("Machine Learning Project")
    
    page = st.sidebar.selectbox("Select a page:", ["Home", "Proposal", "Midterm", "Final Submission"])
    
    if page == "Home":
        st.write("Welcome to our Machine Learning Project!")
    elif page == "Proposal":
        proposal_page()
    elif page == "Midterm":
        midtermPage()
    elif page == "Final Submission":
        final()


def proposal_page():
    st.title("Proposal")
    
    st.header("Introduction/Background")
    st.write(
        "Predicting how well students will do in school is important because it helps identify pain points where the education system can improve. "
        "Machine learning results like in this case can help provide practical information on how teachers can give extra support to students who need it or predict final grades using information like past grades, family background, and student activities. "
        "Previous studies have used different machine learning methods like decision trees, random forests, and neural networks to predict how students will perform. "
        "For example, **Romero and Ventura (2013)** used decision trees to study student data, while **Almeida et al. (2017)** showed that random forests are very accurate for this task."
    )
    
    st.write(
        "For this project, our group is looking into using the 'Student Performance Dataset' from the UCI Machine Learning Repository. "
        "This dataset has information on 395 students, including their past grades, family details, and extracurricular activities. We aim to use machine learning to predict final grades."
    )

    st.markdown("**Dataset Link:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)")

    st.header("Problem Definition")
    st.write(
        "The main problem is to predict how well students will perform so that schools can find students who are at risk of failing. "
        "By predicting final grades early, we can identify students who may struggle early on and can help reduce dropout rates, improve learning, and personalize teaching."
    )

    st.header("Methods")
    st.write(
        "The first data preprocessing method we would like to use is to check for missing values in the dataset. "
        "We can use the **SimpleImputer** function to substitute these missing values in the dataset. "
        "The second data preprocessing method we plan to utilize is **one-hot encoding** to help represent fields such as ‘school’ and ‘address’ in our dataset as binary encodings. "
        "The third data preprocessing method we plan to implement is feature selection to lower the dimensions of our dataset and analyze only the features which are useful to our goal. "
        "**SelectKBest()** would allow us to gain insight on which features would be statistically relevant."
    )

    st.write(
        "The first machine learning algorithm we are considering is decision trees, which can handle both numerical and categorical data from our dataset and will be able to nonlinearly represent our data. "
        "The second machine learning algorithm we are hoping to use is random forests, which creates multiple decision trees, aggregates results, and uses mixed data types. "
        "The third machine learning algorithm we are hoping to implement is a support vector machine, or **SVM**, which is useful for high-dimensional data and classification."
    )

    st.write(
        "We have decided on random forests for our supervised learning method. As mentioned earlier, we would be able to effectively handle a variety of data types and high-dimension data. "
        "We would be able to create a robust and accurate learning framework with random forests."
    )

    st.header("Potential Results and Discussion")
    st.write(
        "The quantitative metrics we would like to include are negative mean absolute error between predicted and actual student grades, **R^2**, and the negative root mean square error that can measure the standard deviation. "
        "Another good metric would be to use the accuracy score for classification."
    )

    st.write(
        "For the project, we expect that key predictors of student performance will be past academic records, performance, time, and parental education. "
        "Our project goals include developing a regression model that predicts the student’s performance with an **R squared** of at least 0.7 and a Negative Root Mean Square Error that is over -2.0 points."
    )

    st.header("References")
    st.write(
        "**Romero, C., & Ventura, S. (2013).** Data mining in education. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 3(1), 12-27. "
        "[https://doi.org/10.1002/widm.1075](https://doi.org/10.1002/widm.1075)"
    )
    
    st.write(
        "**J. Almeida, J. X. Siqueira,** Random forest for student performance prediction, 2017 IEEE International Conference on Data Science and Advanced Analytics (DSAA), pp.15 - 21, 2017."
    )

    st.write(
        "**L. Cortez, A. Silva,** Using Data Mining to Predict Secondary School Student Performance, Proceedings of 5th Future Business Technology Conference, 2008."
    )

    st.write(
        "**B. Kovacic,** Predicting student success by mining enrolment data, Research in Higher Education Journal, vol. 4, pp. 1 - 20, 2012."
    )

    st.write(
        "**P. Brasil, L. Munoz,** Machine learning applied to predicting student performance, Proceedings of the 11th International Conference on Educational Data Mining, 2018."
    )

if __name__ == "__main__":
    main()