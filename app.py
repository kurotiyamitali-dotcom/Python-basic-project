import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

model = joblib.load("employee_attrition_model.pkl")
scaler = joblib.load("employee_attrition_scaler.pkl")
columns = joblib.load("employee_columns.pkl")
encoders = joblib.load("employee_encoders.pkl")

st.markdown("""
<style>

.title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#0E76A8;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
}

.stButton button{
    width:100%;
    height:55px;
    font-size:20px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    "<div class='title'>👨‍💼 Employee Attrition Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Machine Learning based Employee Leave Prediction System</div>",
    unsafe_allow_html=True
)

st.divider()


st.sidebar.title("About Project")

st.sidebar.info(
"""
Employee Attrition Prediction

✔ Random Forest Algorithm

✔ Machine Learning Model

✔ Streamlit Application

✔ Python Based Project
"""
)


left, right = st.columns(2)

data = {}


with left:

    st.subheader("👤 Employee Information")


    data["Age"] = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )


    data["BusinessTravel"] = st.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )


    data["DailyRate"] = st.number_input(
        "Daily Rate",
        value=800
    )


    data["Department"] = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )


    data["DistanceFromHome"] = st.number_input(
        "Distance From Home",
        value=5
    )


    data["Education"] = st.selectbox(
        "Education",
        [1,2,3,4,5]
    )


    data["EducationField"] = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )


    data["Gender"] = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )


    data["JobRole"] = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )


with right:

    st.subheader("💼 Job Information")


    data["JobLevel"] = st.selectbox(
        "Job Level",
        [1,2,3,4,5]
    )


    data["JobSatisfaction"] = st.selectbox(
        "Job Satisfaction",
        [1,2,3,4]
    )


    data["MaritalStatus"] = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )


    data["MonthlyIncome"] = st.number_input(
        "Monthly Income",
        value=5000
    )


    data["NumCompaniesWorked"] = st.number_input(
        "Number of Companies Worked",
        value=2
    )


    data["OverTime"] = st.selectbox(
        "Over Time",
        [
            "Yes",
            "No"
        ]
    )


    data["PercentSalaryHike"] = st.number_input(
        "Percent Salary Hike",
        value=15
    )


    data["PerformanceRating"] = st.selectbox(
        "Performance Rating",
        [1,2,3,4]
    )


    data["YearsAtCompany"] = st.number_input(
        "Years At Company",
        value=5
    )


    data["YearsInCurrentRole"] = st.number_input(
        "Years In Current Role",
        value=3
    )


input_df = pd.DataFrame([data])


if st.button("🔮 Predict Attrition"):


    processed_df = input_df.copy()


    for col in columns:

        if col not in processed_df.columns:
            processed_df[col] = 0


    processed_df = processed_df[columns]


    for col in processed_df.columns:

        if col in encoders:

            try:

                processed_df[col] = encoders[col].transform(
                    processed_df[col].astype(str)
                )

            except Exception:

                processed_df[col] = 0



    for col in processed_df.columns:

        processed_df[col] = pd.to_numeric(
            processed_df[col],
            errors="coerce"
        )


    processed_df = processed_df.fillna(0)



    input_scaled = scaler.transform(processed_df)


    prediction = model.predict(input_scaled)


    probability = model.predict_proba(input_scaled)[0][1]



    st.divider()


    if prediction[0] == 1:


        st.error(
            "⚠ Employee is likely to leave the company"
        )


        st.metric(
            "Attrition Probability",
            f"{probability*100:.2f}%"
        )


    else:


        st.success(
            "✅ Employee is likely to stay in the company"
        )


        st.metric(
            "Retention Probability",
            f"{(1-probability)*100:.2f}%"
        )


    with st.expander("📌 Model Information"):

        st.write(
        """
        **Algorithm:** Random Forest Classifier

        **Project:** Employee Attrition Prediction

        **Technology Used:**
        - Python
        - Pandas
        - Scikit-Learn
        - Streamlit
        - Joblib
        """
        )


    with st.expander("👀 View Employee Data"):

        st.dataframe(input_df)


    report = pd.DataFrame(
        {
            "Feature": processed_df.columns,
            "Value": processed_df.iloc[0].values
        }
    )


    csv = report.to_csv(index=False)


    st.download_button(
        label="⬇ Download Prediction Report",
        data=csv,
        file_name="Employee_Attrition_Report.csv",
        mime="text/csv"
    )



else:

    st.info(
        "Click Predict Attrition button to generate result."
    )


st.divider()

st.markdown(
"""
<center>

<h4>Employee Attrition Prediction System</h4>

<p>
Developed using Machine Learning & Streamlit
</p>

</center>
""",
unsafe_allow_html=True
)