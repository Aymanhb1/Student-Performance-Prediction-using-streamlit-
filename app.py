import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
MODEL_PATH = 'best_student_performance_model.pkl'
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please ensure the model is saved correctly.")
    st.stop()

# Load the training columns
COLUMNS_FILE = 'columns.txt'
try:
    with open(COLUMNS_FILE, 'r') as f:
        training_columns = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    st.error(f"Columns file not found at {COLUMNS_FILE}. Please ensure the columns file is saved correctly.")
    st.stop()

# Load the actual trained scaler
SCALER_PATH = 'trained_scaler.pkl'
try:
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error(f"Scaler file not found at {SCALER_PATH}. Please save your trained scaler using: joblib.dump(scaler, 'trained_scaler.pkl')")
    st.stop()

# App header with improved styling
st.title("🎓 Student Performance Risk Prediction")
st.markdown("*Quick assessment to identify students who may need additional support*")
st.divider()

# Create tabs for better organization
tab1, tab2 = st.tabs(["📋 Student Assessment", "ℹ️ About This Tool"])

with tab1:
    st.markdown("### Fill out this quick form (2-3 minutes)")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("#### 📊 Core Academic Data")
        age = st.number_input("Age", min_value=15, max_value=22, value=16, help="Student's current age")
        studytime = st.selectbox("Weekly Study Time", 
                                options=[1, 2, 3, 4], 
                                format_func=lambda x: {1: "<2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: ">10 hours"}[x],
                                help="Hours spent studying per week")
        absences = st.number_input("Absences", min_value=0, max_value=75, value=0, help="Number of school absences")
        average_grade = st.slider("Average Grade", min_value=0.0, max_value=20.0, value=10.0, step=0.1, help="Current average grade (0-20 scale)")
    
    with col2:
        st.markdown("#### 🎓 Academic Performance")
        attendance_ratio = st.slider("Attendance Ratio", min_value=0.0, max_value=1.0, value=0.95, step=0.01, help="Percentage of classes attended")
        g1 = st.slider("G1 (Period 1 Grade)", min_value=0.0, max_value=20.0, value=10.0, step=0.1, help="First period grade")
        g2 = st.slider("G2 (Period 2 Grade)", min_value=0.0, max_value=20.0, value=10.0, step=0.1, help="Second period grade")
        g3 = st.slider("G3 (Final Grade)", min_value=0.0, max_value=20.0, value=10.0, step=0.1, help="Final grade")
    
    with col3:
        st.markdown("#### 🔥 High-Impact Predictors")
        failures = st.number_input("Past Failures", min_value=0, max_value=4, value=0, help="Number of past class failures (strongest predictor!)")
        medu = st.selectbox("Mother's Education", 
                           options=[0, 1, 2, 3, 4], 
                           format_func=lambda x: {0: "None", 1: "Primary (4th grade)", 2: "5th-9th grade", 3: "Secondary", 4: "Higher education"}[x],
                           help="Mother's education level")
        schoolsup = st.radio("Extra Educational Support", ["No", "Yes"], help="Does student receive extra educational support?")
        
    st.divider()
    
    # Second row for motivational factors
    col4, col5 = st.columns([1, 1])
    
    with col4:
        st.markdown("#### 💡 Motivational Factors")
        higher = st.radio("Wants Higher Education", ["No", "Yes"], index=1, help="Student wants to take higher education")
        internet = st.radio("Internet Access at Home", ["No", "Yes"], index=1, help="Internet access at home")
    
    with col5:
        st.markdown("#### 👨‍👩‍👧‍👦 Family Support")
        famsup = st.radio("Family Educational Support", ["No", "Yes"], index=1, help="Family educational support")
        
        # Add some spacing
        st.write("")
        st.write("")

    # Prediction button
    st.divider()
    predict_btn = st.button("🔮 Predict Risk Category", type="primary", use_container_width=True)

    if predict_btn:
        try:
            # Initialize input data with zeros for all training columns
            input_data = {col: 0.0 for col in training_columns}
            
            # Map the form inputs to the model's expected features
            # Core Academic Data
            if 'age' in training_columns:
                input_data['age'] = float(age)
            if 'studytime' in training_columns:
                input_data['studytime'] = float(studytime)
            if 'absences' in training_columns:
                input_data['absences'] = float(absences)
            if 'Average Grade' in training_columns:
                input_data['Average Grade'] = float(average_grade)
            
            # Academic Performance
            if 'Attendance Ratio' in training_columns:
                input_data['Attendance Ratio'] = float(attendance_ratio)
            if 'G1' in training_columns:
                input_data['G1'] = float(g1)
            if 'G2' in training_columns:
                input_data['G2'] = float(g2)
            if 'G3' in training_columns:
                input_data['G3'] = float(g3)
            
            # High-Impact Predictors
            if 'failures' in training_columns:
                input_data['failures'] = float(failures)
            if 'Medu' in training_columns:
                input_data['Medu'] = float(medu)
            if 'schoolsup_yes' in training_columns:
                input_data['schoolsup_yes'] = 1.0 if schoolsup == "Yes" else 0.0
            
            # Motivational Factors
            if 'higher_yes' in training_columns:
                input_data['higher_yes'] = 1.0 if higher == "Yes" else 0.0
            if 'internet_yes' in training_columns:
                input_data['internet_yes'] = 1.0 if internet == "Yes" else 0.0
            if 'famsup_yes' in training_columns:
                input_data['famsup_yes'] = 1.0 if famsup == "Yes" else 0.0

            # Create DataFrame with correct column order
            input_df = pd.DataFrame([input_data])[training_columns]

            # Scale the input data
            input_scaled = scaler.transform(input_df)

            # Make prediction
            prediction = model.predict(input_scaled)
            
            # Get prediction probability if available
            try:
                prediction_proba = model.predict_proba(input_scaled)
                confidence = np.max(prediction_proba) * 100
            except:
                confidence = None

            # Display results with enhanced formatting
            st.success("✅ Prediction Complete!")
            
            # Create result display
            result_col1, result_col2 = st.columns([2, 1])
            
            with result_col1:
                st.markdown("### 🎯 Risk Assessment Result:")
                
                # Color code the results
                risk_colors = {
                    "High Risk": "🔴",
                    "Medium Risk": "🟡", 
                    "Low Risk": "🟢",
                    "Very High Risk": "🔴",
                    "Very Low Risk": "🟢"
                }
                
                risk_emoji = risk_colors.get(prediction[0], "⚪")
                st.markdown(f"## {risk_emoji} **{prediction[0]}**")
                
                if confidence:
                    st.markdown(f"*Confidence: {confidence:.1f}%*")
            
            with result_col2:
                # Add some contextual information
                if "High Risk" in prediction[0]:
                    st.warning("⚠️ **Intervention Recommended**\n\nThis student may benefit from additional support and monitoring.")
                elif "Medium Risk" in prediction[0]:
                    st.info("📊 **Monitor Progress**\n\nKeep track of this student's performance and provide support as needed.")
                else:
                    st.success("✅ **On Track**\n\nStudent appears to be performing well academically.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.write("**Debug Information:**")
            st.write(f"Number of features provided: {len(input_data)}")
            st.write(f"Expected features: {len(training_columns)}")

with tab2:
    st.markdown("""
    ### About This Prediction Tool
    
    This application uses machine learning to assess student performance risk based on research-backed predictors.
    
    #### 🎯 **Key Predictive Features:**
    
    **Strongest Predictors:**
    - **Past Failures** - Most important factor for identifying at-risk students
    - **Mother's Education** - Strong indicator of family academic background
    - **Extra School Support** - Shows if interventions are already in place
    
    **Academic Performance:**
    - Current grades and attendance patterns
    - Study habits and time investment
    
    **Support Systems:**
    - Family educational support
    - Access to resources (internet)
    - Educational aspirations
    
    #### 📈 **How It Works:**
    1. Enter student information in the form
    2. The model analyzes patterns based on historical data
    3. Receive a risk assessment with confidence level
    4. Use results to guide intervention decisions
    
    #### ⚡ **Quick & Accurate:**
    - Takes only 2-3 minutes to complete
    - Focuses on the most predictive factors
    - Based on validated research
    
    #### 🎯 **Risk Categories:**
    - **🟢 Low Risk:** Student performing well, minimal intervention needed
    - **🟡 Medium Risk:** Monitor progress, provide support as needed  
    - **🔴 High Risk:** Intervention recommended, additional support beneficial
    
    *Note: This tool is designed to assist educators in identifying students who may benefit from additional support. 
    It should be used as part of a comprehensive assessment process.*
    """)
