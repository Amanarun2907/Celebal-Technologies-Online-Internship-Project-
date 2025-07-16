
# # streamlit run app.py

# # app.py
# import streamlit as st
# import joblib
# import pandas as pd
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# # Import necessary metrics from sklearn
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # --- Configuration & Styling ---
# st.set_page_config(
#     page_title="Advanced CLTV Prediction Dashboard",
#     page_icon="📈",
#     layout="wide", # Use wide layout for a dashboard feel
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for a fantastic, creative, and smart look
# st.markdown(
#     """
#     <style>
#     /* General Styling */
#     body {
#         font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
#         color: #333;
#         background-color: #f0f2f6;
#     }

#     /* Main Header */
#     .main-header {
#         font-size: 3.8em; /* Slightly larger */
#         font-weight: bold;
#         color: #FF6347; /* Tomato Red */
#         text-align: center;
#         text-shadow: 3px 3px 6px rgba(0,0,0,0.3); /* Stronger shadow */
#         margin-bottom: 0.6em;
#         letter-spacing: 1.5px;
#         line-height: 1.2;
#     }

#     /* Subheader */
#     .subheader {
#         font-size: 1.8em; /* Slightly larger */
#         color: #2E8B57; /* Sea Green */
#         text-align: center;
#         margin-bottom: 2.5em;
#         font-style: italic;
#         padding-bottom: 10px;
#         border-bottom: 2px solid #ddd;
#     }

#     /* Buttons */
#     .stButton>button {
#         background-color: #4CAF50; /* Green */
#         color: white;
#         font-size: 1.4em; /* Larger font */
#         padding: 15px 30px; /* More padding */
#         border-radius: 12px; /* More rounded */
#         border: none;
#         cursor: pointer;
#         transition: all 0.3s ease-in-out;
#         box-shadow: 3px 3px 8px rgba(0,0,0,0.25); /* More prominent shadow */
#     }
#     .stButton>button:hover {
#         background-color: #45a049;
#         transform: translateY(-5px); /* Lift effect */
#         box-shadow: 6px 6px 15px rgba(0,0,0,0.4); /* Stronger hover shadow */
#     }

#     /* Prediction Box */
#     .prediction-box {
#         background-color: #e6f7ff; /* Light Blue Background */
#         border-left: 12px solid #FF6347; /* Thicker Red Accent */
#         padding: 30px; /* More padding */
#         border-radius: 20px; /* More rounded */
#         margin-top: 3em;
#         box-shadow: 0 10px 20px rgba(0,0,0,0.25); /* Deeper shadow */
#         animation: fadeIn 1.2s ease-out; /* Slower fade-in */
#     }
#     .prediction-text {
#         font-size: 2.8em; /* Larger */
#         font-weight: bold;
#         color: #333333;
#         text-align: center;
#         margin-bottom: 0.3em;
#     }
#     .predicted-value {
#         font-size: 4.5em; /* Significantly larger */
#         font-weight: bolder;
#         color: #007bff; /* Royal Blue */
#         text-align: center;
#         text-shadow: 2px 2px 5px rgba(0,0,0,0.2); /* Added shadow */
#         margin-top: 0.2em;
#     }

#     /* Input Headers */
#     .input-header {
#         font-size: 2em; /* Larger */
#         color: #333333;
#         font-weight: bold;
#         margin-top: 2em;
#         margin-bottom: 1.2em;
#         border-bottom: 3px solid #ddd; /* Thicker border */
#         padding-bottom: 8px;
#     }

#     /* Animation */
#     @keyframes fadeIn {
#         from { opacity: 0; transform: translateY(30px); }
#         to { opacity: 1; transform: translateY(0); }
#     }

#     /* Info/Warning Boxes */
#     .stAlert {
#         border-radius: 8px;
#         padding: 15px;
#         margin-bottom: 1.5em;
#     }

#     /* Metric Cards */
#     .metric-card {
#         background-color: #ffffff;
#         border-radius: 10px;
#         padding: 20px;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#         text-align: center;
#         margin-bottom: 20px;
#     }
#     .metric-title {
#         font-size: 1.2em;
#         color: #555;
#         margin-bottom: 5px;
#     }
#     .metric-value {
#         font-size: 2em;
#         font-weight: bold;
#         color: #4CAF50;
#     }
#     .metric-r2 .metric-value {
#         color: #007bff;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # --- Path Definitions ---
# MODEL_DIR = 'Saving_The_Best_Model_Resources'
# MODEL_PATH = os.path.join(MODEL_DIR, 'gradient_boosting_sklearn_model.joblib')
# SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
# Y_TEST_PATH = os.path.join(MODEL_DIR, 'y_test.joblib')
# Y_PRED_GB_PATH = os.path.join(MODEL_DIR, 'y_pred_gb.joblib')
# FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.joblib')
# RFM_DATA_PATH = os.path.join('data', 'rfm_features_enhanced.xlsx')

# # --- Load All Resources (Cached) ---
# @st.cache_resource
# def load_all_resources():
#     try:
#         model = joblib.load(MODEL_PATH)
#         scaler = joblib.load(SCALER_PATH)
#         y_test_loaded = joblib.load(Y_TEST_PATH)
#         y_pred_gb_loaded = joblib.load(Y_PRED_GB_PATH)
#         feature_names_loaded = joblib.load(FEATURE_NAMES_PATH)
#         full_rfm_df = pd.read_excel(RFM_DATA_PATH)

#         # Reconstruct known_country_columns and unique_countries_from_features
#         known_general_features = ['Recency', 'Frequency', 'AOV', 'Tenure', 'UniqueProducts']
#         known_country_columns = [col for col in feature_names_loaded if col.startswith('Country_')]
        
#         # Ensure that `features_for_model` matches the exact order of features
#         # X.columns when the model and scaler were trained.
#         # feature_names_loaded contains exactly this order.
#         features_for_model = feature_names_loaded

#         unique_countries_from_features = [col.replace('Country_', '') for col in known_country_columns]
#         unique_countries_from_features.sort()
#         unique_countries_from_features.insert(0, 'Select Country')

#         return model, scaler, y_test_loaded, y_pred_gb_loaded, features_for_model, unique_countries_from_features, known_country_columns, full_rfm_df

#     except FileNotFoundError as e:
#         st.error(f"Error: Missing a required file. Please ensure all saved resources are in '{MODEL_DIR}' and `rfm_features_enhanced.xlsx` is in 'data'. Missing: {e.filename}")
#         st.stop()
#     except Exception as e:
#         st.error(f"An unexpected error occurred during resource loading: {e}")
#         st.stop()

# model, scaler, y_test, y_pred_gb, features_for_model, unique_countries_from_features, known_country_columns, full_rfm_df = load_all_resources()

# # --- App Title and Introduction ---
# st.markdown('<p class="main-header">🌟 Customer Lifetime Value (CLTV) Predictor Dashboard 🌟</p>', unsafe_allow_html=True)
# st.markdown('<p class="subheader">Predict, Analyze, and Understand Your Customer\'s Future Value</p>', unsafe_allow_html=True)

# st.success("All models and data resources loaded successfully! Dive into insights and predictions.")

# # --- Navigation Tabs ---
# tab1, tab2, tab3 = st.tabs(["🚀 Predict CLTV", "📊 Model Performance & Insights", "🔍 Data Explorer"])

# # --- Tab 1: Predict CLTV ---
# with tab1:
#     st.markdown('<p class="input-header">Enter Customer Data for Prediction</p>', unsafe_allow_html=True)

#     col1, col2 = st.columns(2)

#     with col1:
#         recency = st.number_input("Recency (Days since last purchase):", min_value=0, value=30, help="Number of days since the customer's last purchase. Lower is generally better.")
#         frequency = st.number_input("Frequency (Total number of purchases):", min_value=1, value=5, help="Total number of unique transactions made by the customer. Higher is better.")
#         aov = st.number_input("Average Order Value (Average spend per purchase):", min_value=0.0, value=50.0, format="%.2f", help="Average monetary value per transaction for this customer.")

#     with col2:
#         tenure = st.number_input("Tenure (Days since first purchase):", min_value=0, value=365, help="Number of days since the customer's very first purchase. Longer tenure implies more loyalty.")
#         unique_products = st.number_input("Unique Products (Number of different items bought):", min_value=1, value=10, help="Diversity of products purchased by the customer. A wider range might indicate broader interest.")
#         selected_country = st.selectbox("Customer's Country:", unique_countries_from_features, help="The primary country of the customer, used for regional insights by the model.")

#     st.write("---")

#     # Prediction Button
#     if st.button("✨ Get CLTV Prediction"):
#         if selected_country == 'Select Country':
#             st.warning("Please select a valid country to get a prediction.")
#         else:
#             # Prepare Input Data for Prediction
#             input_data = {
#                 'Recency': recency,
#                 'Frequency': frequency,
#                 'AOV': aov,
#                 'Tenure': tenure,
#                 'UniqueProducts': unique_products
#             }

#             # Add one-hot encoded country features, initializing all to 0
#             for country_col in known_country_columns:
#                 input_data[country_col] = 0

#             # Set the selected country's column to 1
#             one_hot_col_name = f'Country_{selected_country}'
#             if one_hot_col_name in input_data:
#                 input_data[one_hot_col_name] = 1
#             else:
#                 st.warning(f"Note: Selected country '{selected_country}' was not specifically present in the training data's country features. Its contribution will be generalized.")

#             # Create a DataFrame with the exact column order as `features_for_model`
#             input_df = pd.DataFrame([input_data])
#             input_df = input_df[features_for_model] # CRUCIAL for order consistency

#             # Scale the input data using the loaded scaler
#             scaled_input = scaler.transform(input_df)

#             # Make Prediction
#             predicted_cltv = model.predict(scaled_input)[0]

#             # Display Prediction with an engaging message
#             st.markdown(
#                 f"""
#                 <div class="prediction-box">
#                     <p class="prediction-text">The Estimated CLTV is:</p>
#                     <p class="predicted-value">${predicted_cltv:.2f}</p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
#             st.info("💡 **Insight**: This prediction represents the estimated total monetary value a customer is expected to bring to your business over their lifetime, based on the provided inputs and our powerful Gradient Boosting model.")


# # --- Tab 2: Model Performance & Insights ---
# with tab2:
#     st.markdown('<p class="input-header">Model Performance & Key Insights</p>', unsafe_allow_html=True)

#     # Calculate metrics
#     mae = mean_absolute_error(y_test, y_pred_gb)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
#     r2 = r2_score(y_test, y_pred_gb)

#     st.subheader("Performance Metrics (Gradient Boosting Regressor):")
#     metric_col1, metric_col2, metric_col3 = st.columns(3)
#     with metric_col1:
#         st.markdown(f'<div class="metric-card"><p class="metric-title">Mean Absolute Error (MAE)</p><p class="metric-value">${mae:.2f}</p></div>', unsafe_allow_html=True)
#     with metric_col2:
#         st.markdown(f'<div class="metric-card"><p class="metric-title">Root Mean Squared Error (RMSE)</p><p class="metric-value">${rmse:.2f}</p></div>', unsafe_allow_html=True)
#     with metric_col3:
#         st.markdown(f'<div class="metric-card metric-r2"><p class="metric-title">R-squared (R²)</p><p class="metric-value">{r2:.4f}</p></div>', unsafe_allow_html=True)

#     st.write("---")

#     # --- Visualizations ---
#     st.subheader("Model Insights Visualizations:")

#     # 1. Feature Importance
#     if hasattr(model, 'feature_importances_') and features_for_model is not None:
#         st.markdown("#### Feature Importance: What Drives CLTV?")
#         feature_importances_gb = pd.Series(model.feature_importances_, index=features_for_model).sort_values(ascending=False)

#         top_n_features = 20 # Display top 20 features
#         top_features_to_plot = feature_importances_gb.head(top_n_features)

#         fig_fi, ax_fi = plt.subplots(figsize=(12, max(8, top_n_features * 0.4)))
#         sns.barplot(x=top_features_to_plot.values, y=top_features_to_plot.index, palette='viridis', ax=ax_fi)
#         ax_fi.set_title(f'Gradient Boosting Regressor: Top {top_n_features} Feature Importances', fontsize=16)
#         ax_fi.set_xlabel('Importance (Scaled)', fontsize=12)
#         ax_fi.set_ylabel('Feature', fontsize=12)
#         ax_fi.tick_params(axis='x', labelsize=10)
#         ax_fi.tick_params(axis='y', labelsize=10)
#         ax_fi.grid(axis='x', linestyle='--', alpha=0.7)
#         fig_fi.tight_layout()
#         st.pyplot(fig_fi)
#         plt.close(fig_fi) # Close figure to prevent display issues

#         st.markdown("""
#         **Inference from Feature Importance:**
#         This chart reveals which customer attributes the model considers most crucial in predicting CLTV.
#         Typically, features like **Recency** (how recently they purchased), **Frequency** (how often), and **Average Order Value (AOV)**
#         are strong indicators. Country-specific factors also play a role, highlighting regional differences in customer behavior.
#         Focusing on improving these high-importance metrics can significantly boost overall CLTV.
#         """)
#         st.write("---")
#     else:
#         st.warning("Feature importances are not available for this model type or could not be loaded.")


#     # 2. Actual vs. Predicted Plot
#     st.markdown("#### Actual vs. Predicted CLTV: How Well Does the Model Perform?")
#     fig_ap, ax_ap = plt.subplots(figsize=(10, 7))
#     sns.scatterplot(x=y_test, y=y_pred_gb, alpha=0.6, ax=ax_ap)
#     ax_ap.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
#     ax_ap.set_title('Gradient Boosting Regressor: Actual vs. Predicted Monetary Value', fontsize=16)
#     ax_ap.set_xlabel('Actual Monetary Value (CLTV)', fontsize=12)
#     ax_ap.set_ylabel('Predicted Monetary Value (CLTV)', fontsize=12)
#     ax_ap.grid(True, linestyle=':', alpha=0.7)
#     ax_ap.legend()
#     fig_ap.tight_layout()
#     st.pyplot(fig_ap)
#     plt.close(fig_ap)

#     st.markdown("""
#     **Inference from Actual vs. Predicted Plot:**
#     The closer the scatter points are to the red dashed line, the more accurate our predictions are.
#     A good model will show points clustered along this line. Deviations indicate where the model
#     might be overestimating or underestimating CLTV. Points far from the line (outliers) might be
#     challenging cases for the model.
#     """)
#     st.write("---")

#     # 3. Residuals Distribution
#     st.markdown("#### Residuals Distribution: Are Our Errors Random?")
#     residuals = y_test - y_pred_gb
#     fig_res, ax_res = plt.subplots(figsize=(10, 7))
#     sns.histplot(residuals, kde=True, ax=ax_res, color='skyblue', bins=50)
#     ax_res.set_title('Gradient Boosting Regressor: Residuals Distribution', fontsize=16)
#     ax_res.set_xlabel('Residuals (Actual CLTV - Predicted CLTV)', fontsize=12)
#     ax_res.set_ylabel('Count', fontsize=12)
#     ax_res.axvline(0, color='red', linestyle='--', label='Zero Residuals')
#     ax_res.legend()
#     ax_res.grid(True, linestyle=':', alpha=0.7)
#     fig_res.tight_layout()
#     st.pyplot(fig_res)
#     plt.close(fig_res)

#     st.markdown("""
#     **Inference from Residuals Distribution:**
#     Ideally, residuals should be normally distributed around zero, indicating that the model's errors are random and unbiased.
#     If the distribution is skewed or has a clear pattern, it might suggest that the model is systematically
#     over-predicting or under-predicting for certain segments of customers.
#     """)
#     st.write("---")

# # --- Tab 3: Data Explorer ---
# with tab3:
#     st.markdown('<p class="input-header">Explore the Underlying Data (rfm_features_enhanced)</p>', unsafe_allow_html=True)

#     st.subheader("Dataset Overview:")
#     st.write("This section allows you to explore the characteristics of the data used to train the model.")

#     st.markdown("#### Descriptive Statistics:")
#     st.dataframe(full_rfm_df.drop(columns=[col for col in full_rfm_df.columns if col.startswith('Country_') or col == 'CustomerID']).describe().transpose().round(2))
#     st.markdown("""
#     **Inference from Descriptive Statistics:**
#     This table provides a quick summary of your numerical features.
#     Look at the mean, standard deviation, min, max, and quartiles to understand the distribution and potential outliers.
#     For example, a large difference between mean and median (50% percentile) can indicate skewness.
#     """)
#     st.write("---")

#     st.markdown("#### Distribution of Key Features:")

#     # Plot distributions of key RFM-related features
#     plot_cols = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Tenure', 'UniqueProducts']
#     num_plots = len(plot_cols)
#     cols_per_row = 3
#     rows = int(np.ceil(num_plots / cols_per_row))

#     for r in range(rows):
#         plot_cols_row = st.columns(cols_per_row)
#         for i in range(cols_per_row):
#             idx = r * cols_per_row + i
#             if idx < num_plots:
#                 col_name = plot_cols[idx]
#                 with plot_cols_row[i]:
#                     fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
#                     sns.histplot(full_rfm_df[col_name], kde=True, ax=ax_dist, color=plt.cm.get_cmap('Set2')(i))
#                     ax_dist.set_title(f'Distribution of {col_name}', fontsize=14)
#                     ax_dist.set_xlabel(col_name, fontsize=10)
#                     ax_dist.set_ylabel('Count', fontsize=10)
#                     ax_dist.tick_params(axis='x', labelsize=8)
#                     ax_dist.tick_params(axis='y', labelsize=8)
#                     fig_dist.tight_layout()
#                     st.pyplot(fig_dist)
#                     plt.close(fig_dist)

#     st.markdown("""
#     **Inference from Feature Distributions:**
#     These histograms (with KDE) show the shape of your data for each key feature.
#     - **Skewness:** Many distributions are likely right-skewed (e.g., Monetary, Frequency), meaning most customers have lower values, with a few high-value customers. This is typical for customer data.
#     - **Outliers:** Long tails or isolated bars indicate potential outliers that might have influenced the model.
#     - **Data Cleaning:** Visualizing these confirms the effectiveness of any data cleaning or transformation steps applied during feature engineering.
#     """)
#     st.write("---")

#     st.markdown("#### Top Countries by Customer Count:")
#     country_counts = {}
#     for col in full_rfm_df.columns:
#         if col.startswith('Country_'):
#             country_name = col.replace('Country_', '')
#             country_counts[country_name] = full_rfm_df[col].sum()

#     if country_counts:
#         top_countries_df = pd.Series(country_counts).sort_values(ascending=False).head(10) # Top 10 countries
#         fig_country, ax_country = plt.subplots(figsize=(10, 6))
#         sns.barplot(x=top_countries_df.values, y=top_countries_df.index, palette='coolwarm', ax=ax_country)
#         ax_country.set_title('Top 10 Countries by Customer Count', fontsize=16)
#         ax_country.set_xlabel('Number of Customers', fontsize=12)
#         ax_country.set_ylabel('Country', fontsize=12)
#         ax_country.tick_params(axis='x', labelsize=10)
#         ax_country.tick_params(axis='y', labelsize=10)
#         ax_country.grid(axis='x', linestyle='--', alpha=0.7)
#         fig_country.tight_layout()
#         st.pyplot(fig_country)
#         plt.close(fig_country)
#         st.markdown("""
#         **Inference from Top Countries:**
#         This chart highlights the countries with the largest customer bases in your dataset.
#         Understanding your primary markets is crucial for targeted marketing and strategic planning.
#         You can see where your current customer concentration lies.
#         """)
#     else:
#         st.info("Country distribution insights require 'Country_' columns in the data.")


# # --- Sidebar ---
# st.sidebar.header("About This CLTV Dashboard")
# st.sidebar.info(
#     "This interactive dashboard is designed to empower businesses with deeper insights into "
#     "their customer base. It combines state-of-the-art machine learning (Gradient Boosting) "
#     "for **CLTV Prediction** with rich **Model Performance Visualizations** and a comprehensive "
#     "**Data Explorer** to analyze the underlying customer attributes."
# )
# st.sidebar.markdown("---")
# st.sidebar.subheader("Key Features:")
# st.sidebar.markdown("""
# - **Predictive Analytics**: Estimate future customer value.
# - **Model Transparency**: Understand model strengths and weaknesses.
# - **Data Understanding**: Explore distributions and patterns in your customer data.
# """)
# st.sidebar.markdown("---")
# st.sidebar.write("Developed for Strategic Customer Relationship Management.")
# st.sidebar.write("© 2025 Your Company/Project Name") # Replace with your name/project

# app.py
import streamlit as st
import joblib
import pandas as pd
import os # Import the os module for path operations and directory listing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Import necessary metrics from sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Advanced CLTV Prediction Dashboard",
    page_icon="📈",
    layout="wide", # Use wide layout for a dashboard feel
    initial_sidebar_state="expanded"
)

# Custom CSS for a fantastic, creative, and smart look
st.markdown(
    """
    <style>
    /* General Styling */
    body {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        color: #333;
        background-color: #f0f2f6;
    }

    /* Main Header */
    .main-header {
        font-size: 3.8em; /* Slightly larger */
        font-weight: bold;
        color: #FF6347; /* Tomato Red */
        text-align: center;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3); /* Stronger shadow */
        margin-bottom: 0.6em;
        letter-spacing: 1.5px;
        line-height: 1.2;
    }

    /* Subheader */
    .subheader {
        font-size: 1.8em; /* Slightly larger */
        color: #2E8B57; /* Sea Green */
        text-align: center;
        margin-bottom: 2.5em;
        font-style: italic;
        padding-bottom: 10px;
        border-bottom: 2px solid #ddd;
    }

    /* Buttons */
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        font-size: 1.4em; /* Larger font */
        padding: 15px 30px; /* More padding */
        border-radius: 12px; /* More rounded */
        border: none;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
        box-shadow: 3px 3px 8px rgba(0,0,0,0.25); /* More prominent shadow */
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-5px); /* Lift effect */
        box-shadow: 6px 6px 15px rgba(0,0,0,0.4); /* Stronger hover shadow */
    }

    /* Prediction Box */
    .prediction-box {
        background-color: #e6f7ff; /* Light Blue Background */
        border-left: 12px solid #FF6347; /* Thicker Red Accent */
        padding: 30px; /* More padding */
        border-radius: 20px; /* More rounded */
        margin-top: 3em;
        box-shadow: 0 10px 20px rgba(0,0,0,0.25); /* Deeper shadow */
        animation: fadeIn 1.2s ease-out; /* Slower fade-in */
    }
    .prediction-text {
        font-size: 2.8em; /* Larger */
        font-weight: bold;
        color: #333333;
        text-align: center;
        margin-bottom: 0.3em;
    }
    .predicted-value {
        font-size: 4.5em; /* Significantly larger */
        font-weight: bolder;
        color: #007bff; /* Royal Blue */
        text-align: center;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.2); /* Added shadow */
        margin-top: 0.2em;
    }

    /* Input Headers */
    .input-header {
        font-size: 2em; /* Larger */
        color: #333333;
        font-weight: bold;
        margin-top: 2em;
        margin-bottom: 1.2em;
        border-bottom: 3px solid #ddd; /* Thicker border */
        padding-bottom: 8px;
    }

    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Info/Warning Boxes */
    .stAlert {
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 1.5em;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 1.2em;
        color: #555;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-r2 .metric-value {
        color: #007bff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- DEBUGGING INFORMATION ---
st.header("Debugging Information (Temporary)")

# Print current working directory
current_working_directory = os.getcwd()
st.write(f"Current Working Directory: `{current_working_directory}`")

# List contents of the current directory
st.subheader("Contents of Current Working Directory:")
try:
    current_dir_contents = os.listdir(current_working_directory)
    for item in current_dir_contents:
        st.write(f"- `{item}`")
except Exception as e:
    st.error(f"Error listing current directory: {e}")

# List contents of 'Saving_The_Best_Model_Resources'
st.subheader("Contents of Saving_The_Best_Model_Resources:")
try:
    model_resources_path = os.path.join(current_working_directory, 'Saving_The_Best_Model_Resources')
    if os.path.exists(model_resources_path):
        model_dir_contents = os.listdir(model_resources_path)
        for item in model_dir_contents:
            st.write(f"- `{item}`")
    else:
        st.error(f"Path does not exist: `{model_resources_path}`")
except Exception as e:
    st.error(f"Error listing model resources directory: {e}")

# List contents of 'data'
st.subheader("Contents of data directory:")
try:
    data_path = os.path.join(current_working_directory, 'data')
    if os.path.exists(data_path):
        data_dir_contents = os.listdir(data_path)
        for item in data_dir_contents:
            st.write(f"- `{item}`")
    else:
        st.error(f"Path does not exist: `{data_path}`")
except Exception as e:
    st.error(f"Error listing data directory: {e}")

st.header("End Debugging Information")
# --- END DEBUGGING INFORMATION ---


# --- Path Definitions ---
MODEL_DIR = 'Saving_The_Best_Model_Resources'
MODEL_PATH = os.path.join(MODEL_DIR, 'gradient_boosting_sklearn_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
Y_TEST_PATH = os.path.join(MODEL_DIR, 'y_test.joblib')
Y_PRED_GB_PATH = os.path.join(MODEL_DIR, 'y_pred_gb.joblib')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.joblib')
RFM_DATA_PATH = os.path.join('data', 'rfm_features_enhanced.xlsx')

# --- Load All Resources (Cached) ---
@st.cache_resource
def load_all_resources():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        y_test_loaded = joblib.load(Y_TEST_PATH)
        y_pred_gb_loaded = joblib.load(Y_PRED_GB_PATH)
        feature_names_loaded = joblib.load(FEATURE_NAMES_PATH)
        full_rfm_df = pd.read_excel(RFM_DATA_PATH)

        # Reconstruct known_country_columns and unique_countries_from_features
        known_general_features = ['Recency', 'Frequency', 'AOV', 'Tenure', 'UniqueProducts']
        known_country_columns = [col for col in feature_names_loaded if col.startswith('Country_')]
        
        # Ensure that `features_for_model` matches the exact order of features
        # X.columns when the model and scaler were trained.
        # feature_names_loaded contains exactly this order.
        features_for_model = feature_names_loaded

        unique_countries_from_features = [col.replace('Country_', '') for col in known_country_columns]
        unique_countries_from_features.sort()
        unique_countries_from_features.insert(0, 'Select Country')

        return model, scaler, y_test_loaded, y_pred_gb_loaded, features_for_model, unique_countries_from_features, known_country_columns, full_rfm_df

    except FileNotFoundError as e:
        st.error(f"Error: Missing a required file. Please ensure all saved resources are in '{MODEL_DIR}' and `rfm_features_enhanced.xlsx` is in 'data'. Missing: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        st.stop()

model, scaler, y_test, y_pred_gb, features_for_model, unique_countries_from_features, known_country_columns, full_rfm_df = load_all_resources()

# --- App Title and Introduction ---
st.markdown('<p class="main-header">🌟 Customer Lifetime Value (CLTV) Predictor Dashboard 🌟</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Predict, Analyze, and Understand Your Customer\'s Future Value</p>', unsafe_allow_html=True)

st.success("All models and data resources loaded successfully! Dive into insights and predictions.")

# --- Navigation Tabs ---
tab1, tab2, tab3 = st.tabs(["🚀 Predict CLTV", "📊 Model Performance & Insights", "🔍 Data Explorer"])

# --- Tab 1: Predict CLTV ---
with tab1:
    st.markdown('<p class="input-header">Enter Customer Data for Prediction</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input("Recency (Days since last purchase):", min_value=0, value=30, help="Number of days since the customer's last purchase. Lower is generally better.")
        frequency = st.number_input("Frequency (Total number of purchases):", min_value=1, value=5, help="Total number of unique transactions made by the customer. Higher is better.")
        aov = st.number_input("Average Order Value (Average spend per purchase):", min_value=0.0, value=50.0, format="%.2f", help="Average monetary value per transaction for this customer.")

    with col2:
        tenure = st.number_input("Tenure (Days since first purchase):", min_value=0, value=365, help="Number of days since the customer's very first purchase. Longer tenure implies more loyalty.")
        unique_products = st.number_input("Unique Products (Number of different items bought):", min_value=1, value=10, help="Diversity of products purchased by the customer. A wider range might indicate broader interest.")
        selected_country = st.selectbox("Customer's Country:", unique_countries_from_features, help="The primary country of the customer, used for regional insights by the model.")

    st.write("---")

    # Prediction Button
    if st.button("✨ Get CLTV Prediction"):
        if selected_country == 'Select Country':
            st.warning("Please select a valid country to get a prediction.")
        else:
            # Prepare Input Data for Prediction
            input_data = {
                'Recency': recency,
                'Frequency': frequency,
                'AOV': aov,
                'Tenure': tenure,
                'UniqueProducts': unique_products
            }

            # Add one-hot encoded country features, initializing all to 0
            for country_col in known_country_columns:
                input_data[country_col] = 0

            # Set the selected country's column to 1
            one_hot_col_name = f'Country_{selected_country}'
            if one_hot_col_name in input_data:
                input_data[one_hot_col_name] = 1
            else:
                st.warning(f"Note: Selected country '{selected_country}' was not specifically present in the training data's country features. Its contribution will be generalized.")

            # Create a DataFrame with the exact column order as `features_for_model`
            input_df = pd.DataFrame([input_data])
            input_df = input_df[features_for_model] # CRUCIAL for order consistency

            # Scale the input data using the loaded scaler
            scaled_input = scaler.transform(input_df)

            # Make Prediction
            predicted_cltv = model.predict(scaled_input)[0]

            # Display Prediction with an engaging message
            st.markdown(
                f"""
                <div class="prediction-box">
                    <p class="prediction-text">The Estimated CLTV is:</p>
                    <p class="predicted-value">${predicted_cltv:.2f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.info("💡 **Insight**: This prediction represents the estimated total monetary value a customer is expected to bring to your business over their lifetime, based on the provided inputs and our powerful Gradient Boosting model.")


# --- Tab 2: Model Performance & Insights ---
with tab2:
    st.markdown('<p class="input-header">Model Performance & Key Insights</p>', unsafe_allow_html=True)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred_gb)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    r2 = r2_score(y_test, y_pred_gb)

    st.subheader("Performance Metrics (Gradient Boosting Regressor):")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.markdown(f'<div class="metric-card"><p class="metric-title">Mean Absolute Error (MAE)</p><p class="metric-value">${mae:.2f}</p></div>', unsafe_allow_html=True)
    with metric_col2:
        st.markdown(f'<div class="metric-card"><p class="metric-title">Root Mean Squared Error (RMSE)</p><p class="metric-value">${rmse:.2f}</p></div>', unsafe_allow_html=True)
    with metric_col3:
        st.markdown(f'<div class="metric-card metric-r2"><p class="metric-title">R-squared (R²)</p><p class="metric-value">{r2:.4f}</p></div>', unsafe_allow_html=True)

    st.write("---")

    # --- Visualizations ---
    st.subheader("Model Insights Visualizations:")

    # 1. Feature Importance
    if hasattr(model, 'feature_importances_') and features_for_model is not None:
        st.markdown("#### Feature Importance: What Drives CLTV?")
        feature_importances_gb = pd.Series(model.feature_importances_, index=features_for_model).sort_values(ascending=False)

        top_n_features = 20 # Display top 20 features
        top_features_to_plot = feature_importances_gb.head(top_n_features)

        fig_fi, ax_fi = plt.subplots(figsize=(12, max(8, top_n_features * 0.4)))
        sns.barplot(x=top_features_to_plot.values, y=top_features_to_plot.index, palette='viridis', ax=ax_fi)
        ax_fi.set_title(f'Gradient Boosting Regressor: Top {top_n_features} Feature Importances', fontsize=16)
        ax_fi.set_xlabel('Importance (Scaled)', fontsize=12)
        ax_fi.set_ylabel('Feature', fontsize=12)
        ax_fi.tick_params(axis='x', labelsize=10)
        ax_fi.tick_params(axis='y', labelsize=10)
        ax_fi.grid(axis='x', linestyle='--', alpha=0.7)
        fig_fi.tight_layout()
        st.pyplot(fig_fi)
        plt.close(fig_fi) # Close figure to prevent display issues

        st.markdown("""
        **Inference from Feature Importance:**
        This chart reveals which customer attributes the model considers most crucial in predicting CLTV.
        Typically, features like **Recency** (how recently they purchased), **Frequency** (how often), and **Average Order Value (AOV)**
        are strong indicators. Country-specific factors also play a role, highlighting regional differences in customer behavior.
        Focusing on improving these high-importance metrics can significantly boost overall CLTV.
        """)
        st.write("---")
    else:
        st.warning("Feature importances are not available for this model type or could not be loaded.")


    # 2. Actual vs. Predicted Plot
    st.markdown("#### Actual vs. Predicted CLTV: How Well Does the Model Perform?")
    fig_ap, ax_ap = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x=y_test, y=y_pred_gb, alpha=0.6, ax=ax_ap)
    ax_ap.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')
    ax_ap.set_title('Gradient Boosting Regressor: Actual vs. Predicted Monetary Value', fontsize=16)
    ax_ap.set_xlabel('Actual Monetary Value (CLTV)', fontsize=12)
    ax_ap.set_ylabel('Predicted Monetary Value (CLTV)', fontsize=12)
    ax_ap.grid(True, linestyle=':', alpha=0.7)
    ax_ap.legend()
    fig_ap.tight_layout()
    st.pyplot(fig_ap)
    plt.close(fig_ap)

    st.markdown("""
    **Inference from Actual vs. Predicted Plot:**
    The closer the scatter points are to the red dashed line, the more accurate our predictions are.
    A good model will show points clustered along this line. Deviations indicate where the model
    might be overestimating or underestimating CLTV. Points far from the line (outliers) might be
    challenging cases for the model.
    """)
    st.write("---")

    # 3. Residuals Distribution
    st.markdown("#### Residuals Distribution: Are Our Errors Random?")
    residuals = y_test - y_pred_gb
    fig_res, ax_res = plt.subplots(figsize=(10, 7))
    sns.histplot(residuals, kde=True, ax=ax_res, color='skyblue', bins=50)
    ax_res.set_title('Gradient Boosting Regressor: Residuals Distribution', fontsize=16)
    ax_res.set_xlabel('Residuals (Actual CLTV - Predicted CLTV)', fontsize=12)
    ax_res.set_ylabel('Count', fontsize=12)
    ax_res.axvline(0, color='red', linestyle='--', label='Zero Residuals')
    ax_res.legend()
    ax_res.grid(True, linestyle=':', alpha=0.7)
    fig_res.tight_layout()
    st.pyplot(fig_res)
    plt.close(fig_res)

    st.markdown("""
    **Inference from Residuals Distribution:**
    Ideally, residuals should be normally distributed around zero, indicating that the model's errors are random and unbiased.
    If the distribution is skewed or has a clear pattern, it might suggest that the model is systematically
    over-predicting or under-predicting for certain segments of customers.
    """)
    st.write("---")

# --- Tab 3: Data Explorer ---
with tab3:
    st.markdown('<p class="input-header">Explore the Underlying Data (rfm_features_enhanced)</p>', unsafe_allow_html=True)

    st.subheader("Dataset Overview:")
    st.write("This section allows you to explore the characteristics of the data used to train the model.")

    st.markdown("#### Descriptive Statistics:")
    st.dataframe(full_rfm_df.drop(columns=[col for col in full_rfm_df.columns if col.startswith('Country_') or col == 'CustomerID']).describe().transpose().round(2))
    st.markdown("""
    **Inference from Descriptive Statistics:**
    This table provides a quick summary of your numerical features.
    Look at the mean, standard deviation, min, max, and quartiles to understand the distribution and potential outliers.
    For example, a large difference between mean and median (50% percentile) can indicate skewness.
    """)
    st.write("---")

    st.markdown("#### Distribution of Key Features:")

    # Plot distributions of key RFM-related features
    plot_cols = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Tenure', 'UniqueProducts']
    num_plots = len(plot_cols)
    cols_per_row = 3
    rows = int(np.ceil(num_plots / cols_per_row))

    for r in range(rows):
        plot_cols_row = st.columns(cols_per_row)
        for i in range(cols_per_row):
            idx = r * cols_per_row + i
            if idx < num_plots:
                col_name = plot_cols[idx]
                with plot_cols_row[i]:
                    fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
                    sns.histplot(full_rfm_df[col_name], kde=True, ax=ax_dist, color=plt.cm.get_cmap('Set2')(i))
                    ax_dist.set_title(f'Distribution of {col_name}', fontsize=14)
                    ax_dist.set_xlabel(col_name, fontsize=10)
                    ax_dist.set_ylabel('Count', fontsize=10)
                    ax_dist.tick_params(axis='x', labelsize=8)
                    ax_dist.tick_params(axis='y', labelsize=8)
                    fig_dist.tight_layout()
                    st.pyplot(fig_dist)
                    plt.close(fig_dist)

    st.markdown("""
    **Inference from Feature Distributions:**
    These histograms (with KDE) show the shape of your data for each key feature.
    - **Skewness:** Many distributions are likely right-skewed (e.g., Monetary, Frequency), meaning most customers have lower values, with a few high-value customers. This is typical for customer data.
    - **Outliers:** Long tails or isolated bars indicate potential outliers that might have influenced the model.
    - **Data Cleaning:** Visualizing these confirms the effectiveness of any data cleaning or transformation steps applied during feature engineering.
    """)
    st.write("---")

    st.markdown("#### Top Countries by Customer Count:")
    country_counts = {}
    for col in full_rfm_df.columns:
        if col.startswith('Country_'):
            country_name = col.replace('Country_', '')
            country_counts[country_name] = full_rfm_df[col].sum()

    if country_counts:
        top_countries_df = pd.Series(country_counts).sort_values(ascending=False).head(10) # Top 10 countries
        fig_country, ax_country = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_countries_df.values, y=top_countries_df.index, palette='coolwarm', ax=ax_country)
        ax_country.set_title('Top 10 Countries by Customer Count', fontsize=16)
        ax_country.set_xlabel('Number of Customers', fontsize=12)
        ax_country.set_ylabel('Country', fontsize=12)
        ax_country.tick_params(axis='x', labelsize=10)
        ax_country.tick_params(axis='y', labelsize=10)
        ax_country.grid(axis='x', linestyle='--', alpha=0.7)
        fig_country.tight_layout()
        st.pyplot(fig_country)
        plt.close(fig_country)
        st.markdown("""
        **Inference from Top Countries:**
        This chart highlights the countries with the largest customer bases in your dataset.
        Understanding your primary markets is crucial for targeted marketing and strategic planning.
        You can see where your current customer concentration lies.
        """)
    else:
        st.info("Country distribution insights require 'Country_' columns in the data.")


# --- Sidebar ---
st.sidebar.header("About This CLTV Dashboard")
st.sidebar.info(
    "This interactive dashboard is designed to empower businesses with deeper insights into "
    "their customer base. It combines state-of-the-art machine learning (Gradient Boosting) "
    "for **CLTV Prediction** with rich **Model Performance Visualizations** and a comprehensive "
    "**Data Explorer** to analyze the underlying customer attributes."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Key Features:")
st.sidebar.markdown("""
- **Predictive Analytics**: Estimate future customer value.
- **Model Transparency**: Understand model strengths and weaknesses.
- **Data Understanding**: Explore distributions and patterns in your customer data.
""")
st.sidebar.markdown("---")
st.sidebar.write("Developed for Strategic Customer Relationship Management.")
st.sidebar.write("© 2025 Your Company/Project Name") # Replace with your name/project