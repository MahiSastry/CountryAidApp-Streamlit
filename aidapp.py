import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.decomposition import PCA
import os

pickle_file_path = os.path.join("models", "kmeans_countryaid.pkl")
with open(pickle_file_path, 'rb') as file:
    model = pickle.load(file)

norm_file_path = os.path.join("models", "norm_info.pkl")
with open(norm_file_path, 'rb') as file:
    norm_info = pickle.load(file)

pca_file_path = os.path.join("models", "pca_countryaid.pkl")
with open(pca_file_path, 'rb') as file:
    pca_model = pickle.load(file)


def normalize(column,min,max):
    column_norm = (column - min)/(max - min)
    return np.round(column_norm,2)

def preprocess_input(raw_input_df):
    continent = raw_input_df['continent'].iloc[0]
    print('Continent in preprocess',continent)
    one_hot_columns = ['continent_Africa', 'continent_Asia', 'continent_Europe', 
                   'continent_North America','continent_Oceania', 'continent_South America']
    continent_data = {f'continent_{c}': 1 if c == continent else 0 for c in 
              ['Africa', 'Asia', 'Europe', 'North America','Oceania', 'South America']}
    encode_df = pd.DataFrame([continent_data],columns = one_hot_columns)

    #preprocess numerical columns
    train_min = norm_info['min']
    train_max = norm_info['max']
    print("MIN:",train_min)
    print("MAX:",train_max)
    scaler = preprocessing.MinMaxScaler()
    num_cols = raw_input_df.select_dtypes(include=[np.int64, np.float64]).columns
    print("NUM COLS:",num_cols)
    print("RAW INPUT DF",raw_input_df.info())
    norm_df = raw_input_df[num_cols].apply(lambda col :normalize(col,train_min[col.name],train_max[col.name]))
    #df_norm = pd.DataFrame(scaler.fit_transform(raw_input_df[num_cols]), columns=num_cols, index=raw_input_df[num_cols].index)
    #st.write(norm_df.head())
    df_final = pd.concat([encode_df,norm_df],axis=1)
    return df_final

def predict_cluster(preprocess_df):
    # Apply PCA to the scaled input
    user_input_pca = pca_model.transform(preprocess_df)
    print('PCA INPUT:',user_input_pca)

    # Predict the cluster using the trained KMeans model
    cluster = model.predict(user_input_pca)
    return cluster


st.title("Aid Allocation Advisor: Clustering Based on Development Indicators")
with st.container(border = True):
    st.caption("Use this tool to cluster countries based on key human development and economic indicators. Enter data for child mortality, GDP, healthcare, trade, and other metrics to determine which level of aid or assistance the country requires. The tool assigns countries to one of four aid clusters, each representing a unique development stage and support need.")
st.subheader("Input Summary")


continent_list = ['Africa','Asia','Europe','North America','South America','Oceania']

with st.sidebar.form("input_form"):
    st.header("Input Parameters")
    continent = st.selectbox("Select continent",continent_list)
    child_mort = st.number_input("Child mortality", min_value=3.0,max_value = 116.0,step = 0.01) #Death of children under 5 years of age per 1000 live births
    total_fer = st.number_input("Female fertility", min_value=1.0,max_value = 6.0,step = 0.01)
    life_expec = st.number_input("Life expectency", min_value=55.0,max_value = 82.0,step = 0.01)
    income = st.number_input("Income", min_value=1200.0,max_value = 50000.0,step = 0.01)
    gdpp = st.number_input("GDP", min_value=460.0,max_value = 49000.0,step = 0.01)
    inflation = st.number_input("Inflation", min_value=0.0,max_value = 21.0,step = 0.01) 
    exports = st.number_input("Exports", min_value=12.0,max_value = 81.0,step = 0.01) 
    health = st.number_input("Health Expenditure", min_value=2.0,max_value = 12.0,step = 0.01)
    imports = st.number_input("Imports", min_value=18.0,max_value = 82.0,step = 0.01)

    exp_imp_ratio = exports/imports
    submitted = st.form_submit_button("Submit")

col_list = ['continent', 'child_mort', 'exports','health','imports','income','inflation','life_expec','total_fer','gdpp','exp_imp_ratio']
user_input = [continent, child_mort, exports,health,imports,income,inflation,life_expec,total_fer,gdpp,exp_imp_ratio]
print(user_input)
raw_input_df = pd.DataFrame([user_input],columns = col_list)
st.table(raw_input_df)
cluster_label_dict = {0: 'Moderate Development Zone',1:'Prosperous Economies',2:'Developing Economies',3:'Critical Support Needed'}
cluster_label_color = {0 : 'Yellow',1:'Green',2:'Orange',3:'Red'}
cluster_data = {
    0: {
        "aid_level": "Moderate Aid Priority",
        "total_aid": "15-20%",
        
        "rationale":"Focus on strengthening existing systems and addressing specific gaps in development"
    },
    1: {
        "aid_level": "Low Aid Priority",
        "total_aid": "5-10%",
        
        "rationale":"Limited aid focused on specialized technical support and sustainability"
    },
    2: {
        "aid_level": "High Aid Priority",
        "total_aid": "25-30%",
       
        "rationale":"Substantial aid targeted at maintaining growth momentum and addressing key bottlenecks"
    },
    3: {
        "aid_level": "Urgent Aid Priority",
        "total_aid": "40-50%",
        
        "rationale":"Intensive support needed across all basic development indicators"
    }
    
}

if submitted    :
    preprocess_df = preprocess_input(raw_input_df)
    print(preprocess_df.head())
    cluster = predict_cluster(preprocess_df)
    #st.write(f'This data point belongs to Cluster {cluster[0]}')
    cluster_id = cluster[0]
    st.subheader("Result:")
    #with st.container():
            #st.markdown(f"<span style='color: {cluster_label_color[cluster_id].lower()}; font-weight: bold; font-size: 24px'>{cluster_label_dict[cluster_id]}</span>", unsafe_allow_html=True)
    #col1, col2= st.columns(2)
    #st.subheader("Recommendation")

    with st.container(border=True):
        st.markdown(f"<span style='color: {cluster_label_color[cluster_id].lower()}; font-weight: bold; font-size: 30px'>{cluster_label_dict[cluster_id]}</span>", unsafe_allow_html=True)
        st.markdown(f'<span style="font-weight: bold; font-size: 24px">{cluster_data[cluster_id]["aid_level"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="font-weight: bold;font-size: 16px">{cluster_data[cluster_id]["total_aid"]}</span> of available aid budget', unsafe_allow_html=True)
        #st.markdown(f'<span style="font-size: 24px">Sector Aid Allocation</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="font-weight: bold; font-size: 16px">Rationale:</span>{cluster_data[cluster_id]["rationale"]}', unsafe_allow_html=True)
        

