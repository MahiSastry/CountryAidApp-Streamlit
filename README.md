## Streamlit Repository (Deployed App & Usage Instructions)**  

```md
# Country Aid Clustering App  

## **About the Project**  
This web application allows users to input a country’s economic and human development indicators to determine the required level of aid or assistance. The model categorizes countries into four clusters:  

1️.Critical Aid Required
2️.Targeted Support Needed
3️. Monitoring Required  
4️. Sustainable Progress

## **How It Works**  
- Users enter country-specific data (GDP, child mortality, exports, etc.) in the sidebar.  
- The app processes the input, applies the trained K-Means model, and assigns the country to one of the four clusters.  
- A description of the assigned cluster is provided, explaining why the country falls into that category.  

## **App Workflow**  
1️. User enters country indicators via Streamlit UI  
2️. Data is preprocessed (scaling, encoding, PCA transformation)  
3️. Trained K-Means model assigns a cluster  
4️. Results are displayed with relevant insights  

## On **Streamlit Cloud**
https://countryaidapp-app-rhmtqmdjxvgfotrk7qyyer.streamlit.app/

## **How to Run Locally**  
Clone the Repository  
```bash
git clone https://github.com/MahiSastry/CountryAidApp-Streamlit.git
cd CountryAidApp-Streamlit
