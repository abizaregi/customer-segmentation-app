import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from sklearn.cluster import KMeans

st.title('Customer Segmentation - Application')
st.write("""***App to view customer segmentation based on personal and information data customer***""")
st.sidebar.header("**Filter Display Data**")
df = pd.read_csv('data_app.csv', sep=';')

# Sidebar
sorted_education_unique = sorted(df['education'].unique())
sorted_marital_unique = sorted(df['marital'].unique())
sorted_job = sorted(df['job'].unique())
selected_channel = st.sidebar.multiselect('Select Education', sorted_education_unique, sorted_education_unique)
selected_month = st.sidebar.multiselect('Select Marital', sorted_marital_unique, sorted_marital_unique)
selected_job = st.sidebar.multiselect('Select Job', sorted_job, sorted_job)
# Filtering data
df_selected = df[(df['education'].isin(selected_channel)) & (df['marital'].isin(selected_month)) & (df['job'].isin(selected_job))]

st.write('''input name: Abizar Egi | input password: testapp''')
name = st.text_input('input your account name: ')
pw = st.text_input('input your account password: ')
submit = st.button('login')
if submit:
    if (name == "Abizar Egi") & (pw == "testapp"):
        st.header('Display Data Selected in Sidebar')
        st.dataframe(df_selected)
        st.write('Data Shape: ' + str(df_selected.shape[0]) + ' rows and ' + str(df_selected.shape[1]) + ' columns.')
        def filedownload(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="sales-data.csv">Download CSV File</a>'
            return href
        st.markdown(filedownload(df_selected), unsafe_allow_html=True)
        st.write("""
        #### Distribution of Job ####
        """)
        plt.subplots(figsize=(15,5))
        sns.countplot(x='job', data=df_selected)
        plt.show()
        st.pyplot()

        st.write("""
        #### Distribution of Marital ####
        """)
        sns.countplot(x='marital', data=df_selected)
        plt.show()
        st.pyplot()

        st.write("""
        #### Distribution of Education ####
        """)
        sns.countplot(x='education', data=df_selected)
        plt.show()
        st.pyplot()

        st.write("""
        #### Distribution of Contact ####
        """)
        sns.countplot(x='contact', data=df_selected)
        plt.show()
        st.pyplot()

        st.write("""
        #### Distribution of Age ####
        """)
        df_selected.hist('age', bins=35)
        plt.ylabel('Count')
        plt.xlabel('Age')
        plt.show()
        st.pyplot()

        st.write("""
        #### Age to Balance, Colored by Term Deposit ####
        """)
        sns.scatterplot('age', 'balance', hue='term_deposit', data=df_selected)
        plt.show()
        st.pyplot()

        x = df_selected[['age','balance']]
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
            km.fit(x)
            wcss.append(km.inertia_)
        model = KMeans(n_clusters=5, init='k-means++', random_state=0)
        model.fit_predict(x)

        st.write("""
        #### Before Clustering ####
        """)
        sns.scatterplot('age', 'balance', data=df_selected)
        plt.xlabel('Age')
        plt.ylabel('Balance')
        plt.show()
        st.pyplot()

        st.write("""
        #### After Clustering ####
        """)
        plt.scatter(x=df_selected[df_selected['Cluster_Prediction']==1]['age']
           , y=df_selected[df_selected['Cluster_Prediction']==1]['balance'],
           s=30, edgecolor='black', linewidth=0.3, c='blue', label='Cluster 1')
        plt.scatter(x=df_selected[df_selected['Cluster_Prediction']==2]['age']
           , y=df_selected[df_selected['Cluster_Prediction']==2]['balance'],
           s=30, edgecolor='black', linewidth=0.3, c='red', label='Cluster 2')
        plt.scatter(x=df_selected[df_selected['Cluster_Prediction']==3]['age']
           , y=df_selected[df_selected['Cluster_Prediction']==3]['balance'],
           s=30, edgecolor='black', linewidth=0.3, c='pink', label='Cluster 3')
        plt.scatter(x=df_selected[df_selected['Cluster_Prediction']==4]['age']
           , y=df_selected[df_selected['Cluster_Prediction']==4]['balance'],
           s=30, edgecolor='black', linewidth=0.3, c='deepskyblue', label='Cluster 4')
        plt.scatter(x=df_selected[df_selected['Cluster_Prediction']==0]['age']
           , y=df_selected[df_selected['Cluster_Prediction']==0]['balance'],
           s=30, edgecolor='black', linewidth=0.3, c='purple', label='Cluster 0')

        plt.scatter(x=model.cluster_centers_[:,0], y=model.cluster_centers_[:,1], s=30, c='grey', label='Centroids', edgecolor='black', linewidth=0.3)
        plt.legend(loc='right')
        plt.xlabel('Age')
        plt.ylabel('Balance')
        plt.title('Clusters')
        plt.show()
        st.pyplot()

        st.write('''Cluster 1 = Customer rata-rata  berumur 20 - 60 dengan balance  10.000 keatas\n
        Cluster 2 = Cluster dengan jumlah  customer terendah, akan tetapi  memiliki balance tertinggi\n
        Cluster 4 = Cluster rata-rata  berumur 22 - 60 dengan balance  15.000 - 40.000\n
        Cluster 5 = Cluster dengan balance  terendah''')

    else:
        st.write('There was a problem with your login')

