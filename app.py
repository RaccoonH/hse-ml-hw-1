import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@st.cache_resource
def load_model():
    with open('models/ridge.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names


@st.cache_data
def load_data():
    df_train = pd.read_csv("datasets/train_df.csv", index_col=0)
    df_train['name'] = df_train['name'].str.split().str[0]
    return df_train


@st.cache_data
def load_empty_data(columns):
    return pd.DataFrame(columns=columns)


@st.cache_resource
def plot_corr(df_train):
    corr = df_train.corr(numeric_only=True)
    fig = plt.figure()
    sns.heatmap(corr)
    fig.tight_layout()
    return fig


@st.cache_resource
def plot_bar(df_train):
    fig = plt.figure()
    df_train['name'].value_counts().plot(kind='bar')
    return fig


@st.cache_resource
def plot_boxplot(df_train):
    numeric_df = df_train.select_dtypes(include=np.number)
    fig, axes = plt.subplots((len(numeric_df.columns) // 2) + 1, 2, figsize=(6, 4 * len(numeric_df.columns)))
    ax_counter = 0
    counter = 0
    for col in numeric_df.columns:
        axes[ax_counter][counter].boxplot(numeric_df[col])
        axes[ax_counter][counter].set_title(col)
        counter += 1
        if counter == 2:
            ax_counter += 1
            counter = 0
    return fig


def prepare_features(df, feature_names):
    df_proc = df.copy()
    df_proc['name'] = df_proc['name'].str.split().str[0]
    df_proc['seats'] = df_proc['seats'].astype('object')
    df_proc = pd.get_dummies(df_proc, drop_first=True)

    for i in ["year", "km_driven", "mileage", "engine", "max_power", "torque", "max_torque_rpm"]:
        df_proc[f"{i}_2"] = df_proc[i] ** 2

    missing_cols = set(feature_names) - set(df_proc.columns)
    for c in missing_cols:
        df_proc[c] = False
    extra_cols = set(df_proc.columns) - set(feature_names)
    df_proc.drop(list(extra_cols), axis=1)
    df_proc = df_proc[feature_names]

    return df_proc


st.set_page_config(
    page_title="Car price prediction",
    layout="centered",
)

model, feature_names = load_model()
df_input = load_empty_data(feature_names)

st.title("Car price predict")

tab_input, tab_csv, tab_plots = st.tabs(["Input", "CSV", "Plots"])

with tab_input:
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=1980, max_value=2025, key="year")
        df_input['year'] = [year]
        km_driven = st.number_input("Km driven", min_value=0, key="km")
        df_input['km_driven'] = [km_driven]
        mileage = st.number_input("Mileage", min_value=0, key="mileage")
        df_input['mileage'] = [mileage]
        engine = st.number_input("Engine", min_value=0, key="engine")
        df_input['engine'] = [engine]
        torque = st.number_input("Torque", min_value=0, key="torque")
        df_input['torque'] = [torque]
        max_torque_rpm = st.number_input("Max torque RPM", min_value=0, key="max_torque_rpm")
        df_input['max_torque_rpm'] = [max_torque_rpm]

    with col2:
        car_model = st.selectbox("Model", ["Audi", "BMW", "Chevrolet", "Daewoo", "Datsun", "Fiat", "Force", "Ford", "Honda", 'Hyundai',
                                           'Isuzu', 'Jaguar', 'Jeep', 'Kia', 'Land', 'Lexus', 'MG', 'Mahindra', 'Maruti', 'Mercedes-Benz',
                                           'Mitsubishi', 'Nissan', 'Peugeot', 'Renault', 'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo', 'Ambassador'])
        if car_model != 'Ambassador':
            df_input[f'name_{car_model}'] = [True]

        fuel = st.selectbox("Fuel", ['Diesel', 'LPG', 'Petrol', 'CNG'])
        if fuel != 'CNG':
            df_input[f'fuel_{fuel}'] = [True]

        seller_type = st.selectbox("Seller type", ['Individual', 'Trustmark Dealer', 'Dealer'])
        if seller_type != 'Dealer':
            df_input[f'seller_type_{seller_type}'] = [True]

        owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
        if owner != 'First Owner':
            df_input[f'owner_{owner}'] = [True]

        seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10, 14])
        if seats == 2:
            df_input[f'seats_{seats}'] = [True]

        transmission = st.checkbox("Manual transmission")
        df_input['transmission_Manual'] = [transmission]

    if st.button("Predict price"):
        df_input = df_input[feature_names]
        df_input = df_input.fillna(False)
        st.session_state['predict'] = model.predict(df_input)

    if 'predict' in st.session_state:
        st.dataframe(st.session_state['predict'])

with tab_csv:
    df_file = st.file_uploader("Upload CSV file")
    if df_file:
        df_file = pd.read_csv(df_file)
        df_transformed = prepare_features(df_file, feature_names)
        st.dataframe(model.predict(df_transformed))

with tab_plots:
    df_train = load_data()

    st.write("Correlation matrix")
    st.pyplot(plot_corr(df_train))
    st.write("Car models bar plot")
    st.pyplot(plot_bar(df_train))
    st.write("Boxplot for numeric column")
    st.pyplot(plot_boxplot(df_train))
