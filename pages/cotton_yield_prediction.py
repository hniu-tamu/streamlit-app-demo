import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Lubbock Cotton Yield Prediction App

This app predicts the **Lubbock Cotton Yield**!
""")
st.write('---')

# Loads the Boston House Price Dataset
mean_data = pd.read_csv('my_cotton_yield_data.csv')
X = mean_data[['red', 'green', 'blue', 'mask_size', 'irrigation', 'canopy_cover',
       'canopy_height', 'canopy_volume', 'exgreeness']]
Y = mean_data['yield']

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    red = st.sidebar.slider('red', X.red.min(), X.red.max(), X.red.mean())
    green = st.sidebar.slider('green', X.green.min(), X.green.max(), X.green.mean())
    blue = st.sidebar.slider('blue', X.blue.min(), X.blue.max(), X.blue.mean())
    mask_size = st.sidebar.slider('mask_size', X.mask_size.min(), X.mask_size.max(), X.mask_size.mean())
    irrigation = st.sidebar.slider('irrigation', X.irrigation.min(), X.irrigation.max(), X.irrigation.mean())
    canopy_cover = st.sidebar.slider('canopy_cover', X.canopy_cover.min(), X.canopy_cover.max(), X.canopy_cover.mean())
    canopy_height = st.sidebar.slider('canopy_height', X.canopy_height.min(), X.canopy_height.max(), X.canopy_height.mean())
    canopy_volume = st.sidebar.slider('canopy_volume', X.canopy_volume.min(), X.canopy_volume.max(), X.canopy_volume.mean())
    exgreeness = st.sidebar.slider('exgreeness', X.exgreeness.min(), X.exgreeness.max(), X.exgreeness.mean())
    data = {'red': red,
            'green': green,
            'blue': blue,
            'mask_size': mask_size,
            'irrigation': irrigation,
            'canopy_cover': canopy_cover,
            'canopy_height': canopy_height,
            'canopy_volume': canopy_volume,
            'exgreeness': exgreeness}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor(min_samples_leaf= 10, min_samples_split= 20, n_estimators= 100)
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of Lubbock Cotton Yield')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
st.write('---')

