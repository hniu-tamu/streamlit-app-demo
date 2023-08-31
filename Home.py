import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Cotton Features and Yield in Lubbock, TX')

DATA_COLUMN = 'yield'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
DATA_FILE_PATH = 'my_cotton_yield_data.csv'

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_FILE_PATH, nrows=nrows)
    return data

data_load_state = st.text('Loading data...')
data = load_data(96) # because we only have 96 rows
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Yield distribution')
sns.histplot(data[DATA_COLUMN], kde=True)
plt.xlabel('Yield')
plt.ylabel('Frequency')
plt.title('Yield Distribution')
st.pyplot(plt)


# A demo, change the lat, lon, in the future.
lat = 33.5934
start_point = -101.9029
end_point = -101.9014
num_points = 96



points = np.linspace(start_point, end_point, num_points)

data['lon'] = points
data['lat'] = np.full(num_points, lat)
data['size'] = data['yield']/70

def map_irrigation_color(irrigation_value):
    if irrigation_value == 0:
        return '#FF0000' # red
    elif irrigation_value == 0.1:
        return '#00FF00' # green
    elif irrigation_value == 0.2:
        return '#0000FF' # blue
    else: 
        return '#FFFF00'  # Yellow

# Apply the custom function to create a new column for color
data['color'] = data['irrigation'].apply(map_irrigation_color)

st.map(data, size = 'size', color = 'color') # filtered_data

# image = Image.open('sunrise.jpg')
# image = 'https://i.gifer.com/4j.gif'
# st.image(image, caption='Flash haha')

