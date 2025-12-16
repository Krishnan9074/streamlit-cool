
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# --- Page Config ---
st.set_page_config(page_title="DC Airbnb Investor Dashboard", page_icon="🏠", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        # Load the uploaded file directly
        df = pd.read_csv('clean_airbnb_dc.csv')
        
        # Clean price outliers (IQR Method) for better visuals
        # We do this here to ensure the visuals scale nicely even if the raw data has outliers
        Q1 = df['price'].quantile(0.25)
        Q3 = df['price'].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]
        
        return df, df_clean
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

df, df_clean = load_data()

if df.empty:
    st.stop()

# --- Train Model ---
@st.cache_resource
def train_model(data):
    features = ['bedrooms', 'accommodates', 'dist_to_mall', 'number_of_reviews']
    # Drop rows with missing values in features
    data_model = data.dropna(subset=features + ['price'])
    X = data_model[features]
    y = data_model['price']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# --- Sidebar ---
st.sidebar.header("🕹️ Property Configuration")
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 2)
guests = st.sidebar.slider("Guest Capacity", 1, 12, 4)
dist_mall = st.sidebar.slider("Distance to Mall (miles)", 0.1, 10.0, 1.5)
reviews = st.sidebar.slider("Reviews", 0, 500, 50)

st.sidebar.markdown("---")
st.sidebar.info("**Assumptions:**\n- Occupancy Rate: 65%\n- Valuation Multiple: 15x Annual Rev")

# --- Main Page ---
st.title("🏠 The Capital Investor: DC Market Overview")

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Listings", f"{len(df):,}")
col2.metric("Avg Price", f"${df['price'].mean():.2f}")
col3.metric("Avg Dist to Mall", f"{df['dist_to_mall'].mean():.1f} mi")
col4.metric("Avg Capacity", f"{df['accommodates'].mean():.1f} Guests")

st.markdown("---")

# Prediction
input_data = pd.DataFrame({
    'bedrooms': [bedrooms], 'accommodates': [guests], 
    'dist_to_mall': [dist_mall], 'number_of_reviews': [reviews]
})
pred_price = model.predict(input_data)[0]
monthly_rev = pred_price * 30 * 0.65
asset_val = monthly_rev * 12 * 15

st.subheader("💵 Financial Projections")
c1, c2, c3 = st.columns(3)
c1.markdown(f"#### Predicted Rate\n# ${pred_price:.2f}")
c2.markdown(f"#### Est. Monthly Rev\n# ${monthly_rev:,.2f}")
c3.markdown(f"#### Est. Asset Value\n# ${asset_val:,.0f}")

# Visuals
st.subheader("📊 Market Map")
fig_map = px.scatter_mapbox(
    df, lat="latitude", lon="longitude", color="price", size="price",
    color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
    mapbox_style="carto-positron", height=500
)
st.plotly_chart(fig_map, use_container_width=True)
