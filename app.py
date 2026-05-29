import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Retail Analytics Dashboard",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv(
        "Superstore.csv",
        encoding='latin1'
    )

    df['Order Date'] = pd.to_datetime(
        df['Order Date'],
        errors='coerce'
    )

    df['Month'] = df['Order Date'].dt.month

    return df


df = load_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Filters")

selected_category = st.sidebar.multiselect(
    "Select Category",
    df['Category'].unique(),
    default=df['Category'].unique()
)

selected_region = st.sidebar.multiselect(
    "Select Region",
    df['Region'].unique(),
    default=df['Region'].unique()
)

filtered_df = df[
    (df['Category'].isin(selected_category)) &
    (df['Region'].isin(selected_region))
]

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("Retail Analytics and Smart Offer Recommendation System")

st.markdown(
    """
    This dashboard provides retail sales analysis,
    business insights, and intelligent offer recommendations
    using machine learning and product correlation analysis.
    """
)

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------

st.subheader("Business Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Sales",
        f"${filtered_df['Sales'].sum():,.0f}"
    )

with col2:
    st.metric(
        "Total Profit",
        f"${filtered_df['Profit'].sum():,.0f}"
    )

with col3:
    st.metric(
        "Total Orders",
        filtered_df['Order ID'].nunique()
    )

with col4:
    st.metric(
        "Average Discount",
        f"{filtered_df['Discount'].mean()*100:.1f}%"
    )

# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Sales Analysis",
    "Product Insights",
    "Smart Offers",
    "AI Insights"
])

# ---------------------------------------------------
# TAB 1 - SALES ANALYSIS
# ---------------------------------------------------

with tab1:

    st.subheader("Sales by Category")

    category_sales = (
        filtered_df
        .groupby('Category')['Sales']
        .sum()
        .reset_index()
    )

    fig1 = px.bar(
        category_sales,
        x='Category',
        y='Sales',
        color='Category'
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        filtered_df
        .groupby('Month')['Sales']
        .sum()
        .reset_index()
    )

    fig2 = px.line(
        monthly_sales,
        x='Month',
        y='Sales',
        markers=True
    )

    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------
# TAB 2 - PRODUCT INSIGHTS
# ---------------------------------------------------

with tab2:

    st.subheader("Top Selling Products")

    product_analysis = (
        filtered_df
        .groupby('Sub-Category')
        .agg({
            'Sales':'sum',
            'Profit':'sum',
            'Discount':'mean',
            'Quantity':'sum'
        })
        .reset_index()
    )

    top_products = (
        product_analysis
        .sort_values(by='Sales', ascending=False)
        .head(10)
    )

    fig3 = px.bar(
        top_products,
        x='Sales',
        y='Sub-Category',
        orientation='h',
        color='Sales'
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Sales vs Profit Analysis")

    fig4 = px.scatter(
        product_analysis,
        x='Sales',
        y='Profit',
        size='Quantity',
        color='Discount',
        hover_name='Sub-Category'
    )

    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------
# TAB 3 - SMART OFFERS
# ---------------------------------------------------


with tab3:

    st.subheader("Smart Offer Recommendation System")

    st.write(
        """
        Select a product category to view
        smart promotional offers and
        inventory optimization recommendations.
        """
    )

    # ------------------------------------------------
    # CATEGORY-WISE PRODUCT RECOMMENDATION SYSTEM
    # ------------------------------------------------

    recommendation_system = {

        # ------------------------------------------------
        # PHONES
        # ------------------------------------------------

        'Phones': [

            {
                'Product': 'Apple iPhone 5',
                'Offer Product': 'Mobile Accessories',
                'Offer': '5% Off on Mobile Accessories'
            },

            {
                'Product': 'Cisco TelePresence System EX90 Videoconferencing Unit',
                'Offer Product': 'Wireless Headset',
                'Offer': 'Free Wireless Headset'
            }
        ],

        # ------------------------------------------------
        # MACHINES
        # ------------------------------------------------

        'Machines': [

            {
                'Product': 'Canon imageCLASS 2200 Advanced Copier',
                'Offer Product': 'Printing Paper',
                'Offer': 'Free Premium Printing Paper Pack'
            },

            {
                'Product': 'HP Designjet T520 Inkjet Large Format Printer',
                'Offer Product': 'Printer Ink',
                'Offer': '10% Off on Printer Ink'
            },

            {
                'Product': 'Lexmark MX611dhe Monochrome Laser Printer',
                'Offer Product': 'Toner Cartridge',
                'Offer': '5% Off on Toner Cartridge'
            },

            {
                'Product': '3D Systems Cube Printer',
                'Offer Product': 'Printer Filament',
                'Offer': '5% Off on Printer Filament'
            }
        ],

        # ------------------------------------------------
        # TABLES
        # ------------------------------------------------

       'Tables': [

           {
               'Product': 'Hon Multipurpose Training Table',
               'Offer Product': 'Office Chairs',
                'Offer': 'Get 5% Off on Office Chairs'
          }
        ],

        # ------------------------------------------------
        # BINDERS
        # ------------------------------------------------

        'Binders': [

            {
                'Product': 'GBC Ibimaster 500 Manual ProClick Binding System',
                'Offer Product': 'Paper Pack',
                'Offer': 'Free Paper Pack'
            }
        ],

        # ------------------------------------------------
        # BOOKCASES
        # ------------------------------------------------

        'Bookcases': [

            {
                'Product': 'Riverside Palais Royal Lawyers Bookcase',
                'Offer Product': 'Storage Organizer',
                'Offer': 'Free Storage Organizer'
            }
        ],

        # ------------------------------------------------
        # APPLIANCES
        # ------------------------------------------------

        'Appliances': [

            {
                'Product': 'Hoover Commercial Lightweight Upright Vacuum',
                'Offer Product': 'Cleaning Kit',
                'Offer': 'Free Cleaning Kit'
            },

            {
                'Product': 'Eureka Disposable Bags for Sanitaire Vibra Groomer',
                'Offer Product': 'Vacuum Accessories',
                'Offer': 'Buy 1 Get 1 Free'
            }
        ],

        # ------------------------------------------------
        # OFFICE SUPPLIES
        # ------------------------------------------------

        'Office Supplies': [

            {
                'Product': 'Wirebound Voice Message Log Book',
                'Offer Product': 'Pen Set',
                'Offer': 'Buy 1 Get Pen Set Free'
            },

            {
                'Product': 'Avery 5',
                'Offer Product': 'Storage Products',
                'Offer': 'Free with Storage Purchase'
            },

            {
                'Product': 'Xerox 20',
                'Offer Product': 'Printer Paper',
                'Offer': 'Free with Printer Purchase'
            },

            {
                'Product': 'Maxell 4.7GB DVD+R 5/Pack',
                'Offer Product': 'USB Storage',
                'Offer': '5% Combo Discount'
            },

            {
                'Product': 'Acme Serrated Blade Letter Opener',
                'Offer Product': 'Office Kit',
                'Offer': 'Free Office Utility Item'
            },

            {
                'Product': 'Avery Hi-Liter Pen Style Six-Color Fluorescent Set',
                'Offer Product': 'Notebook',
                'Offer': 'Free Mini Notebook'
            },

            {
                'Product': 'Grip Seal Envelopes',
                'Offer Product': 'Paper Bundle',
                'Offer': 'Free Paper Bundle'
            },

            {
                'Product': 'Portable Personal File Box',
                'Offer Product': 'Labels',
                'Offer': 'Free Labels Included'
            }
        ]
    }

    # ------------------------------------------------
    # CATEGORY SELECTION
    # ------------------------------------------------

    selected_category = st.selectbox(
        "Select Product Category",
        list(recommendation_system.keys())
    )

    # ------------------------------------------------
    # DISPLAY PRODUCT OFFERS
    # ------------------------------------------------

    if selected_category:

        st.subheader(
            f"Recommended Offers for {selected_category}"
        )

        for item in recommendation_system[selected_category]:

            st.success(
                f"""
                Main Product:
                {item['Product']}
                """
            )

            st.info(
                f"""
                Recommended Offer Product:
                {item['Offer Product']}

                Offer Strategy:
                {item['Offer']}
                """
            )

            st.markdown("---")

# ---------------------------------------------------
# TAB 4 - AI INSIGHTS
# ---------------------------------------------------

with tab4:

    st.subheader("Product Recommendation Engine")

    product_matrix = pd.pivot_table(
        filtered_df,
        values='Sales',
        index='Order ID',
        columns='Sub-Category',
        aggfunc='sum',
        fill_value=0
    )

    product_correlation = (
        product_matrix.corr()
    )

    selected_ai_product = st.selectbox(
        "Choose Product",
        product_correlation.columns
    )

    related_products = (
        product_correlation[selected_ai_product]
        .sort_values(ascending=False)
        .iloc[1:4]
    )

    st.write(
        f"Recommended Product Offers for {selected_ai_product}"
    )

    for item, score in related_products.items():

        if score > 0.02:
            offer = "20% Combo Discount"

        elif score > 0.01:
            offer = "Buy Together & Save"

        else:
            offer = "Free Add-on Offer"

        st.info(
            f"""
            Buy {selected_ai_product}
            
            Recommended Product:
            {item}
            
            Offer:
            {offer}
            
            Correlation Score:
            {score:.3f}
            """
        )


