import streamlit as st
import pandas as pd
import sqlite3
# --- Download NLTK Data ---
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab') # Add this line to fix the error
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import base64
import time

# --- Enhanced Config ---
DB_NAME = "feedback.db"
OFFENSIVE_WORDS = {'hate', 'stupid', 'idiot', 'useless', 'garbage', 'worst', 'awful', 'terrible', 'horrible', 'disgusting'}

ENHANCED_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.15) 0%, transparent 50%);
        z-index: -1;
        animation: backgroundShift 15s ease-in-out infinite;
    }
    
    @keyframes backgroundShift {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(-20px, -20px) rotate(1deg); }
        66% { transform: translate(20px, 10px) rotate(-1deg); }
    }
    
    /* Header Styling */
    h1 {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 3.5rem !important;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 4px 20px rgba(0,0,0,0.1);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(69, 183, 209, 0.5)); }
        to { filter: drop-shadow(0 0 30px rgba(255, 107, 107, 0.5)); }
    }
    
    /* Sidebar Enhancement */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        margin: 15px 0;
        box-shadow: 
            0 8px 32px rgba(0,0,0,0.1),
            inset 0 1px 0 rgba(255,255,255,0.5);
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 
            0 20px 40px rgba(0,0,0,0.2),
            inset 0 1px 0 rgba(255,255,255,0.5);
    }
    
    .metric-card h3 {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card .metric-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 30px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 8px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        background: transparent;
        border-radius: 10px;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.7);
        transition: all 0.3s ease;
        border: none;
        font-size: 1.1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: white;
        background: rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* File Uploader Enhancement */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.1);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Success/Error Messages */
    .stAlert {
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.1), rgba(69, 183, 209, 0.1));
        border-left: 4px solid #4ecdc4;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Warning boxes */
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(255, 159, 67, 0.1));
        border-left: 4px solid #ff6b6b;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Loading spinner */
    .stSpinner {
        color: #667eea;
    }
    
    /* Section headers */
    h2, h3 {
        color: white;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
    }
    
    /* Custom selectbox */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Custom text input */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(255, 255, 255, 0.5);
    }
</style>
"""

# --- Enhanced Functions ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS feedback (Feedback_ID TEXT PRIMARY KEY, Feedback_Text TEXT NOT NULL)')
    conn.commit()
    conn.close()

def clear_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM feedback')
    conn.commit()
    conn.close()

def insert_feedback(df):
    conn = sqlite3.connect(DB_NAME)
    df.to_sql('feedback', conn, if_exists='append', index=False, 
              method=lambda table, conn, keys, data_iter: conn.executemany(
                  f"INSERT OR IGNORE INTO {table.name} ({', '.join(keys)}) VALUES ({', '.join(['?']*len(keys))})", 
                  data_iter))
    conn.close()

def fetch_all_feedback():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    return df

def search_feedback(query_term):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(f"SELECT * FROM feedback WHERE Feedback_Text LIKE '%{query_term}%'", conn)
    conn.close()
    return df

# Initialize NLTK components
try:
    stop_words = set(stopwords.words('english'))
    sentiment_analyzer = SentimentIntensityAnalyzer()
except:
    st.error("Please install required NLTK data. Run: nltk.download('stopwords'), nltk.download('vader_lexicon'), nltk.download('punkt')")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words and word.isalpha()])

def get_sentiment(text):
    score = sentiment_analyzer.polarity_scores(str(text))['compound']
    if score >= 0.05:
        return "Positive 😊"
    elif score <= -0.05:
        return "Negative 😡"
    else:
        return "Neutral 😐"

def flag_offensive(text):
    return any(word in str(text).lower().split() for word in OFFENSIVE_WORDS)

def generate_enhanced_wordcloud(text, colormap='plasma'):
    """Enhanced wordcloud with better styling"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('rgba(0,0,0,0)')
    ax.set_facecolor('rgba(0,0,0,0)')
    
    if text.strip():
        wordcloud = WordCloud(
            width=1000, 
            height=500, 
            background_color=None,
            mode='RGBA',
            colormap=colormap,
            max_words=100,
            relative_scaling=0.5,
            font_path=None,
            max_font_size=60
        ).generate(text)
        
        ax.imshow(wordcloud, interpolation='bilinear')
    else:
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=20, color='gray')
    
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def create_enhanced_download_link(df, filename="processed_feedback.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'''
    <div style="text-align: center; margin: 20px 0;">
        <a href="data:file/csv;base64,{b64}" download="{filename}" 
           style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea, #764ba2); 
                  color: white; text-decoration: none; border-radius: 10px; font-weight: 600;
                  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); transition: all 0.3s ease;">
            📥 Download Processed Data
        </a>
    </div>
    '''

def create_enhanced_metrics(sentiment_counts):
    """Create enhanced metric cards with animations"""
    col1, col2, col3 = st.columns(3)
    
    metrics = [
        ("Positive Feedback", sentiment_counts.get("Positive 😊", 0), "💚"),
        ("Neutral Feedback", sentiment_counts.get("Neutral 😐", 0), "💙"),
        ("Negative Feedback", sentiment_counts.get("Negative 😡", 0), "❤️")
    ]
    
    for col, (title, value, icon) in zip([col1, col2, col3], metrics):
        with col:
            st.markdown(f'''
            <div class="metric-card">
                <h3>{icon} {title}</h3>
                <p class="metric-value">{value:,}</p>
            </div>
            ''', unsafe_allow_html=True)

def create_enhanced_pie_chart(sentiment_counts):
    """Create an enhanced pie chart with better styling"""
    if not sentiment_counts.empty:
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=14,
            marker=dict(line=dict(color='#FFFFFF', width=3))
        )])
        
        fig.update_layout(
            title={
                'text': "Sentiment Distribution",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': 'white'}
            },
            font=dict(color='white', size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        return fig
    return None

def create_enhanced_bar_chart(top_words_df):
    """Create an enhanced bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            y=top_words_df['Word'],
            x=top_words_df['Frequency'],
            orientation='h',
            marker=dict(
                color=top_words_df['Frequency'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Frequency")
            ),
            text=top_words_df['Frequency'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Top 10 Most Common Words',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title='Frequency',
        yaxis_title='Words',
        yaxis={'categoryorder': 'total ascending'},
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    
    return fig

# --- Enhanced Main App ---
def main():
    st.set_page_config(
        page_title="🚀 Advanced Feedback Analyzer", 
        layout="wide", 
        initial_sidebar_state="expanded",
        page_icon="🚀"
    )
    
    st.markdown(ENHANCED_CSS, unsafe_allow_html=True)
    init_db()

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Center")
        
        # File Upload Section
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your feedback CSV file", 
            type=["csv"],
            help="Upload a CSV file containing feedback data"
        )
        
        if uploaded_file is not None:
            if "df_uploaded" not in st.session_state or st.session_state.file_name != uploaded_file.name:
                try:
                    st.session_state.df_uploaded = pd.read_csv(uploaded_file)
                    st.session_state.file_name = uploaded_file.name
                    st.success("✅ File loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
        
        if "df_uploaded" in st.session_state:
            df = st.session_state.df_uploaded
            
            st.markdown("---")
            st.markdown("### 🔗 Column Mapping")
            st.markdown('<div class="info-box">Map your CSV columns to the required fields</div>', unsafe_allow_html=True)
            
            id_col = st.selectbox("🆔 ID Column", df.columns, help="Select the column containing unique identifiers")
            text_col = st.selectbox("📝 Text Column", df.columns, help="Select the column containing feedback text")
            
            if st.button("🚀 Process & Analyze", type="primary"):
                try:
                    with st.spinner('Processing your data...'):
                        processed_df = pd.DataFrame({
                            'Feedback_ID': df[id_col], 
                            'Feedback_Text': df[text_col]
                        })
                        insert_feedback(processed_df)
                        
                        # Clean up session state
                        del st.session_state.df_uploaded
                        del st.session_state.file_name
                        
                        time.sleep(1)  # Brief pause for effect
                        st.success(f"🎉 Successfully processed {len(processed_df):,} feedback entries!")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                        
                except Exception as e:
                    st.error("❌ Processing failed:")
                    st.exception(e)
        
        # Search Section
        st.markdown("---")
        st.markdown("### 🔍 Search & Filter")
        search_term = st.text_input(
            "Search feedback by keyword", 
            placeholder="Enter keywords to search...",
            help="Search through all feedback text"
        )
        
        # Database Management
        st.markdown("---")
        st.markdown("### ⚙️ Database Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Refresh", help="Refresh the data"):
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All", help="Delete all data", type="secondary"):
                clear_db()
                st.success("🧹 Database cleared!")
                time.sleep(1)
                st.rerun()
        
        st.markdown('<div class="warning-box">⚠️ <strong>Disclaimer:</strong> Analysis results are automated and may require human review for accuracy.</div>', unsafe_allow_html=True)

    # Main Content
    st.markdown("# 🚀 Advanced Feedback Analyzer")
    st.markdown("### Transform customer feedback into actionable insights with AI-powered sentiment analysis")

    # Fetch data
    feedback_df = search_feedback(search_term) if search_term else fetch_all_feedback()

    if not feedback_df.empty:
        try:
            # Show processing animation
            with st.spinner('🔮 Analyzing feedback with AI magic...'):
                feedback_df['Cleaned_Text'] = feedback_df['Feedback_Text'].apply(clean_text)
                feedback_df['Sentiment'] = feedback_df['Feedback_Text'].apply(get_sentiment)
                feedback_df['Is_Offensive'] = feedback_df['Feedback_Text'].apply(flag_offensive)
                
                # Add confidence scores
                feedback_df['Confidence_Score'] = feedback_df['Feedback_Text'].apply(
                    lambda x: abs(sentiment_analyzer.polarity_scores(str(x))['compound'])
                )
                
                time.sleep(1)  # Brief pause for effect

            # Enhanced Sentiment Overview
            st.markdown("## 📈 Sentiment Overview")
            sentiment_counts = feedback_df['Sentiment'].value_counts()
            create_enhanced_metrics(sentiment_counts)

            # Enhanced Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "📄 Data Explorer", "💡 Insights", "🎯 Advanced Analytics"])

            with tab1:
                st.markdown("## 🎨 Visual Analysis Dashboard")
                
                # Enhanced Sentiment Distribution
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    fig_pie = create_enhanced_pie_chart(sentiment_counts)
                    if fig_pie:
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Sentiment trend (if we had timestamps)
                    st.markdown("### 🌡️ Sentiment Temperature")
                    positive_pct = (sentiment_counts.get("Positive 😊", 0) / len(feedback_df)) * 100
                    neutral_pct = (sentiment_counts.get("Neutral 😐", 0) / len(feedback_df)) * 100
                    negative_pct = (sentiment_counts.get("Negative 😡", 0) / len(feedback_df)) * 100
                    
                    st.metric("Positive Rate", f"{positive_pct:.1f}%", f"{positive_pct-33.3:.1f}%")
                    st.metric("Neutral Rate", f"{neutral_pct:.1f}%", f"{neutral_pct-33.3:.1f}%")
                    st.metric("Negative Rate", f"{negative_pct:.1f}%", f"{negative_pct-33.3:.1f}%")

                st.markdown("---")
                
                # Enhanced Word Clouds
                st.markdown("### ☁️ Sentiment Word Clouds")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 💚 Positive Vibes")
                    positive_text = " ".join(feedback_df[feedback_df['Sentiment'] == 'Positive 😊']['Cleaned_Text'])
                    if positive_text.strip():
                        fig_pos_wc = generate_enhanced_wordcloud(positive_text, colormap='Greens')
                        st.pyplot(fig_pos_wc, use_container_width=True)
                    else:
                        st.info("🤔 No positive feedback available for word cloud generation")

                with col2:
                    st.markdown("#### ❤️ Areas for Improvement")
                    negative_text = " ".join(feedback_df[feedback_df['Sentiment'] == 'Negative 😡']['Cleaned_Text'])
                    if negative_text.strip():
                        fig_neg_wc = generate_enhanced_wordcloud(negative_text, colormap='Reds')
                        st.pyplot(fig_neg_wc, use_container_width=True)
                    else:
                        st.info("🎉 No negative feedback - that's great news!")

                st.markdown("---")
                
                # Enhanced Word Frequency
                st.markdown("### 🔤 Most Discussed Topics")
                all_words = " ".join(feedback_df[feedback_df['Cleaned_Text'].notna()]['Cleaned_Text']).split()
                
                if all_words:
                    word_freq = Counter(all_words)
                    top_words_df = pd.DataFrame(word_freq.most_common(10), columns=['Word', 'Frequency'])
                    
                    fig_bar = create_enhanced_bar_chart(top_words_df)
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Interactive word exploration
                    st.markdown("#### 🔍 Explore Feedback by Topic")
                    selected_word = st.selectbox(
                        "Select a word to see related feedback:", 
                        options=[""] + top_words_df['Word'].tolist(),
                        help="Choose a word from the chart above to see related comments"
                    )
                    
                    if selected_word:
                        related_comments = feedback_df[
                            feedback_df['Feedback_Text'].str.contains(fr'\b{selected_word}\b', case=False, na=False)
                        ]
                        
                        if not related_comments.empty:
                            st.markdown(f"**Found {len(related_comments)} comments containing '{selected_word}':**")
                            st.dataframe(
                                related_comments[['Feedback_ID', 'Feedback_Text', 'Sentiment', 'Confidence_Score']], 
                                use_container_width=True, 
                                height=300
                            )
                        else:
                            st.info(f"No comments found containing '{selected_word}'")

            with tab2:
                st.markdown("## 📄 Data Explorer")
                
                # Enhanced filtering
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_filter = st.multiselect(
                        "🎭 Filter by Sentiment:", 
                        options=feedback_df['Sentiment'].unique(), 
                        default=list(feedback_df['Sentiment'].unique()),
                        help="Select sentiments to display"
                    )
                
                with col2:
                    show_offensive = st.checkbox("🚨 Show only flagged content", help="Display only potentially offensive content")
                
                with col3:
                    min_confidence = st.slider("🎯 Minimum Confidence", 0.0, 1.0, 0.0, 0.1, help="Filter by sentiment confidence score")
                
                # Apply filters
                filtered_df = feedback_df[
                    (feedback_df['Sentiment'].isin(sentiment_filter)) &
                    (feedback_df['Confidence_Score'] >= min_confidence)
                ]
                
                if show_offensive:
                    filtered_df = filtered_df[filtered_df['Is_Offensive'] == True]
                
                st.markdown(f"### 📊 Showing {len(filtered_df):,} of {len(feedback_df):,} feedback entries")
                
                if not filtered_df.empty:
                    st.dataframe(
                        filtered_df[['Feedback_ID', 'Feedback_Text', 'Sentiment', 'Confidence_Score', 'Is_Offensive']], 
                        use_container_width=True,
                        height=400
                    )
                    
                    st.markdown(create_enhanced_download_link(filtered_df), unsafe_allow_html=True)
                else:
                    st.info("🔍 No data matches your current filters. Try adjusting the criteria.")

            with tab3:
                st.markdown("## 💡 AI-Powered Insights")
                
                # Key insights cards
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Key Statistics")
                    total_feedback = len(feedback_df)
                    offensive_count = feedback_df['Is_Offensive'].sum()
                    avg_confidence = feedback_df['Confidence_Score'].mean()
                    dominant_sentiment = sentiment_counts.idxmax() if not sentiment_counts.empty else "N/A"
                    
                    st.markdown(f'''
                    <div class="info-box">
                    <h4>📊 Summary Report</h4>
                    <ul>
                        <li><strong>Total Feedback:</strong> {total_feedback:,} entries</li>
                        <li><strong>Dominant Sentiment:</strong> {dominant_sentiment.split(' ')[0] if dominant_sentiment != "N/A" else "N/A"}</li>
                        <li><strong>Average Confidence:</strong> {avg_confidence:.2f}/1.0</li>
                        <li><strong>Flagged Content:</strong> {offensive_count} ({(offensive_count/total_feedback*100):.1f}%)</li>
                    </ul>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 🎯 Recommendations")
                    
                    # Generate dynamic recommendations
                    recommendations = []
                    
                    if sentiment_counts.get("Negative 😡", 0) > sentiment_counts.get("Positive 😊", 0):
                        recommendations.append("🔴 **Priority:** Address negative feedback - it exceeds positive feedback")
                    
                    if offensive_count > 0:
                        recommendations.append(f"⚠️ **Alert:** {offensive_count} potentially offensive comments need review")
                    
                    if avg_confidence < 0.5:
                        recommendations.append("📝 **Note:** Low confidence scores suggest mixed sentiment - manual review recommended")
                    
                    if sentiment_counts.get("Positive 😊", 0) > total_feedback * 0.6:
                        recommendations.append("🎉 **Great news:** Strong positive sentiment detected!")
                    
                    if not recommendations:
                        recommendations.append("✅ **All good:** Balanced feedback distribution detected")
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {rec}")
                
                # Trend analysis
                st.markdown("### 📈 Sentiment Breakdown")
                
                # Create detailed sentiment analysis
                sentiment_details = []
                for sentiment in feedback_df['Sentiment'].unique():
                    subset = feedback_df[feedback_df['Sentiment'] == sentiment]
                    avg_conf = subset['Confidence_Score'].mean()
                    count = len(subset)
                    
                    sentiment_details.append({
                        'Sentiment': sentiment,
                        'Count': count,
                        'Percentage': f"{(count/total_feedback)*100:.1f}%",
                        'Avg Confidence': f"{avg_conf:.2f}",
                        'Top Words': ', '.join([word for word, _ in Counter(' '.join(subset['Cleaned_Text'])).most_common(3)])
                    })
                
                sentiment_detail_df = pd.DataFrame(sentiment_details)
                st.dataframe(sentiment_detail_df, use_container_width=True)

            with tab4:
                st.markdown("## 🎯 Advanced Analytics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Confidence Distribution")
                    
                    # Confidence histogram
                    fig_conf = px.histogram(
                        feedback_df, 
                        x='Confidence_Score', 
                        nbins=20,
                        title='Sentiment Confidence Distribution',
                        color_discrete_sequence=['#667eea']
                    )
                    
                    fig_conf.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title_font=dict(color='white', size=16)
                    )
                    
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                with col2:
                    st.markdown("### 🔍 Text Length Analysis")
                    
                    # Add text length analysis
                    feedback_df['Text_Length'] = feedback_df['Feedback_Text'].str.len()
                    
                    fig_length = px.box(
                        feedback_df,
                        y='Text_Length',
                        color='Sentiment',
                        title='Text Length by Sentiment',
                        color_discrete_map={
                            "Positive 😊": "#2ecc71",
                            "Negative 😡": "#e74c3c", 
                            "Neutral 😐": "#95a5a6"
                        }
                    )
                    
                    fig_length.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title_font=dict(color='white', size=16)
                    )
                    
                    st.plotly_chart(fig_length, use_container_width=True)
                
                # Advanced word analysis
                st.markdown("### 🔤 Advanced Word Analysis")
                
                tab_pos, tab_neg, tab_neu = st.tabs(["😊 Positive Words", "😡 Negative Words", "😐 Neutral Words"])
                
                with tab_pos:
                    positive_df = feedback_df[feedback_df['Sentiment'] == 'Positive 😊']
                    if not positive_df.empty:
                        pos_words = ' '.join(positive_df['Cleaned_Text']).split()
                        pos_freq = Counter(pos_words)
                        pos_top = pd.DataFrame(pos_freq.most_common(15), columns=['Word', 'Frequency'])
                        
                        fig_pos = px.treemap(
                            pos_top, 
                            path=['Word'], 
                            values='Frequency',
                            title='Most Common Positive Words (Treemap)',
                            color='Frequency',
                            color_continuous_scale='Greens'
                        )
                        
                        fig_pos.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            title_font=dict(color='white', size=16)
                        )
                        
                        st.plotly_chart(fig_pos, use_container_width=True)
                    else:
                        st.info("No positive feedback available for analysis")
                
                with tab_neg:
                    negative_df = feedback_df[feedback_df['Sentiment'] == 'Negative 😡']
                    if not negative_df.empty:
                        neg_words = ' '.join(negative_df['Cleaned_Text']).split()
                        neg_freq = Counter(neg_words)
                        neg_top = pd.DataFrame(neg_freq.most_common(15), columns=['Word', 'Frequency'])
                        
                        fig_neg = px.treemap(
                            neg_top, 
                            path=['Word'], 
                            values='Frequency',
                            title='Most Common Negative Words (Treemap)',
                            color='Frequency',
                            color_continuous_scale='Reds'
                        )
                        
                        fig_neg.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            title_font=dict(color='white', size=16)
                        )
                        
                        st.plotly_chart(fig_neg, use_container_width=True)
                    else:
                        st.info("🎉 No negative feedback found!")
                
                with tab_neu:
                    neutral_df = feedback_df[feedback_df['Sentiment'] == 'Neutral 😐']
                    if not neutral_df.empty:
                        neu_words = ' '.join(neutral_df['Cleaned_Text']).split()
                        neu_freq = Counter(neu_words)
                        neu_top = pd.DataFrame(neu_freq.most_common(15), columns=['Word', 'Frequency'])
                        
                        fig_neu = px.treemap(
                            neu_top, 
                            path=['Word'], 
                            values='Frequency',
                            title='Most Common Neutral Words (Treemap)',
                            color='Frequency',
                            color_continuous_scale='Blues'
                        )
                        
                        fig_neu.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            title_font=dict(color='white', size=16)
                        )
                        
                        st.plotly_chart(fig_neu, use_container_width=True)
                    else:
                        st.info("No neutral feedback available for analysis")

        except Exception as e:
            st.error("🚨 An error occurred during analysis:")
            st.exception(e)
            st.markdown('''
            <div class="warning-box">
            <h4>Troubleshooting Tips:</h4>
            <ul>
                <li>Ensure your CSV has the correct format</li>
                <li>Check that text columns contain valid data</li>
                <li>Try clearing the database and re-uploading</li>
                <li>Verify NLTK packages are installed</li>
            </ul>
            </div>
            ''', unsafe_allow_html=True)

    else:
        # Enhanced empty state
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; background: rgba(255,255,255,0.1); border-radius: 20px; backdrop-filter: blur(20px); margin: 40px 0;">
            <h2 style="color: white; font-size: 2.5rem; margin-bottom: 20px;">🚀 Ready to Analyze Feedback?</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem; margin-bottom: 30px;">
                Upload your CSV file to get started with AI-powered sentiment analysis
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 30px;">
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; max-width: 300px;">
                    <h4 style="color: #4ecdc4; margin-bottom: 10px;">📁 Step 1</h4>
                    <p style="color: white; margin: 0;">Upload your CSV file using the sidebar</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; max-width: 300px;">
                    <h4 style="color: #45b7d1; margin-bottom: 10px;">🔗 Step 2</h4>
                    <p style="color: white; margin: 0;">Map your ID and text columns</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; max-width: 300px;">
                    <h4 style="color: #ff6b6b; margin-bottom: 10px;">🚀 Step 3</h4>
                    <p style="color: white; margin: 0;">Click process to start analysis</p>
                </div>
            </div>
            <div style="margin-top: 40px; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 10px;">
                <h4 style="color: white; margin-bottom: 15px;">✨ What you'll get:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: left;">
                    <div>📊 <strong>Sentiment Analysis:</strong> Positive, negative, neutral classification</div>
                    <div>☁️ <strong>Word Clouds:</strong> Visual representation of key terms</div>
                    <div>📈 <strong>Interactive Charts:</strong> Beautiful data visualizations</div>
                    <div>🎯 <strong>Smart Insights:</strong> AI-generated recommendations</div>
                    <div>🔍 <strong>Advanced Search:</strong> Filter and explore your data</div>
                    <div>📥 <strong>Export Options:</strong> Download processed results</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.6);">
        <p>🚀 Enhanced Feedback Analyzer | Built with Streamlit & AI | 
        <span style="color: #ff6b6b;">❤️</span> Made with passion for data insights</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
