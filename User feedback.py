import streamlit as st
import pandas as pd
import sqlite3
import nltk
import time
import base64
import re
from collections import Counter

# --- NEW: Import TextBlob for sentiment analysis ---
from textblob import TextBlob

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- Download necessary NLTK data ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')


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

# Initialize NLTK components for text cleaning
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    st.error("NLTK 'stopwords' not found. Please ensure you have an internet connection.")
    stop_words = set()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    try:
        tokens = word_tokenize(text)
        return " ".join([word for word in tokens if word not in stop_words and word.isalpha()])
    except LookupError:
        st.error("NLTK 'punkt' model not found. Please ensure you have an internet connection.")
        return ""

# --- MODIFIED: get_sentiment function now uses TextBlob ---
def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def flag_offensive(text):
    return any(word in str(text).lower().split() for word in OFFENSIVE_WORDS)

def generate_enhanced_wordcloud(text, colormap='plasma'):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor((0, 0, 0, 0))
    ax.set_facecolor((0, 0, 0, 0))
    if text.strip():
        wordcloud = WordCloud(width=1000, height=500, background_color=None, mode='RGBA', colormap=colormap, max_words=100, relative_scaling=0.5, font_path=None, max_font_size=60).generate(text)
        ax.imshow(wordcloud, interpolation='bilinear')
    else:
        ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20, color='gray')
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
    col1, col2, col3 = st.columns(3)
    metrics = [
        ("Positive Feedback", sentiment_counts.get("Positive", 0), "💚"),
        ("Neutral Feedback", sentiment_counts.get("Neutral", 0), "💙"),
        ("Negative Feedback", sentiment_counts.get("Negative", 0), "❤️")
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
    if not sentiment_counts.empty:
        color_map = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
        colors = [color_map.get(label, '#3498db') for label in sentiment_counts.index]
        fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, hole=0.4, marker_colors=colors, textinfo='label+percent', textfont_size=14, marker=dict(line=dict(color='#FFFFFF', width=3)))])
        fig.update_layout(title={'text': "Sentiment Distribution", 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20, 'color': 'white'}}, font=dict(color='white', size=12), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
        return fig
    return None

def create_enhanced_bar_chart(top_words_df):
    fig = go.Figure(data=[go.Bar(y=top_words_df['Word'], x=top_words_df['Frequency'], orientation='h', marker=dict(color=top_words_df['Frequency'], colorscale='Viridis', showscale=True, colorbar=dict(title="Frequency")), text=top_words_df['Frequency'], textposition='auto')])
    fig.update_layout(title={'text': 'Top 10 Most Common Words', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20, 'color': 'white'}}, xaxis_title='Frequency', yaxis_title='Words', yaxis={'categoryorder': 'total ascending'}, font=dict(color='white'), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500)
    return fig

# --- Enhanced Main App ---
def main():
    st.set_page_config(page_title="🚀 Advanced Feedback Analyzer", layout="wide", initial_sidebar_state="expanded", page_icon="🚀")
    st.markdown(ENHANCED_CSS, unsafe_allow_html=True)
    init_db()

    with st.sidebar:
        st.markdown("## 🎛️ Control Center")
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader("Upload your feedback CSV file", type=["csv"], help="Upload a CSV file containing feedback data")
        
        if uploaded_file is not None:
            if "df_uploaded" not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
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
            
            options = list(df.columns)
            default_id_col_index, default_text_col_index = 0, 0
            for i, col in enumerate(options):
                col_lower = col.lower()
                if 'id' in col_lower or 'identifier' in col_lower: default_id_col_index = i
                if ('text' in col_lower or 'review' in col_lower or 'comment' in col_lower) and 'id' not in col_lower: default_text_col_index = i

            id_col = st.selectbox("🆔 ID Column", options, index=default_id_col_index, help="Select the column containing unique identifiers")
            text_col = st.selectbox("📝 Text Column", options, index=default_text_col_index, help="Select the column containing feedback text")
            
            if st.button("🚀 Process & Analyze", type="primary"):
                if pd.api.types.is_numeric_dtype(df[text_col]) and not pd.api.types.is_string_dtype(df[text_col]):
                    st.error(f"❌ Validation Error: The selected 'Text Column' ('{text_col}') appears to contain only numbers. Please select the correct column that contains the feedback text.")
                else:
                    try:
                        with st.spinner('Processing your data...'):
                            temp_df = df[[id_col, text_col]].copy()
                            temp_df[text_col] = temp_df[text_col].fillna('').astype(str)
                            processed_df = pd.DataFrame({'Feedback_ID': temp_df[id_col].astype(str), 'Feedback_Text': temp_df[text_col]})
                            
                            insert_feedback(processed_df)
                            
                            if "df_uploaded" in st.session_state: del st.session_state.df_uploaded
                            if "file_name" in st.session_state: del st.session_state.file_name
                            
                            st.success(f"🎉 Successfully processed {len(processed_df):,} feedback entries!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Processing failed: {e}")
        
        st.markdown("---")
        st.markdown("### 🔍 Search & Filter")
        search_term = st.text_input("Search feedback by keyword", placeholder="Enter keywords to search...", help="Search through all feedback text")
        
        st.markdown("---")
        st.markdown("### ⚙️ Database Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Refresh", help="Refresh the data"): st.rerun()
        with col2:
            if st.button("🗑️ Clear All", help="Delete all data", type="secondary"):
                clear_db()
                st.success("🧹 Database cleared!")
                time.sleep(1)
                st.rerun()
        
        st.markdown('<div class="warning-box">⚠️ <strong>Disclaimer:</strong> Analysis results are automated and may require human review for accuracy.</div>', unsafe_allow_html=True)

    st.markdown("# 🚀 Advanced Feedback Analyzer")
    st.markdown("### Transform customer feedback into actionable insights with AI-powered sentiment analysis")

    feedback_df = search_feedback(search_term) if search_term else fetch_all_feedback()

    if not feedback_df.empty:
        try:
            with st.spinner('🔮 Analyzing feedback with AI magic...'):
                feedback_df['Cleaned_Text'] = feedback_df['Feedback_Text'].apply(clean_text)
                feedback_df['Sentiment'] = feedback_df['Feedback_Text'].apply(get_sentiment)
                feedback_df['Is_Offensive'] = feedback_df['Feedback_Text'].apply(flag_offensive)
                feedback_df['Confidence_Score'] = feedback_df['Feedback_Text'].apply(lambda x: abs(TextBlob(str(x)).sentiment.polarity))

            st.markdown("## 📈 Sentiment Overview")
            sentiment_counts = feedback_df['Sentiment'].value_counts()
            
            st.markdown("### Final Result:")
            total_reviews = len(feedback_df)
            positive_pct = (sentiment_counts.get("Positive", 0) / total_reviews) * 100 if total_reviews > 0 else 0
            negative_pct = (sentiment_counts.get("Negative", 0) / total_reviews) * 100 if total_reviews > 0 else 0
            neutral_pct = (sentiment_counts.get("Neutral", 0) / total_reviews) * 100 if total_reviews > 0 else 0
            st.markdown(f"**Positive: {positive_pct:.1f}%**")
            st.markdown(f"**Negative: {negative_pct:.1f}%**")
            st.markdown(f"**Neutral: {neutral_pct:.1f}%**")
            
            create_enhanced_metrics(sentiment_counts)

            tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualizations", "📄 Data Explorer", "💡 Insights", "🎯 Advanced Analytics"])

            with tab1:
                st.markdown("## 🎨 Visual Analysis Dashboard")
                col1, col2 = st.columns([1, 1])
                with col1:
                    fig_pie = create_enhanced_pie_chart(sentiment_counts)
                    if fig_pie: st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    st.markdown("### 🌡️ Sentiment Temperature")
                    st.metric("Positive Rate", f"{positive_pct:.1f}%")
                    st.metric("Neutral Rate", f"{neutral_pct:.1f}%")
                    st.metric("Negative Rate", f"{negative_pct:.1f}%")

                st.markdown("---")
                st.markdown("### ☁️ Sentiment Word Clouds")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 💚 Positive Vibes")
                    positive_text = " ".join(feedback_df[feedback_df['Sentiment'] == 'Positive']['Cleaned_Text'])
                    if positive_text.strip():
                        st.pyplot(generate_enhanced_wordcloud(positive_text, colormap='Greens'), use_container_width=True)
                    else:
                        st.info("🤔 No positive feedback available.")
                with col2:
                    st.markdown("#### ❤️ Areas for Improvement")
                    negative_text = " ".join(feedback_df[feedback_df['Sentiment'] == 'Negative']['Cleaned_Text'])
                    if negative_text.strip():
                        st.pyplot(generate_enhanced_wordcloud(negative_text, colormap='Reds'), use_container_width=True)
                    else:
                        st.info("🎉 No negative feedback!")

                st.markdown("---")
                st.markdown("### 🔤 Most Discussed Topics")
                all_words = " ".join(feedback_df[feedback_df['Cleaned_Text'].notna()]['Cleaned_Text']).split()
                if all_words:
                    top_words_df = pd.DataFrame(Counter(all_words).most_common(10), columns=['Word', 'Frequency'])
                    st.plotly_chart(create_enhanced_bar_chart(top_words_df), use_container_width=True)
                    st.markdown("#### 🔍 Explore Feedback by Topic")
                    selected_word = st.selectbox("Select a word:", options=[""] + top_words_df['Word'].tolist())
                    if selected_word:
                        related_comments = feedback_df[feedback_df['Feedback_Text'].str.contains(fr'\b{selected_word}\b', case=False, na=False)]
                        if not related_comments.empty:
                            st.markdown(f"**Found {len(related_comments)} comments containing '{selected_word}':**")
                            st.dataframe(related_comments[['Feedback_ID', 'Feedback_Text', 'Sentiment', 'Confidence_Score']], use_container_width=True, height=300)
        except Exception as e:
            st.error(f"🚨 An error occurred during analysis: {e}")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; background: rgba(255,255,255,0.1); border-radius: 20px; backdrop-filter: blur(20px); margin: 40px 0;">
            <h2 style="color: white; font-size: 2.5rem; margin-bottom: 20px;">🚀 Ready to Analyze Feedback?</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">Upload your CSV file to get started.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.6);">
        <p>Enhanced Feedback Analyzer | Built with Streamlit| 
        <span style="color: #ff6b6b;">❤️</span> Made By Atharva Borhade</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
