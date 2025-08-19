# Advanced Feedback Analyzer

> Transform your customer feedback into actionable insights with AI-powered sentiment analysis!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Maintained](https://img.shields.io/badge/Maintained-Yes-brightgreen.svg)

## 📖 What is this?

Ever wondered what your customers *really* think about your product or service? This beautiful, AI-powered web application takes your customer feedback and turns it into crystal-clear insights. No more drowning in spreadsheets or trying to read between the lines – let AI do the heavy lifting!

### ✨ What makes this special?

- **🎨 Beautiful UI**: Modern glassmorphic design that's actually pleasant to use
- **🤖 Smart AI**: Powered by VADER sentiment analysis for accurate emotion detection
- **📊 Rich Visualizations**: Interactive charts, word clouds, and treemaps
- **🔍 Deep Insights**: Not just "positive" or "negative" – get *real* recommendations
- **⚡ Super Fast**: SQLite database for lightning-quick analysis
- **🛡️ Content Filtering**: Automatically flags potentially offensive content

## 🎯 Perfect for:

- **Customer Success Teams** analyzing support tickets
- **Product Managers** understanding user feedback
- **Marketing Teams** gauging campaign sentiment
- **Researchers** studying survey responses
- **Anyone** who wants to understand text data better!

## 🚀 Quick Start (3 minutes to insights!)

### Step 1: Get the code
```bash
git clone https://github.com/yourusername/feedback-analyzer.git
cd feedback-analyzer
```

### Step 2: Set up your environment
```bash
# Create a virtual environment (recommended)
python -m venv feedback_env
source feedback_env/bin/activate  # On Windows: feedback_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Download NLTK data
```python
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
```

### Step 4: Launch the app!
```bash
streamlit run main_app.py
```

That's it! Your browser will open to `http://localhost:8501` and you'll see the beautiful interface waiting for your data.

## 📁 What kind of data do you need?

Super simple! Just a CSV file with:
- **An ID column** (customer ID, ticket number, whatever unique identifier you have)
- **A text column** (the actual feedback, review, comment, etc.)

### Example CSV format:
```csv
customer_id,feedback_text
1001,"I absolutely love this product! Amazing quality."
1002,"The service was okay, nothing special."
1003,"Terrible experience, very disappointed."
```

## 🎮 How to use it

### 1. Upload your data
- Click the file upload area in the sidebar
- Select your CSV file
- Map your columns (tell us which column is the ID and which is the text)

### 2. Watch the magic happen
- Click "Process & Analyze" 
- Grab a coffee while AI analyzes your feedback
- Enjoy the confetti celebration! 🎉

### 3. Explore your insights
- **📊 Visualizations**: Beautiful charts showing sentiment distribution
- **📄 Data Explorer**: Filter and search through your feedback
- **💡 Insights**: AI-generated recommendations and key findings  
- **🎯 Advanced Analytics**: Deep dive into confidence scores and word patterns

## 🛠️ Features Deep Dive

### 🎨 Stunning Visualizations
- **Donut Charts**: See sentiment distribution at a glance
- **Word Clouds**: Beautiful visual representation of common themes
- **Interactive Bar Charts**: Explore the most discussed topics
- **Treemaps**: Advanced word frequency analysis
- **Box Plots**: Understand text length patterns by sentiment

### 🧠 Smart Analysis
- **Sentiment Classification**: Positive 😊, Negative 😡, Neutral 😐
- **Confidence Scoring**: How sure is the AI about each classification?
- **Offensive Content Detection**: Automatically flags problematic feedback
- **Topic Extraction**: Find the most discussed themes
- **Trend Analysis**: Understand your feedback patterns

### 🔍 Powerful Search & Filtering
- **Keyword Search**: Find specific topics instantly
- **Sentiment Filtering**: Focus on positive or negative feedback
- **Confidence Filtering**: See only high-confidence classifications
- **Export Options**: Download your analyzed data

## 🚀 Live Demo

Experience the **MVP Website User Feedback Analyzer** live in action! Click the link below to access the interactive web application.

[![Open Live App](https://img.shields.io/badge/Live%20Demo-%F0%9F%9A%80-brightgreen?style=for-the-badge)]([https://python-feedback-analyzer.streamlit.app])

**➡️ Direct Link: [https://python-feedback-analyzer.streamlit.app](https://python-feedback-analyzer.streamlit.app)**

> **Note:** If the project is hosted on a temporary service like `streamlit`, this link may be inactive. To run your own instance, please follow the setup instructions in this README.

## 🤝 Contributing

Found a bug? Have an idea for improvement? We'd love your help!

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin amazing-feature`)
5. **Open** a Pull Request

### 🐛 Reporting Issues
If something's not working right, please [open an issue](https://github.com/yourusername/feedback-analyzer/issues) and include:
- Your operating system
- Python version
- Steps to reproduce the problem
- Any error messages you see

## 📋 Requirements

- **Python 3.8+** (we recommend 3.9 or newer)
- **8GB RAM minimum** (for larger datasets)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

## 🔧 Troubleshooting

### Common Issues & Solutions

**"ModuleNotFoundError: No module named 'nltk'"**
```bash
pip install nltk
# Then in Python:
import nltk
nltk.download('all')
```

**"Database is locked" error**
- Close any other instances of the app
- Restart the application

**Charts not displaying properly**
- Clear your browser cache
- Try a different browser
- Check that all required packages are installed

**App running slowly with large files**
- Try with a smaller sample first
- Ensure you have sufficient RAM
- Close other applications

## 📚 Technical Details

### Architecture
- **Frontend**: Streamlit with custom CSS for modern UI
- **Backend**: Python with SQLite for data storage
- **AI Engine**: NLTK's VADER sentiment analyzer
- **Visualizations**: Plotly for interactive charts, Matplotlib for word clouds

### Performance
- Processes up to 10,000 feedback entries smoothly
- Real-time search and filtering
- Efficient SQLite database storage
- Optimized memory usage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VADER Sentiment Analysis** for accurate emotion detection
- **Streamlit** for making beautiful web apps in Python simple
- **Plotly** for stunning interactive visualizations
- **The open-source community** for all the amazing libraries



## 🎉 What's Next?

We're constantly improving! Here's what's coming:

- [ ] **Real-time feedback analysis** via API integration
- [ ] **Multi-language support** for global feedback
- [ ] **Advanced ML models** for even better accuracy
- [ ] **Team collaboration features** for shared insights
- [ ] **Custom sentiment categories** beyond positive/negative/neutral

---

### ⭐ If this helped you, please star this repository! 

It helps others discover this tool and motivates us to keep improving it.

**Made with ❤️ and lots of ☕ by passionate developers who believe in the power of understanding customer feedback.**
