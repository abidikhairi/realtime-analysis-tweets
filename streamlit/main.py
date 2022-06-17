from multipage import MultiPage
from pages import sentiment_analysis, emotion_analysis

app = MultiPage()

app.add_page('Sentiment Analysis', sentiment_analysis.app)
app.add_page('Emotion Analysis', emotion_analysis.app)

app.run()

