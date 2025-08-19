import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import os

# --- Define the file paths in your Google Drive ---
feedback_file_path = '/content/drive/My Drive/Colab Notebooks/student_feedback.csv'
satisfaction_file_path = '/content/drive/My Drive/Colab Notebooks/Student_Satisfaction_Survey.csv'

# Check if the files exist at the specified paths
if not os.path.exists(feedback_file_path):
    print(f"Error: The file {feedback_file_path} was not found.")
if not os.path.exists(satisfaction_file_path):
    print(f"Error: The file {satisfaction_file_path} was not found.")
else:
    # --- Step 1: Data Cleaning and Preparation ---

    # Load and clean student_feedback.csv
    df_feedback = pd.read_csv(feedback_file_path)
    df_feedback = df_feedback.drop(columns=['Unnamed: 0'])
    df_feedback.columns = df_feedback.columns.str.lower().str.replace(' ', '_')

    # Load and clean Student_Satisfaction_Survey.csv
    df_satisfaction = pd.read_csv(satisfaction_file_path, encoding='latin1')
    df_satisfaction.columns = df_satisfaction.columns.str.lower().str.replace(' ', '_').str.replace('/', '_')
    df_satisfaction['average_score'] = df_satisfaction['average__percentage'].apply(lambda x: float(x.split(' / ')[0]))

    # --- Step 2: NLP - Sentiment Analysis and Word Cloud ---

    # Create a function to get sentiment polarity
    def get_sentiment(text):
        if pd.isna(text):
            return 0
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    # Apply sentiment analysis to the 'questions' column
    df_satisfaction['sentiment_score'] = df_satisfaction['questions'].apply(get_sentiment)

    # Classify sentiment based on polarity score
    def get_sentiment_label(score):
        if score > 0:
            return 'Positive'
        elif score < 0:
            return 'Negative'
        else:
            return 'Neutral'

    df_satisfaction['sentiment'] = df_satisfaction['sentiment_score'].apply(get_sentiment_label)

    # Summarize the sentiment
    sentiment_counts = df_satisfaction['sentiment'].value_counts()
    print('Sentiment Analysis Summary:')
    print(sentiment_counts.to_markdown(numalign="left", stralign="left"))

    # Generate the word cloud
    all_questions = ' '.join(df_satisfaction['questions'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_questions)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Survey Questions')
    plt.tight_layout()
    plt.show()

    # --- Step 3: Visualize Trends (All 8 graphs) ---

    # Graph 1: Bar chart for student feedback average ratings
    rating_columns = [
        'well_versed_with_the_subject',
        'explains_concepts_in_an_understandable_way',
        'use_of_presentations',
        'degree_of_difficulty_of_assignments',
        'solves_doubts_willingly',
        'structuring_of_the_course',
        'provides_support_for_students_going_above_and_beyond',
        'course_recommendation_based_on_relevance'
    ]
    average_ratings = df_feedback[rating_columns].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(average_ratings.index.str.replace('_', ' ').str.title(), average_ratings.values, color='skyblue')
    plt.xlabel('Average Rating (1-10 Scale)')
    plt.ylabel('Category')
    plt.title('Average Student Feedback Ratings')
    plt.tight_layout()
    plt.show()

    # Graph 2: Bar chart for top 5 and bottom 5 rated questions
    top_rated_questions = df_satisfaction.sort_values(by='average_score', ascending=False).head(5)
    bottom_rated_questions = df_satisfaction.sort_values(by='average_score', ascending=True).head(5)
    combined_questions = pd.concat([top_rated_questions, bottom_rated_questions])
    plt.figure(figsize=(12, 8))
    plt.barh(combined_questions['questions'], combined_questions['average_score'], color=['green'] * 5 + ['red'] * 5)
    plt.xlabel('Average Score (1-5 Scale)')
    plt.ylabel('Question')
    plt.title('Top 5 and Bottom 5 Rated Questions')
    plt.tight_layout()
    plt.show()

    # Graph 3: Box plot for a few selected feedback categories
    selected_feedback_columns = [
        'well_versed_with_the_subject',
        'solves_doubts_willingly',
        'degree_of_difficulty_of_assignments'
    ]
    df_selected_feedback = df_feedback[selected_feedback_columns]
    plt.figure(figsize=(10, 6))
    df_selected_feedback.columns = df_selected_feedback.columns.str.replace('_', ' ').str.title()
    df_selected_feedback.boxplot()
    plt.title('Distribution of Student Feedback Ratings (Box Plot)')
    plt.xlabel('Feedback Category')
    plt.ylabel('Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Graph 5: Histogram of all average satisfaction scores
    plt.figure(figsize=(10, 6))
    plt.hist(df_satisfaction['average_score'], bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel('Average Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Satisfaction Scores')
    plt.tight_layout()
    plt.show()

    # Graph 6: Histograms for two specific feedback categories
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df_feedback['well_versed_with_the_subject'], kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribution of "Well Versed with the Subject"')
    axes[0].set_xlabel('Rating')
    sns.histplot(df_feedback['degree_of_difficulty_of_assignments'], kde=True, ax=axes[1], color='red')
    axes[1].set_title('Distribution of "Degree of Difficulty of Assignments"')
    axes[1].set_xlabel('Rating')
    plt.tight_layout()
    plt.show()

    # Graph 7: Correlation heatmap
    correlation_matrix = df_feedback[rating_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Student Feedback Ratings')
    plt.tight_layout()
    plt.show()