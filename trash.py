#%%
def createFeatures(df, text_column):

    df[f'char_count_{text_column}'] = df[text_column].apply(len)
    df[f'word_count_{text_column}'] = df[text_column].apply(lambda x: len(x.split()))
    df[f'unique_word_count_{text_column}'] = df[text_column].apply(lambda x: len(set(x.split())))
    df[f'avg_word_length_{text_column}'] = df[text_column].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()))
    df[f'punctuation_count_{text_column}'] = df[text_column].apply(lambda x: sum(1 for char in x if char in "!?.,;:"))
    df[f'polarity_{text_column}'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)  # Varia entre -1 e 1
    df[f'subjectivity_{text_column}'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.subjectivity)  # Varia entre 0 e 1


    # Inicializar o SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Gerar análise de sentimento
    sentiment_scores = df[text_column].apply(lambda x: sia.polarity_scores(x))

    # Separar as pontuações em colunas individuais
    df['neg'] = sentiment_scores.apply(lambda x: x['neg'])  # Negatividade
    df['neu'] = sentiment_scores.apply(lambda x: x['neu'])  # Neutralidade
    df['pos'] = sentiment_scores.apply(lambda x: x['pos'])  # Positividade
    df['compound'] = sentiment_scores.apply(lambda x: x['compound'])  # Escore geral
    df = df[df['dates_b'] != 'NaT']

    return df.iloc[:,4:]