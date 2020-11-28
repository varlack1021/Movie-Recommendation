#Development Concepts
- ML Algorithms
    - Content Based Filtering
        +   Based on users past views
    - Collaboritve Based Filtering
        +   Uses other users ratings
    - Cosine Similarity
        + Used to measure how similar two items are. Mathematically, it measures the cosine of the angle between two vectors projected in a multi dimensional space. The output value is between 0 - 1 Where 0
        is no similarity where 1 is 100%

        + Basically what we can do is compare all movies to a movie users like and choose a minimum value for the cosine value to be recommended to the user

- Recommendation Systems
    - Most recommendation systems typically use a **similarity matrix** to make recommendations

##Ratings Algorithm
- My first thought was to simply sort movies based on ratings
- I would assign a weighted rating based on TMDB rating algorithm
    + TMDB wr = (v/v+m *R) + (m/v+m * C)
    + m is the minimum votes required to be listed in the chart, 95 percentile is recommended
    + R is the average rating of the movie
    + C is the mean vote across the whole report

##Concepts used to code
- I needed to vectorize some data to use the similarity matrix
    - Python has a module called TF-IDF Vectorizer
        + A TF-IDF is an abbreviation for Term Frequency Inverse Document Frequency
        + A common algorithm to transform text into meaningful representation of numbers which is used to fit machine learning algorith for prediction
        + The algorithm looks at important words only, usually excludes prepositions

###Cosine Sim in Python
- So cosine sim function is part of the *sklearn.metrics.pairwise*
- This cosine function takes in a list as a paramater and assigns the cosine value as a dot product, this row x col (like a matrix)
- The equation for Cosine Sim is
    + cos(x, y) = x * y / ||x|| * ||y||
    + ||x|| is the length of the vector
    + ||x|| * ||y|| is the cross product

###TF IDF Vectorize Matrix
- Need to do more research on what this returns and what it means
- However, what I need it to put into the cosine_sim function to get cos value
- The function for Tf-IDF Vectorizer is TF * IDF
    + TF is Term Frequency TF() = log2(Freq(i, j)+1)/log2(L)
        * L = Total Number of documents in J
        * Freq(i, j) frequecny of term i in document j
    + Term frequency determines a word relative frequency within a document
    + IDF is the inverse document frequency IDF() = log2(1+Nd/f)
        * Nd = Number of documents considered
        * f = number of documents containing term i
- The function returns a sparse matrix
- Hard to demonstrate with real data since the values change based on how large our dataset is
- A way to improve this would be to include movie ratings from the movies that have similar scores
    + An issue arrises though when a similar movie (like in a series such as the dark knight and batman and robin) when the similar doesn't have a high enough rated score. This is a fault of the dataset we use and how we have weighted scores

###Anothe way to improve
- My theorize right now is the more data will result in higher similarity scores. If i include things like director, keywords, genres, cast will I get a better score for movies? 
- I am also giving more weight to directors since thats what people like and I am doing this by placing the name of the director three times in a list
- For keywords we want words that are greater than 1 and change words to its stem

###Collaboritive Filtering
- Making a hybird interpreter
- 
