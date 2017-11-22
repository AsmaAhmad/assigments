
# coding: utf-8

# # Data Science 
# # Assignment '#2 -  Exploratory Data Analysis
# 
# 

# In[1]:


get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#display wide tables 
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)


# We have a list of 10,000 movies with IMDB user rating as imdb.txt. We want to perform a exploratory data analysis of this data in Python by using its Pandas library.  We will perform the cleaning, transformation and then visualization on the raw data. This will help us to understand the data for further processing.

# In[3]:


#!head imdb.txt


# ## 1. Loading data
# 
# Read the imdb.txt into dataframe named data. The data is tab delimited. The columns names are 'imdbID', 'title', 'year', 'score', 'votes', 'runtime', 'genres'

# In[2]:


# Your code here
data = pd.read_csv('imdb.txt', sep="\t", header=None)
data.columns = ["ImdbID", "Title", "Year", "Score","Votes","Runtime","Genres"]
data.head()


# __Marks = 2__

# Check the data types of each column

# In[3]:


data.dtypes


# __Marks = 1__

# ## 2. Clean the DataFrame
# 
# The data frame has several problems
# 
# 1. The runtime column is stored as a string
# 2. The genres column has several genres together. This way, it is hard to check which movies are Action movies and so on.
# 3. The movie year is also present in the title
# 
# 
# ### Fix the runtime column
# Convert the string '142 mins' to number 142.

# In[4]:


#data['Runtime']=pd.Series(data['Runtime']).str.replace("mins."," " , )
given_number = '142 mins.'
number, txt = given_number.split(' ')
convert_string = int(number)
print(number)


# __Marks = 3__

# Perform this conversion on every element in the dataframe `data`

# In[5]:


data['Runtime']  = [int(r.split(' ')[0]) for r in data.Runtime]
data.head()


# __Marks = 2__

# ### Split the genres

# We would like to split the genres column into many columns. Each new column will correspond to a single genre, and each cell will be True or False.
# 
# First, we would like to find the all the unique genres present in any record. Its better to sort the genres to locate easily.

# In[31]:


#determine the unique genres
Genres = set()
for m in data.Genres:
    Genres.update(g for g in str(m).split('|'))
Genres = sorted(Genres)


# __Marks = 4__

# Then make a column for each genre

# In[33]:


#make a column for each genre
for Genre in Genres:
    data[Genre] = [Genre in str(movie).split('|') for movie in data.Genres]
data.head()


# __Marks = 5__

# ### Eliminate year from the title
# We can fix each element by stripping off the last 7 characters

# In[8]:


data['Title'] = data['Title'].str.split('(', 1).str[0].str.strip()
data['Title']


# __Marks = 1__

# ## 3. Descriptive Statistics
# 
# Next, we would like to discover outliers. One possible way is to describe some basic, global summaries of the DataFrame on `score`, `runtime`, `year`, `votes`.

# In[18]:


#Call `describe` on relevant columns
data.describe(include=[np.number])


# Marks = 1

# Do you see any quantity unusual. Better replace with NAN.

# In[35]:


#Your code here
print(len(data.Runtime == 0))
data.Runtime[data.Runtime==0] = np.NAN


# __Marks = 1__

# Lets repeat describe to make sure that it is fine

# In[36]:


#Your code here
data.Runtime.describe()


# __Marks = 1__

# ### Basic plots

# Lets draw histograms for release year, IMDB rating, runtime distribution

# In[43]:


#Your code here
data.hist(column='Year' )


# __Marks = 1__

# In[44]:


#Your code here
data.hist(column='Score')


# __Marks = 1__

# In[45]:


#Your code here
data.hist(column='Runtime')


# __Marks = 1__

# Scatter plot between IMDB rating and years. Does it shows some trend?

# In[64]:


#Your code here
data.plot(kind='scatter', x='Score', y='Year',  c=['red','black'])


# Above grpah shows that, for movies rom 1950 to 1980 the score is mostly higher while after 1980 the movies got average score.  

# __Marks = 2__

# Is there any relationship between IMDB rating and number of votes? Describe

# In[65]:


#Your code here
data.plot(kind='scatter', x='Score', y='Votes',  c=['red','blue'])


# The movies whose score is high, got more scores.

# __Marks = 2__

# ### Data aggregation/Summarization

# *What genres are the most frequent?* Lay down the genres in descending order of count

# In[21]:


#Your code here
#sum sums over rows by default
from collections import Counter
words_to_count = (word for word in split_list if word[:1].isupper())
c = Counter(words_to_count)
a=sorted(c.items() , key=lambda pair:pair[1], reverse=True)
a


# __Marks = 2__

# Draw a bar plot to show top ten genres

# In[85]:


#Your code here
top_ten = a[:10]

labels, ys = zip(*top_ten)
xs = np.arange(len(labels)) 
width = 1

plt.bar(xs, ys, width,align='center')
plt.xticks(xs + width* 0.5, labels, rotation=30) 
plt.show()


# __Marks = 2__

# *How many genres does a movie have, on average?*

# In[35]:


#Your code here
#axis=1 sums over columns instead
genre_count = np.sort(data[Genres].sum())[::-1]
pd.DataFrame({'Genre Count': genre_count})


# __Marks = 2__

# ## Explore Group Properties

# Let's split up movies by decade. Find the decade mean score and draw a plot as follows:
# 
# <img src=score-year-plot.png>

# In[106]:


#Your code here

decade=data.groupby((data['Year']//10)*10).Score.mean()
print(decade)
data.groupby((data['Year']//10)*10).Score.mean().plot(x="Year", y="Score")


# __Marks = 5__

# Find the most popular movie each year
# 

# In[91]:


#Your code here
for year, subset in data.groupby('Year'):
    print(year, subset[subset.Score == subset.Score.max()].Title.values)


# __Marks = 2__
