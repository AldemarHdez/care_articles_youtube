import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import googleapiclient.discovery
import os
from typing import List, Dict
from dotenv import load_dotenv
import dspy


load_dotenv()

def load_articles(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    articles = []
    for entry in data:
        article = {
            'id': entry['id'],
            'url': entry['url'],
            'title': entry['title'],
            'content': entry['content'],
            'categories': entry['prcValue'].split(';')
        }
        articles.append(article)
    return articles

articles = load_articles('20231211-coded-content.json')

model = SentenceTransformer('all-MiniLM-L6-v2')
def embed_articles(articles):
    contents = [article['content'] for article in articles]
    embeddings = model.encode(contents, convert_to_numpy=True)
    return embeddings

article_embeddings = embed_articles(articles)
index = faiss.IndexFlatL2(article_embeddings.shape[1])
index.add(np.array(article_embeddings))

def search_articles(query, index, articles, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [articles[i] for i in indices[0]]



def get_youtube_recommendations(query: str) -> List[Dict[str, str]]:
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = os.getenv('YOUTUBE_API_KEY')

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.search().list(
        part="snippet",
        maxResults=5,
        q=query
    )
    response = request.execute()
    
    videos = [{"title": item['snippet']['title'], "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"} for item in response['items']]
    return videos

class ArticleRetriever(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        retrieved_articles = search_articles(question, index, articles)  # Use global 'articles' variable
        response = "Based on your input, here are some relevant articles:\n"
        for article in retrieved_articles:
            response += f"- [{article['title']}]({article['url']})\n"
        
        response += "\nHere are some YouTube videos you might find helpful:\n"
        youtube_recommendations = get_youtube_recommendations(question)
        for video in youtube_recommendations:
            response += f"- [{video['title']}]({video['url']})\n"

        return response

# Set up the DSPy environment and compile the program
dspy.settings.configure(lm=dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250))

# Initialize the ArticleRetriever
article_retriever = ArticleRetriever()


def generate_response(query):
    return article_retriever.forward(query)

# Example query
query = "my mom got fall down in the kitchen, and she suffers dementia"
print(generate_response(query))
