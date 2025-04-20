#!/usr/bin/env python
# coding: utf-8

# !pip install selenium pandas webdriver-manager
# 

# Scraping the Data From the SHL website using Selenium 

# In[8]:


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Open catalog
driver.get("https://www.shl.com/solutions/products/product-catalog/")
time.sleep(5)

# Grab all <tr> rows in the main catalog table
rows = driver.find_elements(By.CSS_SELECTOR, "table tr")[1:]  # Skipping header

catalog_links = []

for row in rows:
    try:
        cells = row.find_elements(By.TAG_NAME, "td")

        link_elem = row.find_element(By.TAG_NAME, "a")
        name = link_elem.text.strip()
        url = link_elem.get_attribute("href")

        # Check green dot by class ".catalogue__circle.-yes"
        try:
            cells[1].find_element(By.CSS_SELECTOR, ".catalogue__circle.-yes")
            remote_support = "Yes"
        except:
            remote_support = "No"

        try:
            cells[2].find_element(By.CSS_SELECTOR, ".catalogue__circle.-yes")
            adaptive_support = "Yes"
        except:
            adaptive_support = "No"

        # Get test type
        test_type = ", ".join([tt.text.strip() for tt in cells[3].find_elements(By.TAG_NAME, "span")])

        catalog_links.append({
            "name": name,
            "url": url,
            "remote_support": remote_support,
            "adaptive_support": adaptive_support,
            "test_type": test_type
        })

    except Exception as e:
        print("⚠️ Error processing row:", e)
        continue



print(f"✅ Found {len(catalog_links)} valid assessment entries.")

# Save to CSV
df = pd.DataFrame(catalog_links)
df.to_csv("shl_assessments.csv", index=False)
print("✅ Saved to shl_assessments.csv")

driver.quit()


# Now we will Do Embedding

# In[7]:


import subprocess
subprocess.run(["pip", "install", "google-generativeai"])


# In[8]:


import google.generativeai as genai
genai.configure(api_key="your api key")
#Api. key


# In[10]:


def get_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return response['embedding']



# In[17]:


import pandas as pd

df = pd.read_csv("shl_assessments.csv")  # or your actual CSV name
df.head()


# In[18]:


df["full_text"] = (
    df["name"].fillna("") +
    ". Remote Support: " + df["remote_support"].fillna("") +
    ". Adaptive Support: " + df["adaptive_support"].fillna("") +
    ". Type: " + df["test_type"].fillna("")
)


# In[21]:


from tqdm import tqdm
tqdm.pandas()

df["embedding"] = df["full_text"].progress_apply(get_embedding)


# In[22]:


import pickle

with open("shl_assessments_with_embeddings.pkl", "wb") as f:
    pickle.dump(df, f)



# In[28]:


import pandas as pd

# Specify the path to your pickle file
pickle_file_path = "shl_assessments_with_embeddings.pkl"

# Load the pickle file into a DataFrame
df = pd.read_pickle(pickle_file_path)

# Display the DataFrame
print(df.head())


# We Start BUIlding Recommendation system

# In[40]:


import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the DataFrame from .pkl file
with open("shl_assessments_with_embeddings.pkl", "rb") as f:
    df = pickle.load(f)

# Convert the 'embedding' column into a matrix
embeddings = np.vstack(df["embedding"].values)  # shape: (n, 768)

# Gemini embedding function (yours should already be working)
def get_embedding(text: str):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return response['embedding']

# Final recommendation function
def recommend_assessments(query: str, top_k: int = 10):
    query_vector = np.array(get_embedding(query)).reshape(1, -1)
    similarities = cosine_similarity(query_vector, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "assessment_name": row["name"],
            "url": row["url"],
            "remote_testing": row["remote_support"],      # corrected key
            "adaptive_support": row["adaptive_support"],
            "duration": "Not provided",                    # optional: add if you later extract it
            "test_type": row["test_type"],
            "similarity_score": float(similarities[idx])   # optional: useful for debugging
        })

    return results

# Example usage
if __name__ == "__main__":
    query = "Looking for a sales manager with good communication and leadership"
    recommendations = recommend_assessments(query)
    for rec in recommendations:
        print(rec)

