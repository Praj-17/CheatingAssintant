import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import sqlite3
#FaissSearch.
import os

from dotenv import load_dotenv
load_dotenv()
class EmbeddingGenerator():
    def __init__(self,model_name=os.getenv("sentence_transformer_model")):
        self.model = SentenceTransformer(model_name)
        self.openai_embeding = OpenAIEmbeddings()
        
    def embed_query(self, query, open_ai = False): 
        return self.model.encode(query)
    def openai_embed(self, query):
        return self.openai_embed.generate_embedding(query)
    
    def new_embeddings(self,sentence):
        embedding=np.array(self.model.encode([sentence]))
        embedding=embedding.astype(np.float32)
        return embedding

        

    def generate(self,text_data_list, is_openai = True):
        embedding_lists = np.empty((0, 768)).astype(np.float32)
        self.model = SentenceTransformer(os.getenv("sentence_transformer_model"))
        self.openai_embeding = OpenAIEmbeddings()
        for _, text in enumerate(text_data_list):
            if is_openai:
                embedding = self.openai_embeding.embed_query(text)
            else:
                embedding = self.model.encode(text).reshape(1, -1)
            embedding_lists = np.vstack((embedding_lists, embedding))
        return embedding_lists
    
    def generate_allmini(self,text_data_list):
        embedding_lists=np.empty(shape=(0,384),dtype=np.float32)
        for text in text_data_list:
            embedding=self.new_embeddings(text)
            embedding_lists=np.concatenate([embedding_lists,embedding])
        return embedding_lists


    
    def generate_openai_embeddings(self, text_data_list):
        embedding_lists=np.empty(shape=(0,3072),dtype=np.float32)
        
        for i, text in enumerate(text_data_list):
            embedding = self.model.encode(text)
            # embedding = self.embed_query(text, open_ai=True)
            embedding_lists=np.concatenate([embedding_lists,[embedding]])
        return embedding_lists
            

    def regenerate(self,db_file):
        conn=sqlite3.connect(db_file)
        cursor=conn.cursor()
        try:
            cursor.execute("""SELECT text FROM chat_images""")
            data=cursor.fetchall()['text']
        except:
            print("failed")    
        pass
class FaissSearch(EmbeddingGenerator):
    def __init__(self):
        super().__init__()
    def get_saved_index_path(self, pdf_path):
        file_name = os.path.splitext(os.path.basename(pdf_path))[0] + ".index"
        folder_path = os.path.dirname(pdf_path)
        index_file = os.path.join(folder_path, file_name)
        return index_file

    def create_index(self,embeddings_path):
        vectors = np.load(embeddings_path)
        vector_dimension = vectors.shape[1]
        index = FAISS.IndexFlatL2(vector_dimension)
        FAISS.normalize_L2(vectors)
        index.add(vectors)
        return index

    def save_faiss_index(self,embeddings_path):
        index_file_path = self.get_saved_index_path(embeddings_path)
        index = self.create_index(embeddings_path)
        FAISS.write_index(index, index_file_path)
        print("Index Created Sucessfully")
        return index_file_path
    def get_entities_from_ids(self, results):
        try:
            self.cursor.execute(f"SELECT page_title, image_url FROM chat_images WHERE id IN ({', '.join(map(str,list( results.ann)))}) and is_active = 1 order by priority desc")
            data = self.cursor.fetchall()
        except Exception as e:
            print("Exception", str(e))
            data = pd.DataFrame()

        # Convert the fetched data into a DataFrame
        if len(data)>0:
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(data, columns=columns)
        else:
            df = pd.DataFrame()
        return df

    def load_faiss_index(self,index_file_path):
    # Load the index
        return FAISS.read_index(index_file_path)
    def get_the_index_number(self,results):
        return int(results[0][0] + 1)
    def search(self, index, query, k,is_openai=False):
        index = self.load_FAISS_index(index)
        search_vector = self.embed_query(query,is_openai)
        _vector = np.array([search_vector])
        FAISS.normalize_L2(_vector)
        distances, ann = index.search(_vector, k=k)
        return self.get_the_index_number(ann)
    
            


if __name__ == "__main__":
    # f = FaissSearch(r"Chat\FaissSearch\data\embedding_lists.npy", r"Chat\FaissSearch\data\chat_images.db")
    # query = "What is Moon?"
    # data = f.search(query=query, k=10)
    # print(data)
    # a=EmbeddingGenerator()
    # a=a.generate(["Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum. ",
    #               "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.",
    #               "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."])
    # print(a.shape)

    # data=pd.read_csv(r'C:\Yozu\Updated Backend\Chat\FaissSearch\data\20k_images.csv')
    # print(len(data['text'].to_list()))
    # data=data['text'].to_list()
    # old_e=EmbeddingGenerator(model_name="paraphrase-albert-small-v2")
    # old_embeddings =  old_e.generate(data)
    # print(old_embeddings.shape)
    # np.save('old_embeddings.npy',old_embeddings)

    # baai_e=EmbeddingGenerator(model_name='BAAI/bge-base-en-v1.5')
    # baai_embeddings =  baai_e.generate(data)
    # print(baai_embeddings.shape)
    # np.save('baai_embeddings.npy',baai_embeddings)

    # baai_e=EmbeddingGenerator(model_name='BAAI/bge-base-en-v1.5')
    # baai_embeddings =  baai_e.generate(data)
    # print(baai_embeddings.shape)
    # np.save('baai_embeddings.npy',baai_embeddings)

    
    
    e = EmbeddingGenerator()
    e1 = e.generate(["My name is prajwal", "She is Sakshi"], is_openai=True)
    print(e1)
    print(e1.shape)
    

    
