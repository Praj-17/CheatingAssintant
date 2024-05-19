from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.document import Document
from typing import List
from dotenv import load_dotenv
import math
import os
load_dotenv()


class MyVectorStoreRetriever(VectorStoreRetriever):
    # See https://github.com/langchain-ai/langchain/blob/61dd92f8215daef3d9cf1734b0d1f8c70c1571c3/libs/langchain/langchain/vectorstores/base.py#L500
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
        )

        # Make the score part of the document metadata
        for doc, similarity in docs_and_similarities:
            doc.metadata["score"] = similarity

        docs = [doc for doc, _ in docs_and_similarities]
        return docs




class Chat:
    def __init__(self):
        '''
        return ConversationalRetrievalChain.from_llm(
            model,
            #retriever=vector_store.as_retriever(),
            retriever = MyVectorStoreRetriever(
                vectorstore=vector_store,
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.2, "k": 3},
            ),
            return_source_documents=True,
            # verbose=True,
        )
    '''
        self.embedding = OpenAIEmbeddings()

        self.vector_store = Chroma(
            collection_name=os.getenv("default_collection_name"),
            embedding_function=self.embedding,
            persist_directory=os.getenv("default_data_directory"),
        )
        self.chain = RetrievalQA.from_chain_type(
            ChatOpenAI(
            model_name=os.getenv("model_name"),
            temperature=os.getenv("temperature"),
            # verbose=True
        ),
            chain_type=os.getenv("chain_type"),
            retriever = MyVectorStoreRetriever(
                vectorstore=self.vector_store,
                search_type=os.getenv("search_type"),
                search_kwargs={"score_threshold": float(os.getenv("score_threshold")), "k": int(os.getenv("top_k_to_search"))},
            ),
            return_source_documents=True,
        )
    
    def chat(self, question, history = ()):
        chat_history = []
        answer = None
        if history:
            for messages in history:
                if len(messages) ==2:
                    question,answer  = messages
                    chat_history.append(HumanMessage(content=question))
                    chat_history.append(AIMessage(content=answer))
                else:
                    continue
        response = self.chain({"query": question, "chat_history": history})
        answer = response["result"]
        source = response["source_documents"]

        pgs = []
        for document in source:
            pgs.append(document.metadata['page_number'])
            #print(f"List after inserting:", pgs)
            
        for i in range(0, len(pgs)):
            for j in range(i+1, len(pgs)):
                #if(l[i] == l[j]):
                if(math.isclose(pgs[i], pgs[j], abs_tol = 2)):
                        pgs.insert(0, pgs[i])
        pgs = list(set(pgs))
        return answer, pgs
    
        



    
if __name__ == "__main__":
    load_dotenv()
    chat = Chat()

    while True:
        question = input("Question: ")
        answer, pgs = chat.chat(question = question )
        print(answer, pgs)


       