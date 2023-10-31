import arxiv
import os
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever, PreProcessor, PDFToTextConverter, PromptNode, PromptTemplate, TopPSampler
from haystack.nodes.ranker import LostInTheMiddleRanker
from haystack.pipelines import Pipeline
import gradio as gr


document_store = InMemoryDocumentStore()
embedding_retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/All-MiniLM-L6-V2", model_format="sentence_transformers", top_k=20)
FILE_PATH = ""
DIR = "./arxiv"


class ArxivComponent:
    """
    This component is responsible for retrieving arXiv articles based on an arXiv ID.
    """

    def run(self, arxiv_id: str = None):
        """
        Retrieves and stores an arXiv article for the given arXiv ID.

        Args:
            arxiv_id (str): ArXiv ID of the article to be retrieved.
        """
        # Set the directory path where arXiv articles will be stored
        dir: str = DIR

        # Create an instance of the arXiv client
        arxiv_client = arxiv.Client()

        # Check if an arXiv ID is provided; if not, raise an error
        if arxiv_id is None:
            raise ValueError("Please provide the arXiv ID of the article to be retrieved.")

        # Search for the arXiv article using the provided arXiv ID
        search = arxiv.Search(id_list=[arxiv_id])
        response = arxiv_client.results(search)
        paper = next(response)  # Get the first result
        title = paper.title  # Extract the title of the article

        # Check if the specified directory exists
        if os.path.isdir(dir):
            # Check if the PDF file for the article already exists
            if os.path.isfile(dir + "/" + title + ".pdf"):
                return {"file_path": [dir + "/" + title + ".pdf"]}
        else:
            # If the directory does not exist, create it
            os.mkdir(dir)

        # Attempt to download the PDF for the arXiv article
        try:
            paper.download_pdf(dirpath=dir, filename=title + ".pdf")
            return {"file_path": [dir + "/" + title + ".pdf"]}
        except:
            # If there's an error during the download, raise a ConnectionError
            raise ConnectionError(message=f"Error occurred while downloading PDF for \
                                            arXiv article with ID: {arxiv_id}")   

def indexing_pipeline(file_path: str = None):
    pdf_converter = PDFToTextConverter()
    preprocessor = PreProcessor(split_by="word", split_length=250, split_overlap=30)

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(
        component=pdf_converter, 
        name="PDFConverter", 
        inputs=["File"]
        )
    indexing_pipeline.add_node(
        component=preprocessor, 
        name="PreProcessor", 
        inputs=["PDFConverter"]
        )
    indexing_pipeline.add_node(
        component=embedding_retriever,
        name="EmbeddingRetriever", 
        inputs=["PreProcessor"]
        )
    indexing_pipeline.add_node(
        component=document_store, 
        name="InMemoryDocumentStore", 
        inputs=["EmbeddingRetriever"]
        )

    indexing_pipeline.run(file_paths=file_path)


def query_pipeline(query: str = None):
    if not query:
        raise gr.Error("Please provide a query.")
    prompt_text = """
Synthesize a comprehensive answer from the provided paragraphs of an Arxiv article and the given question.\n
Focus on the question and avoid unnecessary information in your answer.\n
\n\n Paragraphs: {join(documents)} \n\n Question: {query} \n\n Answer:
"""
    prompt_node = PromptNode(
                         "gpt-3.5-turbo",
                          default_prompt_template=PromptTemplate(prompt_text),
                          api_key="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",
                          max_length=768,
                          model_kwargs={"stream": False},
                         )
    query_pipeline = Pipeline()
    query_pipeline.add_node(
        component = embedding_retriever, 
        name = "Retriever", 
        inputs=["Query"]
        )
    query_pipeline.add_node(
        component=TopPSampler(
        top_p=0.90), 
        name="Sampler", 
        inputs=["Retriever"]
        )
    query_pipeline.add_node(
        component=LostInTheMiddleRanker(1024), 
        name="LostInTheMiddleRanker", 
        inputs=["Sampler"]
        )
    query_pipeline.add_node(
        component=prompt_node, 
        name="Prompt", 
        inputs=["LostInTheMiddleRanker"]
        )

    pipeline_obj = query_pipeline.run(query = query)
    
    return pipeline_obj["results"]

arxiv_obj = ArxivComponent()
def embed_arxiv(arxiv_id: str):
    """
        Args:
            arxiv_id: Arxiv ID of the article to be retrieved.
            dir: Directory where the articles are stored.
            file_path: File path of existing PDF file.
        """
    global FILE_PATH
    dir: str = DIR
    file_path: str = None
    if not arxiv_id:
        raise gr.Error("Provide an Arxiv ID")
    file_path_dict = arxiv_obj.run(arxiv_id)
    file_path = file_path_dict["file_path"]
    FILE_PATH = file_path
    indexing_pipeline(file_path=file_path)

    return "Successfully embedded the file"

def get_response(history, query: str):
    if not query:
        gr.Error("Please provide a query.")
    
    response = query_pipeline(query=query)
    for text in response[0]:
        history[-1][1] += text
        yield history

def add_text(history, text: str):
    if not text:
         raise gr.Error('enter text')
    history = history + [(text,'')] 
    return history

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=60):
            text_box = gr.Textbox(placeholder="Input Arxiv ID", interactive=True).style(container=False)
        with gr.Column(scale=40):
            submit_id_btn = gr.Button(value="Submit")
    with gr.Row():
        chatbot = gr.Chatbot(value=[]).style(height=600)
    
    with gr.Row():
        with gr.Column(scale=70):
            query = gr.Textbox(placeholder = "Enter query string", interactive=True).style(container=False)
    
    submit_id_btn.click(
        fn = embed_arxiv, 
        inputs=[text_box],
        outputs=[text_box],
        )
    query.submit(
            fn=add_text, 
            inputs=[chatbot, query], 
            outputs=[chatbot, ], 
            queue=False
            ).success(
            fn=get_response,
            inputs = [chatbot, query],
            outputs = [chatbot,]
            )
demo.queue()
demo.launch()
