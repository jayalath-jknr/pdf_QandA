# import streamlit as st
# from clarifai.client.model import Model
# import base64
# from dotenv import load_dotenv
# from PIL import Image
# from io import BytesIO

# load_dotenv()
# import os

# clarifai_pat = os.getenv("CLARIFAI_PAT")
# openai_api_key = os.getenv("OPEN_AI")

# def generate_image(user_description, api_key):
#     prompt = f"You are a professional comic artist. Based on the below user's description and content, create a proper story comic: {user_description}"
#     inference_params = dict(quality="standard", size="1024x1024")
#     model_prediction = Model(
#         f"https://clarifai.com/openai/dall-e/models/dall-e-3?api_key={api_key}"
#     ).predict_by_bytes(
#         prompt.encode(), input_type="text", inference_params=inference_params
#     )
#     output_base64 = model_prediction.outputs[0].data.image.base64
#     with open("generated_image.png", "wb") as f:
#         f.write(output_base64)
#     return "generated_image.png"

# def understand_image(base64_image, api_key):
#     prompt = "Analyze the content of this image and write a creative, engaging story that brings the scene to life. Describe the characters, setting, and actions in a way that would captivate a young audience:"
#     inference_params = dict(temperature=0.2, image_base64=base64_image, api_key=api_key)
#     model_prediction = Model(
#         "https://clarifai.com/openai/chat-completion/models/gpt-4-vision"
#     ).predict_by_bytes(
#         prompt.encode(), input_type="text", inference_params=inference_params
#     )
#     return model_prediction.outputs[0].data.text.raw

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

# def text_to_speech(input_text, api_key):
#     inference_params = dict(voice="alloy", speed=1.0, api_key=api_key)
#     model_prediction = Model(
#         "https://clarifai.com/openai/tts/models/openai-tts-1"
#     ).predict_by_bytes(
#         input_text.encode(), input_type="text", inference_params=inference_params
#     )
#     audio_base64 = model_prediction.outputs[0].data.audio.base64
#     return audio_base64

# def main():
#     st.set_page_config(page_title="Interactive Media Creator", layout="wide")
#     st.title("Interactive Media Creator")

#     with st.sidebar:
#         st.header("Controls")
#         image_description = st.text_area("Description for Image Generation", height=100)
#         generate_image_btn = st.button("Generate Image")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.header("Comic Art")
#         if generate_image_btn and image_description:
#             with st.spinner("Generating image..."):
#                 image_path = generate_image(image_description, clarifai_pat)
#                 if image_path:
#                     st.image(
#                         image_path,
#                         caption="Generated Comic Image",
#                         use_column_width=True,
#                     )
#                     st.success("Image generated!")
#                 else:
#                     st.error("Failed to generate image.")

#     with col2:
#         st.header("Story")
#         if generate_image_btn and image_description:
#             with st.spinner("Creating a story..."):
#                 base64_image = encode_image(image_path)
#                 understood_text = understand_image(base64_image, openai_api_key)
#                 audio_base64 = text_to_speech(understood_text, openai_api_key)
#                 st.audio(audio_base64, format="audio/mp3")
#                 st.success("Audio generated from image understanding!")

# if __name__ == "__main__":
#     main()

# from dotenv import load_dotenv
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.callbacks import get_openai_callback


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Ask your PDF", page_icon="🧊")
#     st.header("Ask your PDF 💬")

#     # upload file
#     pdf = st.file_uploader("Upload your PDF", type="pdf")

#     # extract the text
#     if pdf is not None:
#         pdf_reader = PdfReader(pdf)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()

#         # split into chunks
#         text_splitter = CharacterTextSplitter(
#             separator="\n",
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )
#         chunks = text_splitter.split_text(text)

#         # create embeddings
#         embeddings = OpenAIEmbeddings()
#         knowledge_base = FAISS.from_texts(chunks, embeddings)

#         # show user input
#         user_question = st.text_input("Ask a question about your PDF:")
#         if user_question:
#             docs = knowledge_base.similarity_search(user_question)

#             llm = OpenAI()
#             chain = load_qa_chain(llm, chain_type="stuff")
#             with get_openai_callback() as cb:
#                 response = chain.run(input_documents=docs, question=user_question)
#                 print(cb)

#             st.write(response)


# if __name__ == '__main__':
#     main()

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.callbacks import get_openai_callback



from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

PAT = os.getenv("CLARIFAI_PAT")
USER_ID = 'openai'
APP_ID = 'chat-completion'
MODEL_ID = 'gpt-4-vision-alternative'
MODEL_VERSION_ID = '12b67ac2b5894fb9af9c06ebf8dc02fb'
RAW_TEXT = 'I love your product very much'

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF", page_icon="🧊")
    st.header("Ask your PDF 💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()

            # Clarifai code integration
            channel = ClarifaiChannel.get_grpc_channel()
            stub = service_pb2_grpc.V2Stub(channel)

            metadata = (('authorization', 'Key ' + PAT),)
            userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

            post_model_outputs_response = stub.PostModelOutputs(
                service_pb2.PostModelOutputsRequest(
                    user_app_id=userDataObject,
                    model_id=MODEL_ID,
                    version_id=MODEL_VERSION_ID,
                    inputs=[
                        resources_pb2.Input(
                            data=resources_pb2.Data(
                                text=resources_pb2.Text(
                                    raw=RAW_TEXT
                                )
                            )
                        )
                    ]
                ),
                metadata=metadata
            )
            if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
                print(post_model_outputs_response.status)
                raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")

            output = post_model_outputs_response.outputs[0]

            st.write("Clarifai Completion:\n")
            st.write(output.data.text.raw)

            # Continue with the original code
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write("LangChain Completion:\n")
            st.write(response)


if __name__ == '__main__':
    main()
