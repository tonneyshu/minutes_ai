import time
import streamlit as st
import streamlit_scrollable_textbox as stx
import whisperx
import gc
import torch
import os

from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

llm_model = "llama-3.1-70b-versatile"

whisper_model = "large-v3"
language = "en"
device, compute_type = ("cuda", "float16") if torch.cuda.is_available() else ("cpu", "float32")
batch_size = 12
chunk_size = 6000

def free_gpu(model):
    gc.collect(); torch.cuda.empty_cache(); del model

def print_time_taken(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    st.write(f"Took: {m}m{s}s" if seconds >= 60 else f"Took: {int(seconds)}s")

@st.cache_resource
def load_whisper_model():
    model = whisperx.load_model(whisper_model, device, compute_type=compute_type, language=language)
    return model

def transcribe_audio(audio_file, model, num_speakers):
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name
    
    audio = whisperx.load_audio(temp_file_path)
    result = model.transcribe(audio, batch_size=batch_size); free_gpu(model)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False); free_gpu(model_a)

    model_d = whisperx.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)
    diarize_segments = model_d(audio, min_speakers=num_speakers, max_speakers=num_speakers); free_gpu(model_d)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    texts = [f"{e.get('speaker', 'SPEAKER_UNKNOWN')}: {e.get('text', '')}" for e in result.get("segments", [])]
    return "\n".join(texts)

def split_text(text, return_as_docs=True):
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    from langchain.docstore.document import Document
    return [Document(page_content=t) for t in chunks[:len(chunks)]] if return_as_docs else chunks

def load_llm(rate_limit=False):
    from langchain_core.rate_limiters import InMemoryRateLimiter
    rate_limiter = InMemoryRateLimiter(requests_per_second=0.01667, check_every_n_seconds=60, max_bucket_size=1) if rate_limit else None
    return ChatGroq(model=llm_model, temperature=0.0, rate_limiter=rate_limiter)

def fix_speaker_label(text):
    prompt_template = """
    You are tasked with reviewing the following conversation text to identify any instances of mis-labeled speakers. Please adhere to the following instructions:
    1. Identify Mis-labeled Instances: Carefully read through the text and pinpoint any occurrences where the speaker labels do not correspond accurately with the content of their speech, ensuring over '95%' confidentiality.
    2. Correct and Annotate: For each instance you identify as mis-labeled, wrap the corrected label in "**", followed by the original text.
    3. Output Format: Present the corrected conversation text with annotations clearly marked, but do not provide any additional explanations or justifications for the changes made.

    Here is the conversation text:
    {text}"""

    use_rate_limit = len(text) >= chunk_size
    llm = load_llm(rate_limit=use_rate_limit)
    if use_rate_limit:
        chunks = split_text(text, False)
        final_results = " ".join(llm.invoke(PromptTemplate.from_template(prompt_template).format(text=chunk)).content for chunk in chunks)
    else:
        final_results = llm.invoke(PromptTemplate.from_template(prompt_template).format(text=text)).content
    return final_results
        
def summarize_text(text):
    prompt_template = """
    You are an assistant that summarizes meeting minutes and identifies action items.

    Please summarize the following conversation text, focusing on:
    1. Follow-Up on Last Action Items: Review any action items from previous meetings, noting their status and whether they were completed or still pending.
    2. Main Topics Discussed: Highlight the key points and subjects covered during the meeting.
    3. Decisions Made: Note any decisions or conclusions reached by the participants.
    4. Action Items: Identify any follow-up tasks or responsibilities assigned, including who is responsible for each task and any deadlines mentioned.
    5. Participants Involved: List the speaker labels, names of the participants and their contributions to the discussion.

    Here is the conversation text:
    {text}"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = """
    Your job is to produce a final summary of the meeting minutes.

    We have provided an existing summary up to a certain point:
    {existing_answer}

    We have the opportunity to refine the existing summary (only if needed) with some more context below:
    {text}

    Given the new context, refine the original summary by addressing:
    1. Follow-Up on Last Action Items
    2. Main Topics Discussed
    3. Decisions Made
    4. Action Items
    5. Participants Involved
    If the context isn't useful, return the original summary."""
    refine_prompt = PromptTemplate.from_template(refine_template)

    chain = load_summarize_chain(
        llm=load_llm(rate_limit=len(text)>=chunk_size),
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        verbose=True,
        input_key="input_documents",
        output_key="output_text",
    )
    return chain({"input_documents": split_text(text)}, return_only_outputs=True)["output_text"]

st.title("AI Minutes")
whisper_model = load_whisper_model()
num_speakers = st.number_input("Number of Speakers", min_value=1, max_value=100, value=1)

uploaded_file = st.file_uploader("Upload an audio file (mp3, wav)", type=["mp3", "wav"])
if uploaded_file is not None:
    start_time = time.time()
    with st.spinner("Transcribing..."):
        st.subheader("Transcription:")
        transcription = transcribe_audio(uploaded_file, whisper_model, num_speakers)
        use_rate_limit = len(transcription) >= chunk_size
        fixed_transcription = fix_speaker_label(transcription) if num_speakers >= 2 else transcription
        stx.scrollableTextbox(fixed_transcription, height=150)
        end_time = time.time()
        print_time_taken(time.time() - start_time)
        
    start_time = time.time()
    with st.spinner("Summarizing..."):
        summary = summarize_text(fixed_transcription)
        st.subheader("Summary:")
        stx.scrollableTextbox(summary, height=150)
        print_time_taken(time.time() - start_time)