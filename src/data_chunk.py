from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain import hub
import os
from dataloader import load_high
from agentic_chunker import AgenticChunker
from typing import List

# Định nghĩa schema đầu ra
class Sentences(BaseModel):
    sentences: List[str]


def run_chunk(essay):
    # Lấy prompt template từ LangChain Hub
    obj = hub.pull("wfh/proposal-indexing")

    # Khởi tạo LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # hoặc "gpt-4-1106-preview"
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Parser cho đầu ra JSON theo schema
    parser = PydanticOutputParser(pydantic_object=Sentences)

    # Tạo prompt mẫu
    prompt = ChatPromptTemplate.from_template(
        """
        Extract main propositions from the following text. 
        Return strictly in JSON format matching this schema:
        {format_instructions}

        Text:
        {text}
        """
    ).partial(format_instructions=parser.get_format_instructions())

    # Gộp pipeline
    chain = prompt | llm | parser

    # Xử lý từng đoạn
    paragraphs = essay.split("\n\n")
    essay_propositions = []

    for i, para in enumerate(paragraphs):
        try:
            result = chain.invoke({"text": para})
            essay_propositions.extend(result.sentences)
            print(f"✅ Done paragraph {i+1}")
        except Exception as e:
            print(f"⚠️ Skipped paragraph {i+1}: {e}")

    # Chunk lại
    ac = AgenticChunker()
    ac.add_propositions(essay_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type="list_of_strings")

    return chunks
