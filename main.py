import os
import json
import fitz
import re
import datetime
from dotenv import load_dotenv
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
import pymongo
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.docstore.document import Document
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
