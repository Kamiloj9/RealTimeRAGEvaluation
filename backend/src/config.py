from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from src.utils import load_config

# Load API key
API_KEY = load_config()

# Model configurations
BASE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize LangChain interfaces
llm = ChatOpenAI(api_key=API_KEY, model=BASE_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(api_key=API_KEY, model=EMBEDDING_MODEL)

# Default processing settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_RETRIEVER_K = 4