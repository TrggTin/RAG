import os
import logging
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EnhancedRAGSystem:
    def __init__(self, 
                 model_name: str = "gemini-1.5-flash",
                 embedding_model: str = "models/embedding-001",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 vectorstore_path: str = "vectorstore"):
        """
        Initialize the Enhanced RAG System with Google Gemini
        
        Args:
            model_name: Gemini model to use for chat
            embedding_model: Gemini embedding model
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            vectorstore_path: Path to store vector database
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore_path = vectorstore_path
        
        # Validate API key
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Initialize components
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1,
                convert_system_message_to_human=True
            )
            logger.info(f"Initialized RAG system with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini models: {e}")
            raise
        
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        
        # Create vectorstore directory if it doesn't exist
        Path(self.vectorstore_path).mkdir(parents=True, exist_ok=True)
    
    def load_documents_from_directory(self, directory_path: str, 
                                    file_types: List[str] = [".pdf", ".txt", ".md"]) -> List[Document]:
        """Load all documents from a directory"""
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.warning(f"Directory {directory_path} does not exist")
            return documents
        
        for file_type in file_types:
            pattern = f"*{file_type}"
            loader = DirectoryLoader(
                str(directory),
                glob=pattern,
                loader_cls=PyPDFLoader if file_type == ".pdf" else TextLoader,
                show_progress=True
            )
            try:
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} {file_type} files")
            except Exception as e:
                logger.error(f"Error loading {file_type} files: {e}")
        
        return documents
    
    def load_and_process_documents(self, file_paths: Optional[List[str]] = None, 
                                 directory_path: Optional[str] = None):
        """Load and process documents from file paths or directory"""
        documents = []
        
        # Load from specific file paths
        if file_paths:
            for file_path in file_paths:
                try:
                    path = Path(file_path)
                    if not path.exists():
                        logger.warning(f"File not found: {file_path}")
                        continue
                    
                    if path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(path))
                    elif path.suffix.lower() in ['.txt', '.md']:
                        loader = TextLoader(str(path))
                    else:
                        logger.warning(f"Unsupported file format: {file_path}")
                        continue
                    
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Loaded document: {file_path}")
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        # Load from directory
        if directory_path:
            dir_docs = self.load_documents_from_directory(directory_path)
            documents.extend(dir_docs)
        
        if not documents:
            logger.warning("No documents were loaded")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        try:
            splits = text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(splits)} chunks")
            
            # Create and save vector store
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            
            # Save vectorstore
            vectorstore_file = Path(self.vectorstore_path) / "faiss_index"
            self.vectorstore.save_local(str(vectorstore_file))
            
            # Setup retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            logger.info(f"Created and saved vector store with {len(splits)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
    
    def load_existing_vectorstore(self) -> bool:
        """Load an existing vector store from disk"""
        vectorstore_file = Path(self.vectorstore_path) / "faiss_index"
        
        if not vectorstore_file.exists():
            logger.info("No existing vectorstore found")
            return False
        
        try:
            self.vectorstore = FAISS.load_local(
                str(vectorstore_file),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            logger.info("Successfully loaded existing vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error loading existing vectorstore: {e}")
            return False
    
    def format_docs(self, docs) -> str:
        """Helper function to format retrieved documents"""
        if not docs:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = getattr(doc, 'metadata', {}).get('source', 'Unknown')
            content = doc.page_content.strip()
            formatted.append(f"Document {i} (Source: {source}):\n{content}")
        
        return "\n\n---\n\n".join(formatted)
    
    def setup_rag_chain(self):
        """Set up the RAG chain with improved prompt template"""
        template = """You are a helpful AI assistant. Answer the question based on the provided context. 
If the context doesn't contain enough information to answer the question completely, say so and provide what information you can.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context
- If information is incomplete, acknowledge this
- Use specific details from the context when possible
- Be concise but comprehensive

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain
        self.rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("RAG chain setup complete")
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        if not self.retriever:
            raise ValueError("No documents loaded. Please load documents first.")
        
        if not self.rag_chain:
            self.setup_rag_chain()
        
        try:
            logger.info(f"Processing query: {question[:50]}...")
            answer = self.rag_chain.invoke(question)
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get similar documents for a query"""
        if not self.retriever:
            raise ValueError("No documents loaded. Please load documents first.")
        
        return self.retriever.get_relevant_documents(query)
    
    def add_documents(self, new_documents: List[str]):
        """Add new documents to existing vectorstore"""
        if not self.vectorstore:
            logger.warning("No existing vectorstore. Use load_and_process_documents instead.")
            return
        
        # Load new documents
        documents = []
        for file_path in new_documents:
            try:
                path = Path(file_path)
                if path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(path))
                elif path.suffix.lower() in ['.txt', '.md']:
                    loader = TextLoader(str(path))
                else:
                    continue
                
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not documents:
            return
        
        # Split and add to vectorstore
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        
        # Add to existing vectorstore
        self.vectorstore.add_documents(splits)
        
        # Save updated vectorstore
        vectorstore_file = Path(self.vectorstore_path) / "faiss_index"
        self.vectorstore.save_local(str(vectorstore_file))
        
        logger.info(f"Added {len(splits)} new document chunks")

def main():
    """Main function with interactive CLI"""
    try:
        # Initialize RAG system
        rag = EnhancedRAGSystem()
        
        # Check if vectorstore exists
        if not rag.load_existing_vectorstore():
            print("No existing vectorstore found.")
            print("Please provide documents to process:")
            
            choice = input("Load from (1) specific files or (2) directory? Enter 1 or 2: ")
            
            if choice == "1":
                files = input("Enter file paths (comma-separated): ").strip().split(",")
                files = [f.strip() for f in files if f.strip()]
                rag.load_and_process_documents(file_paths=files)
            elif choice == "2":
                directory = input("Enter directory path: ").strip()
                rag.load_and_process_documents(directory_path=directory)
            else:
                print("Invalid choice. Exiting.")
                return
        
        # Interactive query loop
        print("\n" + "="*50)
        print("Enhanced RAG System with Google Gemini")
        print("Type 'quit' to exit, 'help' for commands")
        print("="*50)
        
        while True:
            try:
                question = input("\n🤖 Ask a question: ").strip()
                
                if question.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif question.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- Ask any question about your documents")
                    print("- 'quit' - Exit the program")
                    print("- 'help' - Show this help message")
                    continue
                elif not question:
                    continue
                
                print("\n💭 Thinking...")
                answer = rag.query(question)
                print(f"\n📝 Answer:\n{answer}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Set GOOGLE_API_KEY in your .env file")
        print("2. Installed required packages: pip install langchain-google-genai langchain-community faiss-cpu python-dotenv")

if __name__ == "__main__":
    main()