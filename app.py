import os  
from flask import Flask, render_template, redirect, url_for, request
from langchain_pinecone import PineconeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from flask import Flask, render_template, redirect, url_for, request, session, flash


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Pinecone and Groq API
pinecone_api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY") 
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key") # Replace with your actual key

pc = Pinecone(api_key=pinecone_api_key)
llm_groq = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-3b-preview")

# Pinecone index name
PINECONE_INDEX_NAME = "car-data-index"


# Ensure Pinecone index exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{PINECONE_INDEX_NAME}' does not exist. Create the index first.")
    exit(1)

# Initialize PineconeEmbeddings for retrieval
embeddings = PineconeEmbeddings(model="multilingual-e5-large")

# Initialize Pinecone vector store with embedding function
vector_store = PineconeVectorStore(
    index=pc.Index(PINECONE_INDEX_NAME),
    embedding=embeddings  # Provide the embedding function for retrieval
)

# Configure LangChain with Groq and Pinecone
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm_groq, retriever=vector_store.as_retriever(), memory=memory
)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")



users = {
    "admin": "123456",
    "user1": "password456"
}

# Your chatbot route with login validation
@app.route("/chat", methods=["GET", "POST"])
def chat():
    print(f"Session data: {session}") 
    if "user" not in session:  # Check if user is logged in
        return redirect(url_for("login"))  # Redirect to login page if not logged in

    answer = ""
    source_documents = []

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            response = chain.invoke({"question": user_input})
            answer = response.get("answer", "No answer available.")
            source_documents = response.get("source_documents", [])

    return render_template("chat.html", answer=answer, source_documents=source_documents)

# Login route to handle POST request
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Validate user credentials
        if username in users and users[username] == password:
            session["user"] = username  # Set session variable
            flash("You are now logged in!", "success")
            return redirect(url_for("index"))  # Redirect to homepage

        flash("Invalid username or password", "error")
    
    return render_template("login.html")


# Logout route to clear session
@app.route("/logout")
def logout():
    session.pop("user", None)  # Clear the session
    flash("You have been logged out!", "success")
    return redirect(url_for("index"))  # Redirect to homepage
 # Replace "index" with the name of your index page route


if __name__ == "__main__":
    app.run(debug=True)
