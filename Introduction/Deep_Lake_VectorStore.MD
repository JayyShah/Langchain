- Deep Lake provides storage for embeddings and their corresponding metadata in the context of LLM apps. It enables hybrid searches on these embeddings and their attributes for efficient data retrieval. It also integrates with LangChain, facilitating the development and deployment of applications.

- Deep Lake provides several advantages over the typical vector store:

    - It’s multimodal, which means that it can be used to store items of diverse modalities, such as texts, images audio, and video, along with their vector representations. 
    - It’s serverless, which means that we can create and manage cloud datasets without creating and managing a   database instance. This aspect gives a great speedup to new projects.
    - Last, it’s possible to easily create a data loader out of the data loaded into a Deep Lake dataset. It is   convenient for fine-tuning machine learning models using common frameworks like PyTorch and TensorFlow.

## Install DeepLake

`pip install deeplake`

--

```python

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Before executing the following code, make sure to have your
# Activeloop key saved in the “ACTIVELOOP_TOKEN” environment variable.

# instantiate the LLM and embeddings models
llm = OpenAI(model="text-davinci-003", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = "<YOUR-ACTIVELOOP-ORG-ID>" 
my_activeloop_dataset_name = "<ENTER_DATASET_NAME>"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# add documents to our Deep Lake dataset
db.add_documents(docs)

```

- If everything works correctly, you should see a printed output like this: Your Deep Lake dataset has been successfully created! The dataset is private so make sure you are logged in!

## Creating a RetrievalQA chain

```python

retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm,
	chain_type="stuff",
	retriever=db.as_retriever()
)

```

## Create an Agent that uses the RetrievalQA chain as a Tool

```python

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

```

## Using the Agent

```python

response = agent.run("When was Napoleone born?")
print(response)

```

```MD

# Response

**> Entering new AgentExecutor chain...**

***I need to find out when Napoleone was born.
Action: Retrieval QA System
Action Input: When was Napoleone born?***

Observation:

***Napoleon Bonaparte was born on 15 August 1769.***

Thought:

***I now know the final answer.
Final Answer: Napoleon Bonaparte was born on 15 August 1769.***

**> Finished chain.**

Napoleon Bonaparte was born on 15 August 1769.

```

- Reload an existing vector store from Deep Lake that's located at a specified dataset path. Then, load new textual data and split it into manageable chunks. Finally, add these chunks to the existing dataset, creating and storing corresponding embeddings for each added text segment:

```python

# load the existing Deep Lake dataset and specify the embedding function
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

# create new documents
texts = [
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# add documents to our Deep Lake dataset
db.add_documents(docs)

```

## Recreate the Previous Agent to ask questions

```python

# instantiate the wrapper class for GPT3
llm = OpenAI(model="text-davinci-003", temperature=0)

# create a retriever from the db
retrieval_qa = RetrievalQA.from_chain_type(
	llm=llm, chain_type="stuff", retriever=db.as_retriever()
)

# instantiate a tool that uses the retriever
tools = [
    Tool(
        name="Retrieval QA System",
        func=retrieval_qa.run,
        description="Useful for answering questions."
    ),
]

# create an agent that uses the tool
agent = initialize_agent(
	tools,
	llm,
	agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
	verbose=True
)

```

- The LLM successfully retrieves accurate information by using the power of Deep Lake as a vector store and the OpenAI language model.