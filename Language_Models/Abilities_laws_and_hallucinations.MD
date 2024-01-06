# Emergent abilities, Scaling laws, and hallucinations

- Another aspect of LLMs is their emergent abilities, which arise as a result of extensive pre-training on vast datasets. These capabilities are not explicitly programmed but emerge as the model discerns patterns within the data. LangChain models capitalize on these emergent abilities by working with various types of models, such as chat models and text embedding models. This allows LLMs to perform diverse tasks, from answering questions to generating text and offering recommendations.

- Lastly, scaling laws describe the relationship between model size, training data, and performance. Generally, as the model size and training data volume increase, so does the model's performance. However, this improvement is subject to diminishing returns and may not follow a linear pattern. It is essential to weigh the trade-off between model size, training data, performance, and resources spent on training when selecting and fine-tuning LLMs for specific tasks.

- While Large Language Models boast remarkable capabilities but are not without shortcomings, one notable limitation is the occurrence of hallucinations, in which these models produce text that appears plausible on the surface but is actually factually incorrect or unrelated to the given input. Additionally, LLMs may exhibit biases originating from their training data, resulting in outputs that can perpetuate stereotypes or generate undesired outcomes.

## Examples with Easy Prompts: Text Summarization, Text Translation, and Question Answering

- In the realm of natural language processing, Large Language Models have become a popular tool for tackling various text-based tasks. These models can be promoted in different ways to produce a range of results, depending on the desired outcome.

### Setting Up the Environment

- To begin, we will need to install the huggingface_hub library in addition to previously installed packages and dependencies. Also, keep in mind to create the Huggingface API Key by navigating to Access Tokens page under the account’s Settings. The key must be set as an environment variable with HUGGINGFACEHUB_API_TOKEN key.

```bash
    !pip install -q huggingface_hub
```

###  Creating a Q-A Prompt Template

```python

from langchain import PromptTemplate

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question = "What is the capital city of France?"

```

- Use the Hugging Face model google/flan-t5-large to answer the question. The HuggingfaceHub class will connect to Hugging Face’s inference API and load the specified model.

```python

from langchain import HuggingFaceHub, LLMChain

# initialize Hub LLM
hub_llm = HuggingFaceHub(
        repo_id='google/flan-t5-large',
    model_kwargs={'temperature':0}
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm
)

# ask the user question about the capital of France
print(llm_chain.run(question))

```
*We can also modify the prompt template to include multiple questions*

```python
    qa = [
    {'question': "What is the capital city of France?"},
    {'question': "What is the largest mammal on Earth?"},
    {'question': "Which gas is most abundant in Earth's atmosphere?"},
    {'question': "What color is a ripe banana?"}
]
res = llm_chain.generate(qa)
print( res )
```

- We can modify our prompt template to include multiple questions to implement a second approach. The language model will understand that we have multiple questions and answer them sequentially. This method performs best on more capable models.

```python

multi_template = """Answer the following questions one at a time.

Questions:
{questions}

Answers:
"""
long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=llm
)

qs_str = (
    "What is the capital city of France?\n" +
    "What is the largest mammal on Earth?\n" +
    "Which gas is most abundant in Earth's atmosphere?\n" +
		"What color is a ripe banana?\n"
)
llm_chain.run(qs_str)

```


```MD
# Response

1. The capital city of France is Paris.
2. The largest mammal on Earth is the blue whale.
3. The gas that is most abundant in Earth's atmosphere is nitrogen.
4. A ripe banana is yellow.

```

## Text Summarization

```python

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

```

- Define a Prompt Template

```python

summarization_template = "Summarize the following text to one sentence: {text}"
summarization_prompt = PromptTemplate(input_variables=["text"], template=summarization_template)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

```
- To use the summarization chain, simply call the predict method with the text to be summarized


```python

text = "LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications. The most basic building block of LangChain is calling an LLM on some input. Let’s walk through a simple example of how to do this. For this purpose, let’s pretend we are building a service that generates a company name based on what the company makes."
summarized_text = summarization_chain.predict(text=text)

```
## Text Translation

- It is one of the great attributes of Large Language models that enables them to perform multiple tasks just by changing the prompt. However, pass a different prompt that asks for translating the query from a source_language to the target_language.

```python

translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
translation_prompt = PromptTemplate(input_variables=["source_language", "target_language", "text"], template=translation_template)
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

# To use the translation chain, call the predict method with the source language, target language, and text to be translated

source_language = "English"
target_language = "French"
text = "Your text here"
translated_text = translation_chain.predict(source_language=source_language, target_language=target_language, text=text)

```


