# Tokens Distributions and Predicting Next Tokens

- Large language models like GPT-3 and GPT-4 are pretrained on vast amounts of text data and learn to predict the next token in a sequence based on the context provided by the previous tokens. GPT-family models use Causal Language modeling, which predicts the next token while only having access to the tokens before it. This process enables LLMs to generate contextually relevant text.

```python

from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", temperature=0)

text = "What would be a good company name for a company that makes colorful socks?"

print(llm(text))

```

## Tracking Token usage

```python

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2)

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)

```

## Few-Shot Learning

- Few-shot learning is a remarkable ability that allows LLMs to learn and generalize from limited examples. Prompts serve as the input to these models and play a crucial role in achieving this feature. With LangChain, examples can be hard-coded, but dynamically selecting them often proves more powerful, enabling LLMs to adapt and tackle tasks with minimal training data swiftly.

- This approach involves using the FewShotPromptTemplate class, which takes in a PromptTemplate and a list of a few shot examples. The class formats the prompt template with a few shot examples, which helps the language model generate a better response. We can streamline this process by utilizing LangChain's FewShotPromptTemplate to structure the approach.

```python

from langchain import PromptTemplate
from langchain import FewShotPromptTemplate

# create our examples
examples = [
    {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
]

# create an example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are excerpts from conversations with an AI
assistant. The assistant is known for its humor and wit, providing
entertaining and amusing responses to users' questions. Here are some
examples:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few-shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

```

- After creating a template, pass the example and user query:

```python

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

# load the model
chat = ChatOpenAI(model_name="gpt-4", temperature=0.0)

chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
chain.run("What's the meaning of life?")

```