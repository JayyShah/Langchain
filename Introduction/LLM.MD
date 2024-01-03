- The fundamental component of LangChain involves invoking an LLM with a specific input.Let's imagine we are building a service that suggests personalized workout routines based on an individual's fitness goals and preferences.

```python
    from langchain.llms import OpenAI
```

- The temperature parameter in OpenAI models manages the randomness of the output. When set to 0, the output is mostly predetermined and suitable for tasks requiring stability and the most probable result. At a setting of 1.0, the output can be inconsistent and interesting but isn't generally advised for most tasks. For creative tasks, a temperature between 0.70 and 0.90 offers a balance of reliability and creativity. The best setting should be determined by experimenting with different values for each specific use case. The code initializes the GPT-3 model’s Davinci variant.

```python
    # Before executing the following code, make sure to have
    # your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
    llm = OpenAI(model="text-davinci-003", temperature=0.9)


    # Call it on some input
    text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
    print(llm(text))
```

```raw
    # Response
    1) Monday: Jog for 30 minutes outside 
    2) Tuesday: 30 minutes of jumping jacks 
    3) Wednesday: 30 minutes of running stairs 
    4) Thursday: 30 minutes of sprints 
    5) Friday: 30 minutes of jogging on an inclined treadmill 
    6) Saturday: 30 minutes of interval training 
    7) Sunday: Walk for 30 minutes
```

# Chains

- In LangChain, a chain is an end-to-end wrapper around multiple individual components, providing a way to accomplish a common use case by combining these components in a specific sequence. The most commonly used type of chain is the LLMChain, which consists of a PromptTemplate, a model (either an LLM or a ChatModel), and an optional output parser.

The LLMChain works as follows:

1. Takes (multiple) input variables.
2. Uses the PromptTemplate to format the input variables into a prompt.
3. Passes the formatted prompt to the model (LLM or ChatModel).
4. If an output parser is provided, it uses the OutputParser to parse the output of the LLM into a final format.

```python

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(model="text-davinci-003", temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("eco-friendly water bottles"))

```

# The Memory

- In LangChain, Memory refers to the mechanism that stores and manages the conversation history between a user and the AI. It helps maintain context and coherency throughout the interaction, enabling the AI to generate more relevant and accurate responses. Memory, such as ConversationBufferMemory, acts as a wrapper around ChatMessageHistory, extracting the messages and providing them to the chain for better context-aware generation.

```python

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = OpenAI(model="text-davinci-003", temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# Continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

# Display the conversation
print(conversation)

```

```MD

## Response

**> Entering new ConversationChain chain...**

Prompt after formatting:

***The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Tell me about yourself.
AI:  Hi there! I'm an AI created to help people with their daily tasks. I'm programmed to understand natural language and respond to questions and commands. I'm also able to learn from my interactions with people, so I'm constantly growing and improving. I'm excited to help you out!

Human: What can you do?
AI:  I can help you with a variety of tasks, such as scheduling appointments, setting reminders, and providing information. I'm also able to answer questions about topics like current events, sports, and entertainment. I'm always learning new things, so I'm sure I can help you with whatever you need.

Human: How can you help me with data analysis?
AI:*  I'm not familiar with data analysis, but I'm sure I can help you find the information you need. I can search the web for articles and resources related to data analysis, and I can also provide you with links to helpful websites.**

**> Finished chain.**

```