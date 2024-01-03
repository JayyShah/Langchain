# Tools in Langchain

- LangChain provides a variety of tools for agents to interact with the outside world. These tools can be used to create custom agents that perform various tasks, such as searching the web, answering questions, or running Python code. 

- The Google Search tool, using the GoogleSearchAPIWrapper, will handle queries that involve finding recent event information. The Language Model tool leverages the capabilities of a language model to summarize texts. These tools are designed to be used interchangeably by the agent, depending on the nature of the user's query.

```python

# Dependencies

from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

# Instantiate a LLM chain specifically for text Summarization

llm = OpenAI(model="text-davinci-003", temperature=0)

prompt = PromptTemplate(
    input_variables=["query"],
    template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

# Google Search via API.
search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events"
    ),
    Tool(
       name='Summarizer',
       func=summarize_chain.run,
       description='useful for summarizing texts'
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

response = agent("What's the latest news about the Mars rover? Then please summarize the results.")
print(response['output'])

```

```MD

# Response

> Entering new AgentExecutor chain...
 I should search for recent news about the Mars rover and then summarize the results.
Action: Search
Action Input: Latest news about the Mars rover
Observation: Mars 2020 Perseverance Rover · The largest and most capable rover ever sent to Mars. ... Curiosity Rover · Measures Mars' atmosphere to understand its climate ... Dec 15, 2021 ... Mars Sample Return is going to have great stuff to choose from!” Get the Latest JPL News. SUBSCRIBE TO THE NEWSLETTER. The mission has concluded that the solar-powered lander has run out of energy after more than four years on the Red Planet. Oct 19, 2022 ... NASA's Curiosity Mars rover used its Mast Camera, or Mastcam, to capture this panorama of a hill nicknamed ... Get the Latest JPL News. NASA's Mars 2020 Perseverance rover will look for signs of past microbial life, cache rock and soil samples, and prepare for future human exploration. Latest Updates · Curiosity rover finds water-carved 'book' rock on Mars (photo) · Curiosity rover on Mars gets a brain boost to think (and move) faster · Curiosity ... Feb 8, 2023 ... Curiosity Rover Finds New Clues to Mars' Watery Past ... at Gediz Vallis Ridge twice last year but could only survey it from a distance. Sep 16, 2022 ... Since July, NASA's Perseverance rover has drilled and collected four slim cores of sedimentary rock, formed in what was once a lake on Mars. Mar 7, 2023 ... 27, helps provide scientists with information about the particle sizes within the clouds. Sign up for National Breaking News Alerts. All the latest content about Nasa Perseverance Mars rover from the BBC. ... James Tytko presents science news and we're putting tuberculosis under the ...
Thought: I should summarize the results of the search.
Action: Summarizer
Action Input: Mars 2020 Perseverance Rover is the largest and most capable rover ever sent to Mars. It measures Mars' atmosphere to understand its climate and has run out of energy after more than four years on the Red Planet. NASA's Curiosity Mars rover used its Mast Camera to capture a panorama of a hill nicknamed "Book Rock". NASA's Mars 2020 Perseverance rover will look for signs of past microbial life, cache rock and soil samples, and prepare for future human exploration. Curiosity rover finds water-carved 'book' rock on Mars, Curiosity rover on Mars gets a brain boost to think (and move) faster, Curiosity Rover Finds New Clues to Mars' Watery Past, and NASA's Perseverance rover has drilled and collected four slim cores of sedimentary rock.
Observation: 

NASA's Mars 2020 Perseverance rover is the largest and most capable rover ever sent to Mars. It has been on the Red Planet for more than four years, measuring the atmosphere to understand its climate and searching for signs of past microbial life. The Curiosity rover has captured a panorama of a hill nicknamed "Book Rock" and has been given a brain boost to think and move faster. It has also found new clues to Mars' watery past. The Perseverance rover has drilled and collected four slim cores of sedimentary rock, which will be used to cache rock and soil samples and prepare for future human exploration.
Thought: I now know the final answer.
Final Answer: NASA's Mars 2020 Perseverance rover is the largest and most capable rover ever sent to Mars. It has been on the Red Planet for more than four years, measuring the atmosphere to understand its climate and searching for signs of past microbial life. The Curiosity rover has captured a panorama of a hill nicknamed "Book Rock" and has been given a brain boost to think and move faster. It has also found new clues to Mars' watery past. The Perseverance rover has drilled and collected four slim cores of sedimentary rock, which will be used to cache rock and soil samples and prepare for future human exploration.

> Finished chain.
NASA's Mars 2020 Perseverance rover is the largest and most capable rover ever sent to Mars. It has been on the Red Planet for more than four years, measuring the atmosphere to understand its climate and searching for signs of past microbial life. The Curiosity rover has captured a panorama of a hill nicknamed "Book Rock" and has been given a brain boost to think and move faster. It has also found new clues to Mars' watery past. The Perseverance rover has drilled and collected four slim cores of sedimentary rock, which will be used to cache rock and soil samples and prepare for future human exploration.

```

- Notice how the agents used at first the “Search” tool to look for recent information about the Mars rover and then used the “Summarizer” tool for writing a summary.

- LangChain provides an expansive toolkit that integrates various functions to improve the functionality of conversational agents. Here are some examples:

- SerpAPI: This tool is an interface for the SerpAPI search engine, allowing the agent to perform robust online searches to pull in relevant data for a conversation or task.
- PythonREPLTool: This unique tool enables the writing and execution of Python code within an agent. This opens up a wide range of possibilities for advanced computations and interactions within the conversation.