## Introduction

- In LangChain, agents are high-level components that use language models (LLMs) to determine which actions to take and in what order. An action can either be using a tool and observing its output or returning it to the user. Tools are functions that perform specific duties, such as Google Search, database lookups, or Python REPL.

- Agents involve an LLM making decisions about which Actions to take, taking that Action, seeing an Observation, and repeating that until done.

- Several types of agents are available in LangChain:

    - The zero-shot-react-description agent uses the ReAct framework to decide which tool to employ based purely on the tool's description. It necessitates a description of each tool.
    - The react-docstore agent engages with a docstore through the ReAct framework. It needs two tools: a Search tool and a Lookup tool. The Search tool finds a document, and the Lookup tool searches for a term in the most recently discovered document.
    - The self-ask-with-search agent employs a single tool named Intermediate Answer, which is capable of looking up factual responses to queries. It is identical to the original self-ask with the search paper, where a Google search API was provided as the tool.
    - The conversational-react-description agent is designed for conversational situations. It uses the ReAct framework to select a tool and uses memory to remember past conversation interactions.

## Importing Necessary Modules

- langchain.llms.OpenAI: This is used to create an instance of the OpenAI language model, which can generate human-like text based on the input it's given.
- langchain.agents.load_tools: This function is used to load a list of tools that an AI agent can use.
- langchain.agents.initialize_agent: This function initializes an AI agent that can use a given set of tools and a language model to interact with users.
- langchain.agents.Tool: This is a class used to define a tool that an AI agent can use. A tool is defined by its name, a function that performs the tool's action, and a description of the tool.
- langchain.utilities.GoogleSearchAPIWrapper: This class is a wrapper for the Google Search API, allowing it to be used as a tool by an AI agent. It likely contains a method that sends a search query to Google and retrieves the results.

```python

from langchain.llms import OpenAI

from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper

# Initialize the LLM

llm = OpenAI(model="text-davinci-003", temperature=0)


# Define the Google search Wrapper

# remember to set the environment variables
# “GOOGLE_API_KEY” and “GOOGLE_CSE_ID” to be able to use
# Google Search via API.
search = GoogleSearchAPIWrapper()

# Agent that Uses Google Search Tool

tools = [
    Tool(
        name = "google-search",
        func=search.run,
        description="useful for when you need to search google to answer questions about current events"
    )
]

agent = initialize_agent(tools, 
                         llm, 
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                         verbose=True,
                         max_iterations=6)

# Check out Response
response = agent("What's the latest news about the Mars rover?")
print(response['output'])

```

```MD

# Reponse

> Entering new AgentExecutor chain...
 I need to find out the latest news about the Mars rover
Action: google-search
Action Input: "latest news Mars rover"
Observation: The mission has concluded that the solar-powered lander has run out of energy after more than four years on the Red Planet. Dec 15, 2021 ... Mars Sample Return is going to have great stuff to choose from!” Get the Latest JPL News. SUBSCRIBE TO THE NEWSLETTER. NASA's Curiosity rover is discovering odd rocks halfway into a region called the "clay-bearing unit." ... This panorama of a location called "Teal Ridge" was ... 2 days ago ... LEGO Technic 42158 NASA Mars Rover Perseverance officially revealed as latest real-life space set [News]. Today LEGO has officially revealed ... Oct 19, 2022 ... NASA's Curiosity Mars rover used its Mast Camera, or Mastcam, to capture this panorama of a hill nicknamed ... Get the Latest JPL News. Latest Updates · Curiosity rover finds water-carved 'book' rock on Mars (photo) · Curiosity rover on Mars gets a brain boost to think (and move) faster · Curiosity ... Mar 7, 2023 ... 27, helps provide scientists with information about the particle sizes within the clouds. Sign up for National Breaking News Alerts. Sep 16, 2022 ... Since July, NASA's Perseverance rover has drilled and collected four slim cores of sedimentary rock, formed in what was once a lake on Mars. Nasa Mars lander study reveals 'main source of heat' on Red Planet · NASA/ESA/A. Simon (Goddard Space Flight Center) and M.H. Wong. All the latest content about Nasa Perseverance Mars rover from the BBC. ... James Tytko presents science news and we're putting tuberculosis under the ...
Thought: I now know the final answer
Final Answer: The latest news about the Mars rover is that the mission has concluded that the solar-powered lander has run out of energy after more than four years on the Red Planet. NASA's Curiosity rover is discovering odd rocks halfway into a region called the "clay-bearing unit." LEGO Technic 42158 NASA Mars Rover Perseverance has been officially revealed as the latest real-life space set. NASA's Curiosity Mars rover used its Mast Camera, or Mastcam, to capture a panorama of a hill nicknamed "Teal Ridge." NASA's Perseverance rover has drilled and collected four slim cores of sedimentary rock, formed in what was once a lake on Mars. A study of the Mars lander has revealed the 'main source of heat' on the Red Planet.

> Finished chain.
The latest news about the Mars rover is that the mission has concluded that the solar-powered lander has run out of energy after more than four years on the Red Planet. NASA's Curiosity rover is discovering odd rocks halfway into a region called the "clay-bearing unit." LEGO Technic 42158 NASA Mars Rover Perseverance has been officially revealed as the latest real-life space set. NASA's Curiosity Mars rover used its Mast Camera, or Mastcam, to capture a panorama of a hill nicknamed "Teal Ridge." NASA's Perseverance rover has drilled and collected four slim cores of sedimentary rock, formed in what was once a lake on Mars. A study of the Mars lander has revealed the 'main source of heat' on the Red Planet.

```
- The Tool object represents a specific capability or function the system can use. In this case, it's a tool for performing Google searches.

- It is initialized with three parameters:

    - name parameter: This is a string that serves as a unique identifier for the tool. In this case, the name of the tool is "google-search.”
    - func parameter: This parameter is assigned the function that the tool will execute when called. In this case, it's the run method of the search object, which presumably performs a Google search.
    - description parameter: This is a string that briefly explains what the tool does. The description explains that this tool is helpful when you need to use Google to answer questions about current events.

- initialize_agent(): This function call creates and initializes an agent. An agent is a component that determines which actions to take based on user input. These actions can be using a tool, returning a response to the user, or something else.
- tools:  represents the list of Tool objects that the agent can use.
- agent="zero-shot-react-description": The "zero-shot-react-description" type of an Agent uses the ReAct framework to decide which tool to use based only on the tool's description.
- verbose=True: when set to True, it will cause the Agent to print more detailed information about what it's doing. This is useful for debugging and understanding what's happening under the hood.
- max_iterations=6: sets a limit on the number of iterations the Agent can perform before stopping. It's a way of preventing the agent from running indefinitely in some cases, which may have unwanted monetary costs.