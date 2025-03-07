---
title: "Claude Plays Pokemon"
date: 2025-03-01T10:00:00-05:00
draft: true

---


![Claude Plays Pokemon](/assets/images/claude-plays-pokemon/claude-plays-pokemon.png)

## The Core Loop


1. Compose Prompt and Call Model
2. Execute Tool Calls
3. Check for Summarization/Managing Long Context
4. Repeat



## The Prompt



### Tool Definition
   
Claude is allowed to use the following 3 tools:
- `update_knowledge_base`: Update the knowledge base with the latest information.
- `use_emulator`: Use the emulator to play the game.
- `navigator`: Navigate the game.

### System Prompt


The system prompt remains the same throughout the game. To my knowledge, this was never shown on stream. However, Anthropic claims "is mostly tips and tricks about how to use the tools and a few short reminders about things that Claude is (i.e. don't trust your own knowledge, use the knowledge base, etc.)". 


### Knowledge Base

The knowledge base gives Claude access to **long term memory**. It is a dictionary with the keys "Current Status", "Conversation History", "Navigation Progress", and "Immediate Plans" that can be edited by a tool call `update_knowledge_base - edit section`. 

```
Current Status:

Location: Cerulean City at coordinates (21, 26)
Activity: Standing direction in front of the Pokemon Center Entrance

Team Conditions:
- BOLT (Pikachu) at critically low HP (10/36)
- SPIKE (Nidorino) at low HP (10/51)
- SHELL (Wartortle) at reduced HP (28/51)
- All other Pokemon are in healthy condition
Money: $1839

Recent Discoveries:
- Successfully navigated through Mt. Moon and Route 4 to successfully reach Cerulean City
- Found TM04 on Route 4 at coordinates (57, 3)
- Completed inital exploration of Cerulean City and mapped key locations:
    - Pokemon Center: Located at coordinates (18-23, 24-26) with a blue roof (currently at entrance)
    - PokeMart: Located at coordinates (24-26, 24-26) with a red roof with "MART" sign
    - Bike Shop: Located at coordinates (13, 25) with yellow door (initially confused with Pokemon Center)

Navigation Progress:
- Successfully mapped the layout of southwestern and central Cerulean City
- Identified paths between major buildings and obstacles that blocked direct routes
- Currently positioned directly in front of the Pokemon Center Entrance


Immediate Plans:
1. Enter the Pokemon Center to heal the team
2. After healing, systematically explore Cerulean City to identify:
    - Misty's Gym (for 2nd badge)
    - Nugget Bridge (leads to Route 25)
    - Any Team Rocket activity mentioned by NPC at Route 4 Pokemon Center
3. Consider evolution options using MOON STONE after team is healed
```

### Conversation History 


The conversation history is a list of past tool use parameters and responses. To manage long contexts beyond Claude's 200k context window, the core loop will trigger a summarization event once `conversation_history > max_turns` where Claude will write a summary of what happened in the last `max_turns`. 




