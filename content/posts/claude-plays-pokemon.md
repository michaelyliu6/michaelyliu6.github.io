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


The system prompt remains the same throughout the game. It is mostly tips and tricks about how to use the tools and a few short reminders about things that Claude is (i.e. don't trust your own knowledge, use the knowledge base, etc.)


### Knowledge Base

The knowledge base gives Claude access to **long term memory**. It is basically a python diction

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
    - Pokemon Center: located at coordinates (21, 26)
```

### Conversation History 





