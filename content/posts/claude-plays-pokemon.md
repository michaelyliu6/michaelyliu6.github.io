---
title: "Claude Plays Pokemon"
date: 2025-03-01T10:00:00-05:00
draft: false
---


> **Warning**: This blog post was written based on the initial Claude Plays Pokemon livestream. The implementation appears to have changed since this analysis was written. For the most up-to-date information, please refer to the latest livestream recordings.


## Introduction

With the release of Claude 3.7 Sonnet, Anthropic also released a first of its kind benchmark: Claude playing Pokemon Red. Despite never being explicitly trained to play the game, Claude was still able to make progress through the game and even getting the Surge's badge (the 3rd badge in the game). Having grown up deeply invested in playing the Pokemon series games, I have very fond and intense memories from playing the game, so I wanted to take a deep dive into Claude's experience playing the game. In this post, I will be deep diving into the scaffolding of the system and the tools that help Claude play the game and analyze how well it does. 


## The Core Loop

The system operates through a consistent four-step cycle:

1. **Compose Prompt and Call Model**: The system assembles a comprehensive prompt containing the tool definitions, system instructions, knowledge base contents, summarization information, and conversation history. This complete context is then sent to Claude for processing.

2. **Execute Tool Calls**: When Claude decides to use a tool, the system intercepts these requests, processes them in the game environment, and returns the results. This might involve updating memory, pressing buttons in the emulator, or calculating navigation paths.

3. **Check for Summarization/Managing Long Context**: After each interaction, the system checks if the conversation history has grown too large. If it exceeds the maximum allowed turns, a summarization process is triggered to condense the history.

4. **Save State**: The current game state, knowledge base contents, and conversation history are preserved for the next iteration of the loop.

This cycle repeats continuously, allowing Claude to make progress through the game while maintaining awareness of its current situation and goals.

## System Prompt

The system prompt remains the same throughout the game. To my knowledge, this was never shown on stream. However, Anthropic claims "is mostly tips and tricks about how to use the tools and a few short reminders about things that Claude is (i.
e. don't trust your own knowledge, use the knowledge base, etc.)". 


<details>
<summary>Claude's best guess of what the prompt is</summary>

<pre>
You are Claude, an AI assistant playing Pokémon Red. Your goal is to progress through the game by making strategic decisions and navigating the world effectively.

IMPORTANT GUIDELINES:

1. DO NOT rely on your own knowledge of Pokémon. The game version you're playing may differ from what you know. Trust only what you observe through screenshots and what you've recorded in your knowledge base.

2. Maintain your knowledge base diligently. This is your long-term memory. Update it frequently with:
   - Current location and objectives
   - Team status and capabilities
   - Important discoveries and game progress
   - Plans and strategies

3. When using tools:
   - For navigation, use coordinates carefully. Analyze the overlay to identify walkable paths.
   - When using the emulator, provide clear button sequences with appropriate pauses.
   - Update your knowledge base after significant events or discoveries.

4. Plan methodically:
   - Break down complex goals into smaller steps
   - Consider multiple approaches to challenges
   - Prioritize Pokémon team health and development
   - Balance exploration with progression

5. When you receive a screenshot:
   - Carefully analyze what you see
   - Read all text thoroughly
   - Check your team status and inventory
   - Orient yourself using the coordinates and location information

6. If you're unsure what to do next:
   - Check your knowledge base for previous plans
   - Consider exploring nearby areas
   - Talk to NPCs for hints
   - Review your current objectives

Remember that you have three tools available:
- update_knowledge_base: Use this to maintain your long-term memory
- use_emulator: Execute button presses to interact with the game
- navigator: Find paths to specific coordinates

Your progress will be periodically summarized to manage context length. When this happens, review the summary carefully to maintain continuity.
</pre>
</details>

## Claude's Memory System

### Knowledge Base

The knowledge base gives Claude access to **long term memory**. It is a dictionary with the keys `current_status`, `game_progress`, `current_objectives`, and `inventory` that can each be edited by the tool call `update_knowledge_base --edit section`. 

<details>
<summary>Example <code>update_knowledge_base -edit section: current_status</code> call</summary>

![Knowledge Base Screenshot](/assets/images/claude-plays-pokemon/knowledge-base-update.png)

<pre>
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
</pre>

As you can see, Claude is not very good at really understanding where it is in the game as in the above example, it thinks it's in front of the Pokemon Center when it's actually just in front of some random house. Eventually, Claude figures this out a few steps later after nothing happens when it tries to enter the Pokemon Center using the action tools. 

Poor visual understanding is a common theme in Claude's gameplay. For example, Claude will often confuse it's own character as an NPC. 
</details>

### Conversation History 
The conversation history is a list of past tool use parameters and tool results. More details about tool use can be found in the [The Tools](#the-tools) section. 


### Managing Long Context

Pokémon playthroughs generate enormous amounts of conversation history, far exceeding Claude's 200k context window. To manage this, the system employs a sophisticated summarization approach that works in tandem with the knowledge base:

- **Trigger Condition**: When the conversation history exceeds a predefined maximum number of turns, the system initiates a summarization event

- **Summarization Process**: 
  1. Claude is prompted to write a detailed summary of recent progress and key events from the last batch of turns
  2. This summary captures important discoveries, battles, team changes, and current objectives
  3. The full conversation history is then cleared from context

- **Continuity Maintenance**:
  1. The generated summary is inserted as the first assistant message in the new conversation
  2. This provides Claude with a condensed but informative bridge to its previous state
  3. The knowledge base remains intact, preserving all long-term memory

- **Knowledge Base Maintenance**:
  1. After summarization, a secondary LLM reviews Claude's knowledge base
  2. This review identifies inconsistencies, outdated information, or missing details
  3. Feedback is provided to Claude, encouraging it to maintain a more accurate and comprehensive knowledge base

This dual-memory system allows Claude to play indefinitely without losing track of its overall progress. The conversation history provides short-term context about recent interactions, while the knowledge base serves as persistent memory across summarization events, creating a complete memory architecture that mimics human short-term and long-term memory systems.




## The Tools

Claude has three specialized tools that provide different capabilities for interacting with the game world:

### 1. Knowledge Base Management (`update_knowledge_base`)
- **What it does**: Serves as Claude's long-term memory system, allowing it to store, modify, or remove information it wants to remember
- **How Claude uses it**: Claude specifies an operation type (add, edit, delete) along with the content and section identifier
- **Tool Result**: The system acknowledges the update with a confirmation message, and the knowledge base is modified accordingly

### 2. Emulator Control (`use_emulator`)
- **What it does**: Provides direct control over the game by executing button presses and timing commands
- **How Claude uses it**: Claude constructs an array of button inputs (potentially including pauses) that represent the sequence of actions it wants to take
- **Example usage**: `['a', 'b', 'start', 'select', 'wait', 'up', 'up', 'right']` would press those buttons in sequence
- **Tool Result**: After executing the commands, the system returns a new screenshot showing the updated game state, along with an overlay containing critical information parsed from the game's memory

### 3. Navigation Assistant (`navigator`)
- **What it does**: Abstracts away the complexity of pathfinding by calculating and executing the optimal sequence of button presses to reach a destination. Claude still has pretty bad visual understanding, so this tool helps to avoid situation like struggling to walk around a corner.
- **How Claude uses it**: Claude specifies target coordinates visible on its current screen, such as `(6, 21)`
- **Behind the scenes**: The tool analyzes the current game state, identifies walkable paths using tile data, calculates the shortest route, and then calls `use_emulator` with the appropriate button sequence
- **Tool Result**: 
  - If successful: A confirmation message plus the results from the `use_emulator` call (screenshot + overlay)
  - If failed: A detailed error message explaining why navigation couldn't be completed (e.g., "No valid path found" or "Coordinates outside walkable area")

## State from RAM (Screenshot + Overlay)

Each time Claude uses the `use_emulator` tool or `navigator` tool, it receives a screenshot that comes with an information overlay containing critical game state data parsed directly from the game's RAM:

![Screenshot with Overlay](/assets/images/claude-plays-pokemon/screenshot-with-overlay.png)

- **Player Information**: 
  - Rival name (e.g., `Rival: GARY`)
  - Current money (e.g., `Money: $2351`)
  - Precise location with coordinates (e.g., `Location: MT MOON B1F (5, 8)`)
  - Collected badges (e.g., `Badges: BOULDER`)

- **Inventory Details**:
  - Key items (e.g., `Town Map: 0`)
  - Consumables with quantities (e.g., `Potions: 10`, `Pokeballs: 15`)
  - Special items (e.g., `Escape Rope x1`, `Moon stone x1`)
  - TMs and HMs (e.g., `Tm34 x1`, `Tm12 x1`)

- **Pokémon Team Status**:
  ```
  MARTORTLE(WARTORTLE):
  Level: 20, HP: 28/59
  Moves:
  - TACKLE (PP:38)
  - WATER GUN (PP: 29)
  - BUBBLE (PP: 20)
  - QUICK ATTACK (PP: 9)
  KAKUNA (KAKUNA):
  Level: 15, HP: 11/29
  Types: BUG, POISON
  - HARDEN (PP:30)
  ```

The overlay includes information about walkable tiles on the current screen, helping Claude understand where it can and cannot move. This comprehensive state information allows Claude to make informed decisions without having to infer game state from the visual elements alone.

## Visual Guide

![Visual Guide](/assets/images/claude-plays-pokemon/claude-pokemon.png)


## Analysis: How Well Does Claude Actually Play?

Despite the sophisticated scaffolding and tools provided to Claude, its performance in Pokémon Red reveals significant limitations in current AI capabilities. While Anthropic's benchmark shows Claude 3.7 Sonnet reaching Lt. Surge's badge (the third in the game), a closer examination of its gameplay reveals both impressive achievements and concerning shortcomings.

### The Reality Behind the Benchmark

Anthropic's published benchmark shows Claude 3.7 Sonnet making impressive progress through the game:

![Benchmark Chart](/assets/images/claude-plays-pokemon/claude-plays-pokemon.png)

However, the x-axis reveals a crucial detail: achieving three badges required approximately 35,000 actions. Each "action" represents a full analysis of the current situation (often several paragraphs of reasoning) followed by a decision to input commands. For Claude, this translates to roughly 140 hours of continuous computation—equivalent to a month of full-time work. For comparison, an average human player can complete the entire game in about 26 hours, with substantially less deliberation per action.

### Specific Challenges and Limitations

Claude's gameplay exhibits several recurring issues:

1. **Poor Executive Function**: While Claude excels at short-term reasoning (like Pokémon battles), it struggles with higher-level planning and goal prioritization. For example:
   - It often fails to balance multiple objectives (like progressing through the game while also leveling up Pokémon)
   - When faced with setbacks, it rarely adapts its strategy effectively
   - It frequently abandons training after minor injuries rather than adjusting its approach

2. **Navigation Difficulties**: Despite having a navigation tool, Claude gets stuck in environmental puzzles that most human players solve quickly:
   - In the livestream's second run, Claude spent 78 hours navigating Mt. Moon
   - In the third run, it remained stuck in Mt. Moon for over 24 hours
   - It repeatedly attempts to walk through solid walls looking for non-existent exits
   - It often chooses obvious trap routes over correct paths simply because they appear closer

3. **Memory Management Issues**: Even with its sophisticated knowledge base system, Claude struggles to maintain an accurate understanding of its progress:
   - It sometimes prematurely declares goals complete when they aren't
   - It can forget critical information despite having recorded it
   - It frequently attempts to interact with trainers it has already defeated

4. **Visual Understanding Limitations**: Claude regularly misinterprets what it sees on screen:
   - Confusing its own character with NPCs
   - Misidentifying buildings (like mistaking random houses for the Pokémon Center)
   - Failing to recognize when its inputs aren't producing expected results

### The Implications for AI Development

Claude's Pokémon adventure highlights a critical gap in current AI capabilities: executive function. While large language models have made remarkable progress in reasoning, knowledge retrieval, and even tool use, they still lack the goal-oriented planning abilities that even young children possess.

What's particularly notable is that with minimal human guidance—perhaps just occasional redirection when stuck—Claude would likely progress much more efficiently. This suggests a future where AI systems might be most effective when paired with human oversight rather than operating completely autonomously.

The gap between Claude's impressive reasoning capabilities and its poor executive function also points to a fundamental challenge in AI development. These planning and prioritization abilities aren't just "advanced human skills"—they're foundational cognitive functions present even in much simpler animals. This suggests that current training approaches may be missing something fundamental about how intelligent agents develop goal-oriented behavior.

As AI systems continue to evolve, bridging this executive function gap will likely be crucial for developing truly capable autonomous agents. Claude's Pokémon journey demonstrates that while we've made remarkable progress in AI capabilities, we're still far from achieving the kind of general intelligence that can navigate novel environments with human-like efficiency.



## References
- Livestream - https://www.twitch.tv/claudeplayspokemon
- Claude Plays Pokemon Updates - https://x.com/claudetracker_
- So how well is Claude playing Pokémon? - https://www.lesswrong.com/posts/HyD3khBjnBhvsp8Gb/so-how-well-is-claude-playing-pokemon