+++
title = 'Projects'
draft = false
showToc = true
TocOpen = true
hideCitation = true
+++

## GPT-2 Transformer Implementation and Mechanistic Interpretability Experimentation
![GPT-2 Transformer](/assets/images/gpt2-transformer-image.png)


A comprehensive implementation and exploration of transformer-based language models, focusing on GPT-2 architecture and mechanistic interpretability. This project features three main components: 
- a production-ready lightweight implementation with efficient training pipelines based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- a clean, educational GPT-2 implementation from scratch with detailed documentation and intuition (includes sampling techniques with temperature, top-k, top-p, beam search and KV caching)
- advanced mechanistic interpretability tools for analyzing model internals including attention patterns, induction heads, feature representations, circuit behavior, and toy models of superposition

Built with PyTorch, [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), and integrated with Weights & Biases for experiment tracking.

[View Project on GitHub](https://github.com/michaelyliu6/transformers)

<details>
<summary>Image Prompt</summary>
<i>An anime-style visualization of a transformer architecture laboratory. In the foreground, a character with digital glasses is analyzing a glowing, multi-layered neural network structure. The central feature is an exploded view of a transformer block with attention heads visualized as colorful beams connecting token representations. Each attention head is depicted as a unique anime-style entity with its own personality, examining different aspects of the input text. The scene shows multiple screens displaying attention patterns, with one large display showing how different heads attend to different parts of a sentence. Another screen visualizes the internal representations of words transforming as they pass through each layer. The laboratory features circuit diagrams floating in holographic displays, showing the flow of information through the model with particular emphasis on induction heads and trigram detection circuits. In the background, several smaller anime characters represent different components of the architecture: embedding lookup tables, feed-forward networks, and layer normalization. The entire scene is bathed in a blue-green digital glow, with streams of token embeddings flowing between components. Mathematical equations for attention mechanisms and layer transformations are elegantly integrated into the scene's design elements. The visualization combines technical accuracy with an artistic anime aesthetic, making the complex architecture both beautiful and comprehensible.</i>
<br><br>
</details>


<details>
<summary>References</summary>
- ARENA Chapter 1: Transformer Interpretability - <a href="https://arena-chapter1-transformer-interp.streamlit.app/">https://arena-chapter1-transformer-interp.streamlit.app/</a><br>
- ML Alignment Bootcamp (MLAB) - <a href="https://github.com/Kiv/mlab2">https://github.com/Kiv/mlab2</a><br>
- Attention Is All You Need - <a href="https://arxiv.org/pdf/1706.03762">https://arxiv.org/pdf/1706.03762</a><br>
- Language Models are Unsupervised Multitask Learners (GPT-2) - <a href="https://arxiv.org/pdf/2005.14165">https://arxiv.org/pdf/2005.14165</a><br>
- Language Models are Few-Shot Learners (GPT-3) - <a href="https://arxiv.org/pdf/2005.14165">https://arxiv.org/pdf/2005.14165</a><br>
- What is a Transformer? (Transformer Walkthrough Part 1/2) - <a href="https://youtu.be/bOYE6E8JrtU?si=aZ2KFIXRjOyxWr52">https://youtu.be/bOYE6E8JrtU?si=aZ2KFIXRjOyxWr52</a><br>
- A Mathematical Framework for Transformer Circuits - <a href="https://transformer-circuits.pub/2021/framework/index.html">https://transformer-circuits.pub/2021/framework/index.html</a><br>
- An Analogy for Understanding Transformers - <a href="https://www.lesswrong.com/posts/euam65XjigaCJQkcN/an-analogy-for-understanding-transformers">https://www.lesswrong.com/posts/euam65XjigaCJQkcN/an-analogy-for-understanding-transformers</a><br>
- Induction heads - illustrated - <a href="https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated">https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated</a><br>
- Transformer Feed-Forward Layers Are Key-Value Memories - <a href="https://arxiv.org/pdf/2012.14913">https://arxiv.org/pdf/2012.14913</a><br>
- Toy Models of Superposition - <a href="https://transformer-circuits.pub/2022/toy_model/index.html">https://transformer-circuits.pub/2022/toy_model/index.html</a>
</details>


## Reinforcement Learning Implementations
![Reinforcement Learning](/assets/images/rl-project-image.png)


This project implements a range of RL techniques, culminating in Reinforcement Learning from Human Feedback (RLHF) for transformer language models. It demonstrates a progression from basic RL (multi-armed bandits, Q-learning) to advanced methods (DQN, PPO, and RLHF). This project features four main components:

- Core RL concepts by implementing classic bandit algorithms (Epsilon-Greedy, UCB, etc.) using `gymnasium`.
- Model-free RL with Q-Learning, SARSA, and Deep Q-Networks (DQN), including a replay buffer. Uses CartPole and probe environments.
- Policy-gradient algorithm, actor-critic architecture, GAE, and a clipped surrogate objective.  Applies to CartPole (and Atari work in progress).
- Applying RL to train transformer language models using human feedback (simulated).  Includes a value head, reward function, KL divergence penalty, and a complete RLHF training loop using `transformer_lens`.
  
Built with PyTorch, OpenAI Gym, and integrated with Weights & Biases for experiment tracking.
   

[View Project on GitHub](https://github.com/michaelyliu6/reinforcement-learning)

<details>
<summary>Image Prompt</summary>
<i>An anime-style scene depicting a group of cute robot characters in a world made of classic Atari game elements. In the foreground, an excited robot with glowing eyes and animated facial expressions has just successfully navigated through a Pac-Man-style maze filled with colorful dots and ghosts. The robot stands triumphantly at the maze exit, surrounded by sparkling reward particles and a floating '10000 POINTS' text in retro pixelated font. Behind it, the conquered maze shows its successful path highlighted in glowing light. From the successful robot's core, streams of colorful data and code are flowing back to three other robot characters waiting at different Atari-inspired challenges: one facing a wall of Space Invaders aliens, another preparing to bounce a Breakout ball with a paddle, and a third positioned before a Pong game setup. Each watching robot has holographic displays showing the successful algorithm and strategy being shared. All robots have distinct anime designs with expressive digital eyes, sleek bodies with retro gaming color schemes (reds, blues, yellows), and cute proportions. The background features a pixelated landscape with more Atari game elements including Adventure dragons and Asteroids space rocks. The scene is rendered in vibrant anime style with clean lines, digital effects, and the characteristic glow of arcade screens illuminating the robots' metallic surfaces.</i> - Generated by Flux 1.1 Pro
<br><br>

</details>

<details>
<summary>References</summary>
- ARENA Chapter 2: Reinforcement Learning - <a href="https://arena-chapter2-rl.streamlit.app/">https://arena-chapter2-rl.streamlit.app/</a><br>
- ML Alignment Bootcamp (MLAB) - <a href="https://github.com/Kiv/mlab2">https://github.com/Kiv/mlab2</a><br>
- Reinforcement Learning by Richard S. Sutton and Andrew G. Barto - <a href="https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf">https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf</a><br>
- Q-Learning - <a href="https://link.springer.com/content/pdf/10.1007/BF00992698.pdf">https://link.springer.com/content/pdf/10.1007/BF00992698.pdf</a><br>
- Playing Atari with Deep Reinforcement Learning - <a href="https://arxiv.org/pdf/1312.5602">https://arxiv.org/pdf/1312.5602</a><br>
- An introduction to Policy Gradient methods - Deep Reinforcement Learning - <a href="https://www.youtube.com/watch?v=5P7I-xPq8u8">https://www.youtube.com/watch?v=5P7I-xPq8u8</a><br>
- Proximal Policy Optimization Algorithms - <a href="https://arxiv.org/pdf/1707.06347">https://arxiv.org/pdf/1707.06347</a><br>
- The 37 Implementation Details of Proximal Policy Optimization - <a href="https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/">https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/</a><br>
- Deep Reinforcement Learning from Human Preferences - <a href="https://arxiv.org/pdf/1706.03741">https://arxiv.org/pdf/1706.03741</a><br>
- Illustrating Reinforcement Learning from Human Feedback (RLHF) - <a href="https://huggingface.co/blog/rlhf">https://huggingface.co/blog/rlhf</a><br>
- Training language models to follow instructions with human feedback - <a href="https://arxiv.org/pdf/2203.02155">https://arxiv.org/pdf/2203.02155</a>
</details>

---

## LLM Evaluation and Agent Framework
![LLM Evaluation](/assets/images/llm-eval-image.png)

This project investigates the emergent behaviors of Large Language Models (LLMs) through a rigorous evaluation framework, emphasizing AI safety and alignment. The work encompasses the development and assessment of LLM-based agents, the automated generation of adversarial test cases, and the application of the [`inspect`](https://inspect.ai-safety-institute.org.uk/) library for structured evaluation. Key areas of investigation include:


- **Alignment Faking:** Replication of findings from [Alignment Faking in Large Language Models](https://arxiv.org/pdf/2412.14093), demonstrating the potential for deceptive behavior in LLMs under varying deployment contexts, highlighting critical vulnerabilities in LLM deployment strategies.
- **Dataset Generation via Meta-Evaluation:** Threat Model design and automated generation of multiple-choice question datasets using LLMs, employing techniques such as few-shot prompting and iterative refinement. Concurrency is leveraged via `ThreadPoolExecutor` for efficient dataset creation.
- **Structured Evaluation with `inspect`:** Utilization of the `inspect` library to conduct systematic evaluations. This includes defining custom `solvers` to manipulate model inputs, `scorers` to quantify model outputs, and `tasks` to orchestrate the evaluation pipeline.
- **Agent-Based Evaluation:** Construction of autonomous agents leveraging LLM APIs, including the implementation of function calling for tool interaction. A "WikiGame" agent is developed using elicitation methods such as ReAct, Reflexion, etc, and evaluated on failure modes.


[View Project on GitHub](https://github.com/michaelyliu6/llm-evals)

<details>
<summary>Image Prompt</summary>
<i>An anime-style scene showcasing a recursive AI evaluation laboratory. In the foreground, a scientist character with glasses and a digital tablet is orchestrating a multi-layered evaluation system. The central feature is a striking "evaluation inception" visualization - a series of nested, glowing rings representing LLMs evaluating other LLMs. Each ring contains AI entities analyzing the output of inner-ring AIs, with data flowing between layers. One AI character is generating test cases, passing them to a second AI that's producing responses, while a third AI is scoring those responses with complex metrics floating around it. A fourth AI is analyzing those scores and refining the evaluation criteria, creating a perfect loop. Holographic displays show this recursive process with labels like "Meta-Evaluation Layer 3" and "Alignment Verification Loop." In the background, several agent robots navigate a Wikipedia-themed maze, but now they're being observed by evaluator robots taking notes on clipboards. The laboratory features fractal-like screens showing the same evaluation patterns repeating at different scales. Digital metrics flow between systems in colorful streams, with some screens showing "Evaluator Bias Analysis" and "Meta-Alignment Testing." The entire scene has a recursive aesthetic with evaluation processes visibly nested within each other, all rendered in vibrant anime style with expressive AI characters showing varying degrees of concentration as they evaluate their peers.</i> - Generated by Flux 1.1 Pro
<br><br>

</details>

<details>
<summary>References</summary>
- ARENA Chapter 3: LLM Evaluations - <a href="https://arena-chapter3-llm-evals.streamlit.app/">https://arena-chapter3-llm-evals.streamlit.app/</a><br>
- Alignment faking in large language models - <a href="https://arxiv.org/pdf/2412.14093">https://arxiv.org/pdf/2412.14093</a><br>
- Discovering Language Model Behaviors with Model-Written Evaluations - <a href="https://arxiv.org/pdf/2212.09251">https://arxiv.org/pdf/2212.09251</a><br>
- A starter guide for evals - <a href="https://www.alignmentforum.org/posts/2PiawPFJeyCQGcwXG/a-starter-guide-for-evals">https://www.alignmentforum.org/posts/2PiawPFJeyCQGcwXG/a-starter-guide-for-evals</a><br>
- LLM Powered Autonomous Agents - <a href="https://lilianweng.github.io/posts/2023-06-23-agent/">https://lilianweng.github.io/posts/2023-06-23-agent/</a><br>
- Evaluating Language-Model Agents on Realistic Autonomous Tasks - <a href="https://arxiv.org/pdf/2312.11671">https://arxiv.org/pdf/2312.11671</a><br>
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models - <a href="https://arxiv.org/pdf/2201.11903">https://arxiv.org/pdf/2201.11903</a><br>
- Large Language Models can Strategically Deceive their Users when Put Under Pressure - <a href="https://arxiv.org/pdf/2311.07590">https://arxiv.org/pdf/2311.07590</a><br>
- Answering Questions by Meta-Reasoning over Multiple Chains of Thought - <a href="https://arxiv.org/pdf/2304.13007">https://arxiv.org/pdf/2304.13007</a><br>
- Toolformer: Language Models Can Teach Themselves to Use Tools - <a href="https://arxiv.org/pdf/2302.04761">https://arxiv.org/pdf/2302.04761</a><br>
- ReAct: Synergizing Reasoning and Acting in Language Models - <a href="https://arxiv.org/pdf/2210.03629">https://arxiv.org/pdf/2210.03629</a><br>
- Reflexion: Language Agents with Verbal Reinforcement Learning - <a href="https://arxiv.org/pdf/2303.11366">https://arxiv.org/pdf/2303.11366</a>
</details>


---

## Diffusion and Multimodal Models Implementation
![Diffusion Models](/assets/images/diffusion-image.png)


A comprehensive implementation of a 2 MLP layer + U-Net diffusion models and multimodal architectures from scratch. This project features implementations of Denoising Diffusion Probabilistic Models (DDPM), Denoising Diffusion Implicit Models (DDIM), and CLIP (Contrastive Language-Image Pre-training). The codebase includes sophisticated U-Net architectures with attention mechanisms, classifier and CLIP guidance techniques for conditional generation, and various sampling methods. 

Built with PyTorch and integrated with Weights & Biases for experiment tracking.


[View Project on GitHub](https://github.com/michaelyliu6/diffusion)

<details>
<summary>Image Prompt</summary>
<i>An anime-style tech laboratory scene visualizing diffusion image generation. A central anime character with digital glasses operates a futuristic console labeled 'DIFFUSION MODEL' with multiple screens showing the same image at different denoising steps. The main display shows a 3D visualization of probability space with , where noise visibly transforms into multiple diverse anime images: HD of a cat, a cartoon of a friendly robot, and high definition landscape. Each generation step is marked with glowing nodes on an upward path, with t=1000 at the bottom (pure noise) and t=0 at the peak (clear images). The noise-to-image transition is clearly shown as particles coalescing into recognizable forms as they ascend the probability gradient. Floating holographic displays around the console show close-ups of the denoising process: one display shows sequential image frames evolving from static to clarity, another shows a visual representation of noise prediction at each step. A third display shows a heat map of where the model is focusing its attention during the current denoising step. The character manipulates particle streams flowing between time steps, with each stream containing tiny image fragments that become progressively more defined as they approach t=0. The lighting transitions from chaotic blue-purple for the noisy regions to structured golden light for the final image. The laboratory walls display animated equations and diagrams specifically showing the forward and reverse diffusion processes, with arrows indicating the direction of optimization. Above it all, a banner reads 'Denoising Diffusion Probabilistic Model' in stylized anime text. The scene includes multiple small denoising stages visible as floating platforms, each showing the diverse anime images getting clearer as the algorithm climbs toward the optimal distribution at the summit. Small holographic labels identify key concepts in the diffusion process: 'noise prediction,' 'variance scheduling,' and 'sampling path optimization.'</i> - Generated by Flux 1.1 Pro
<br><br>

</details>

<details>
<summary>References</summary>
- ML for Alignment Bootcamp (MLAB) - <a href="https://github.com/Kiv/mlab2">https://github.com/Kiv/mlab2</a><br>
- Denoising Diffusion Probabilistic Models - <a href="https://arxiv.org/pdf/2006.11239">https://arxiv.org/pdf/2006.11239</a><br>
- Denoising Diffusion Implicit Models - <a href="https://arxiv.org/pdf/2010.02502">https://arxiv.org/pdf/2010.02502</a><br>
- Learning Transferable Visual Models From Natural Language Supervision - <a href="https://arxiv.org/pdf/2103.00020">https://arxiv.org/pdf/2103.00020</a>
</details>

