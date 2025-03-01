+++
title = '"Real" Reinforcement Learning will create the strongest technical moats'
date = 2025-02-16T07:07:07+01:00
draft = false
showToc = true
TocOpen = true
+++


The AI landscape has undergone rapid shifts in recent years. While 2023-2024 saw the commoditization of pre-training and supervised fine-tuning, 2025 will mark the emergence of "real" Reinforcement Learning (RL) as the primary technical moat in AI development. Unlike pre-training, which focuses on learning statistical correlations from massive datasets, RL allows models to actively explore solution spaces and discover novel strategies that generalize beyond static training data.

## The Limitations of RLHF and the Promise of "Real" RL

Unlike RLHF (Reinforcement Learning from Human Feedback), which optimizes for human approval rather than actual task performance, genuine RL with sparse rewards will enable models to solve complex end-to-end tasks autonomously. RLHF is fundamentally limited because it optimizes for a proxy objective (what looks good to humans) rather than directly solving problems correctly. Furthermore, models quickly learn to game reward models when trained with RLHF for extended periods. In contrast, true RL with sparse rewards—similar to what powered AlphaGo's breakthrough—will create significant competitive advantages for several reasons.

## Why RL Will Be the Next AI Moat

### Proprietary Environments as a Barrier to Entry

Creating effective sandboxes for RL agents requires substantial engineering effort that cannot be easily replicated. Companies that build perfect environments tailored to their specific domains will gain lasting advantages as these environments represent significant intellectual property that's difficult to reverse-engineer or leak through espionage. Unlike pre-training architectures that can be replicated through academic papers, these specialized environments combine domain expertise with technical implementation in ways that are much harder to reproduce. The environment design itself becomes a form of tacit knowledge that doesn't transfer easily between organizations.

### Debugging RL: The Unseen Complexity

Debugging RL systems presents unique challenges compared to pre-training. While pre-training errors are typically local and feedback is immediate, RL suffers from non-local errors, noisy performance metrics, and delayed feedback. This makes expertise in RL debugging incredibly valuable and creates a talent moat that will be difficult for competitors to overcome. The debugging process for RL involves identifying complex causal chains across multiple timesteps, requiring specialized tooling and expertise that will take years to develop. Companies that build these capabilities first will maintain significant leads.

### Compute and Experimentation: The Ultimate Differentiator

Successful RL will require massive computational resources dedicated to failed experiments—likely more than what's required for pre-training. As new datacenters come online throughout 2025, the gap between resource-rich companies and others will widen substantially. The ability to run thousands of failed experiments before finding successful approaches will become a critical differentiator. This is fundamentally different from pre-training, where scaling laws are relatively well-understood. In RL, the relationship between compute investment and performance improvement is far more stochastic, favoring organizations that can afford to explore the solution space more thoroughly.

This computational divide is further exacerbated by the nature of RL training itself. While pre-training and supervised fine-tuning benefit from high parallelization and efficient prefilling of prompts, RL methods are more decode-heavy and sequential. Each policy update typically requires generating complete responses rather than just predicting the next token, making the process inherently less efficient at scale. This computational profile means that even with equivalent hardware, RL experimentation consumes disproportionately more resources than other training paradigms.

### Proprietary Data: From 99% to 99.999% Reliability

While proprietary data showed limited advantages in pre-training (where RAG proved more effective), it will finally prove decisive in RL by pushing performance from 99% to 99.999% reliability—the difference between market winners and also-rans. This proprietary data advantage will be particularly pronounced in domain-specific applications where companies have accumulated unique interaction histories. Unlike pre-training, where data diversity matters more than specificity, RL benefits enormously from data that captures rare edge cases and unusual interaction patterns that only emerge in production environments.

### Reward Design: The Art and Science of RL

Designing effective reward functions without introducing reward hacking or specification gaming requires rare intuition and creativity. Companies with talent that can anticipate these pitfalls will save enormous resources and avoid damaging public failures. This "evaluation artistry" will become a prized skill set, with certain individuals possessing the empathy for models and foresight to design robust reward mechanisms becoming highly sought after. The challenge of reward design is fundamentally different from loss function design in supervised learning, as it requires anticipating how agents might exploit loopholes in ways that aren't apparent from static datasets.

## The Moat Effect: Why RL Will Define AI Leadership

These advantages compound in ways that create insurmountable leads. Companies that develop better environments collect better data, which improves their models, which generates more useful data, creating a virtuous cycle that competitors can't easily break into. Additionally, the trust established by avoiding catastrophic failures becomes its own moat—organizations that demonstrate reliable, safe RL systems will be permitted to deploy in increasingly sensitive domains, generating even more valuable data and experience.

Effective deployment will likely involve hierarchical oversight systems, where larger models supervise smaller ones, stepping in when the primary agent gets stuck. Human oversight becomes the final backstop, but the goal will be minimizing human intervention through this cascade of increasingly capable oversight models. Companies that perfect this hierarchy will achieve both safety and scalability.

## Where RL Will Shine First

We'll likely see the first breakthroughs in domains with verifiable rewards (mathematics, coding) and contained environments with clear success metrics. Customer service stands out as a particularly promising early application—not only does it offer measurable outcomes, but the worst-case scenarios are relatively contained, where a suboptimal interaction might frustrate a customer but rarely leads to catastrophic consequences. Companies like Sierra are well-positioned to lead this transition with their focus on agent-based systems. These initial domains will serve as proving grounds for techniques that will later be applied to more complex problems. Beyond these easily verifiable areas, however, the path forward remains highly uncertain. How to effectively define, measure, and optimize for rewards in subjective or open-ended domains without clear ground truth continues to be a significant open research question that will likely take longer to resolve.

## The Future Competitive Landscape

The implications are significant: while base models may continue to commoditize, the ability to create effective RL systems will concentrate power among companies with the right expertise, data, and compute resources. Open source efforts may struggle to keep pace in this new paradigm, as the domain-specific nature of effective RL environments favors concentrated corporate efforts. While open source has successfully replicated many pre-training advances, RL requires concentrated investment in specific domains rather than general capabilities, making it less amenable to distributed development efforts.

