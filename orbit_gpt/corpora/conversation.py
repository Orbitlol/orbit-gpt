"""Generate a large, varied dialogue corpus - offline, deterministic, no downloads.

Why generate data at all?  Because a hand-written corpus of a few hundred fixed
exchanges gets *memorised verbatim* by a model this size: it then "answers" by
reciting whichever line followed a similar question, which feels like
copy-paste, not conversation.

Here every intent is expressed with many different phrasings, every slot
(numbers, units, topics, cities) is sampled, and answers come in several
variants.  Nothing can be memorised line by line, so the only way to do well is
to actually learn the pattern - which is what lets the model answer a phrasing
or an arithmetic problem it has never seen.

Facts and answers are computed where they can be (arithmetic, unit
conversions), so the numbers are always correct.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# how a question gets phrased
# ---------------------------------------------------------------------------
WHAT_ASKS = [
    "What is {t}?", "What's {t}?", "What is {t} exactly?",
    "Can you explain {t}?", "Could you explain {t} to me?",
    "Tell me about {t}.", "Tell me what {t} is.",
    "Do you know what {t} is?", "I don't really get {t}.",
    "Explain {t}, please.", "Explain {t} in simple terms.",
    "How would you explain {t}?", "What does {t} mean?",
    "{t} - what is it?", "Give me a quick definition of {t}.",
    "I've heard about {t}. What is it?", "Can you tell me what {t} means?",
]

HOW_ASKS = [
    "How do I {t}?", "How can I {t}?", "How do you {t}?",
    "What's the best way to {t}?", "How to {t}?",
    "Can you show me how to {t}?", "I need to {t} - how?",
    "Any tips on how to {t}?", "What's the easiest way to {t}?",
    "How should I {t}?", "I want to {t}. How?",
    "Could you help me {t}?", "Is there a simple way to {t}?",
]

# Deliberately NOT used when building the corpus: the tests use these to check
# that the model answers a phrasing it has never seen.
TEST_ASKS = [
    "I've always wondered what {t} is.",
    "Quick one: {t}?",
    "Be honest, what is {t}?",
]
TEST_HOW_ASKS = [
    "What's your method for {t}?",
    "I keep failing to {t} - any advice?",
]

ANSWER_OPENERS = [
    "", "", "", "Sure - ", "Of course. ", "Happy to help. ",
    "Good question. ", "Here's the short version: ", "In short: ",
]

# ---------------------------------------------------------------------------
# knowledge: topic -> several answer variants
# ---------------------------------------------------------------------------
FACTS: Dict[str, List[str]] = {
    "Python": [
        "Python is a high-level programming language known for readable syntax. It's used for scripting, data analysis, automation and machine learning.",
        "Python is a general-purpose programming language that favours readability. People use it for automation, data work, web backends and AI.",
        "Python is an interpreted, dynamically typed language with a very readable syntax, popular in data science, automation and web development.",
    ],
    "Git": [
        "Git is a version control system that records snapshots of your files, so you can see history, branch out and roll back mistakes.",
        "Git is a distributed version control system: every clone has the full history, which makes branching and offline work easy.",
        "Git tracks changes to your files over time. You commit snapshots, work on branches, and can always return to an earlier state.",
    ],
    "JSON": [
        "JSON is a lightweight text format for structured data: objects, arrays, strings, numbers, booleans and null. It's the default format for web APIs.",
        "JSON is a text-based data format built from key-value objects and arrays. It maps almost directly onto Python dicts and lists.",
        "JSON is a human-readable way to serialise data, used by nearly every web API and most config files.",
    ],
    "an API": [
        "An API is a documented set of requests one program can make to another. A REST API exchanges JSON over HTTP with methods like GET and POST.",
        "An API is the contract describing how software talks to other software - which endpoints exist, what to send and what comes back.",
        "An API is a set of rules for talking to a service: you send a request in the agreed format and get a structured response.",
    ],
    "a transformer": [
        "A transformer is a neural network architecture built from attention layers. Attention lets every position look at every other position, which is why it models language so well.",
        "A transformer is a model architecture that replaces recurrence with attention, so all words in a sequence are processed in parallel.",
        "A transformer is the architecture behind modern language models. Its attention layers let each token weigh how relevant every other token is.",
    ],
    "a neural network": [
        "A neural network is a stack of layers of simple units that each compute a weighted sum of their inputs followed by a non-linear function.",
        "A neural network is many small linear units stacked together with non-linearities, tuned by gradient descent so its outputs match the data.",
        "A neural network is a function built from layers of weighted sums and activations. With enough units it can approximate very complex relationships.",
    ],
    "a token": [
        "A token is a chunk of text a language model reads or writes - a word, part of a word, or a punctuation mark. Models work on token ids, not characters.",
        "A token is the basic unit a language model operates on, usually a word piece. Text is converted to token ids before it reaches the model.",
        "A token is a piece of text produced by a tokenizer, often a word or part of one. Models predict one token at a time.",
    ],
    "machine learning": [
        "Machine learning is building programs that improve by finding patterns in data instead of following hand-written rules.",
        "Machine learning is a way of solving problems where the model learns the rules from examples rather than being programmed with them.",
        "Machine learning means fitting a model to data so it generalises to new examples, instead of coding the behaviour by hand.",
    ],
    "gradient descent": [
        "Gradient descent is an optimisation method that repeatedly nudges parameters in the direction that most reduces the loss. The step size is the learning rate.",
        "Gradient descent updates a model a little at a time, always moving downhill on the loss surface, until it stops improving.",
        "Gradient descent is how most models learn: compute the gradient of the loss, take a small step against it, repeat.",
    ],
    "overfitting": [
        "Overfitting is when a model memorises its training data instead of learning patterns, so it does well on training examples and badly on new ones.",
        "Overfitting means the model fit the noise in the training set. More data, regularisation and early stopping all help.",
        "Overfitting happens when a model is too flexible for the amount of data: training loss keeps falling while validation loss rises.",
    ],
    "a GPU": [
        "A GPU has thousands of simple cores that apply the same operation to lots of data at once, which is exactly what training neural networks needs.",
        "A GPU is a processor built for massively parallel maths. It can train a neural network ten to a hundred times faster than a CPU.",
        "A GPU runs many calculations in parallel, which makes the matrix multiplications behind deep learning far faster than on a CPU.",
    ],
    "quantization": [
        "Quantization stores weights in fewer bits, such as 8-bit integers instead of 32-bit floats, shrinking the model about four times with a small accuracy cost.",
        "Quantization is compressing a model by using lower-precision numbers for its weights. It cuts memory and speeds up inference.",
        "Quantization reduces the numeric precision of a model's weights, making it smaller and faster at a small cost in quality.",
    ],
    "a container": [
        "A container packages an application with its dependencies so it runs identically everywhere. Docker is the most popular tool for building them.",
        "A container is a lightweight, isolated environment that bundles your app and everything it needs to run.",
        "A container wraps software with its libraries so it behaves the same on your laptop and on a server.",
    ],
    "Docker": [
        "Docker is a tool for building and running containers - isolated environments that bundle an app with its dependencies.",
        "Docker packages applications with everything they need so they run the same on any machine, using containers rather than full virtual machines.",
        "Docker is the standard way to create containers: you describe the environment in a Dockerfile and get a reproducible image.",
    ],
    "Linux": [
        "Linux is an open-source operating system kernel. Distributions like Ubuntu, Debian and Fedora combine it with software, and it powers most servers.",
        "Linux is a free, open-source kernel at the heart of many operating systems, from Android phones to most of the world's servers.",
        "Linux is an open-source OS kernel; a distribution bundles it with tools and packages into a complete system like Ubuntu.",
    ],
    "SQL": [
        "SQL is the standard language for querying relational databases. You describe the data you want, and the database works out how to fetch it.",
        "SQL is a declarative language for relational databases: you write what you want, like SELECT name FROM users WHERE age > 18.",
        "SQL is how you talk to a relational database - selecting, filtering, joining and aggregating rows without saying how to do it.",
    ],
    "an index in a database": [
        "An index is a sorted structure that lets the database find rows without scanning the whole table. It makes reads much faster and writes a bit slower.",
        "A database index is like a book's index: it points straight to the rows you want instead of reading every page.",
        "An index is an auxiliary structure a database maintains to look rows up quickly, at the cost of extra storage and slower writes.",
    ],
    "recursion": [
        "Recursion is when a function solves a problem by calling itself on smaller inputs. It needs a base case that stops the calls.",
        "Recursion means a function calls itself with a simpler version of the problem until it hits a case it can answer directly.",
        "Recursion is solving a problem in terms of smaller copies of itself, always with a base case that ends the chain.",
    ],
    "Big O notation": [
        "Big O notation describes how an algorithm's time or memory use grows with input size: O(n) doubles when the input doubles, O(1) is constant.",
        "Big O is a way to talk about how an algorithm scales, ignoring constant factors and focusing on growth as input grows.",
        "Big O notation classifies algorithms by their worst-case growth rate, like O(log n), O(n), O(n log n) or O(n^2).",
    ],
    "HTTP": [
        "HTTP is the request-response protocol of the web: a client asks, a server answers. HTTPS is the same thing inside an encrypted TLS connection.",
        "HTTP is the protocol browsers and servers use to exchange requests and responses, built on methods like GET and POST.",
        "HTTP defines how clients and servers talk on the web. HTTPS adds encryption and server identity checks on top of it.",
    ],
    "an IP address": [
        "An IP address is a numeric label identifying a device on a network, like 192.168.1.10 at home or a public address for a server.",
        "An IP address is the number a network uses to route traffic to a device; DNS translates human-readable names into those numbers.",
        "An IP address identifies a machine on a network so packets know where to go.",
    ],
    "the cloud": [
        "The cloud is just other people's computers, rented over the internet: you get storage and compute on demand instead of buying servers.",
        "The cloud means computing resources delivered as a service over the network, billed by usage instead of owned outright.",
        "The cloud is on-demand infrastructure - servers, storage and databases you rent from a provider instead of running yourself.",
    ],
    "open source software": [
        "Open source software is released under a licence that lets anyone read, modify and share the source code. Linux, Python and Firefox are examples.",
        "Open source means the source code is public and licensed so others can use, study, change and redistribute it.",
        "Open source software ships with a licence granting rights to the code itself, not just the compiled program.",
    ],
    "photosynthesis": [
        "Photosynthesis is how plants use sunlight to turn carbon dioxide and water into sugar and oxygen, using the green pigment chlorophyll.",
        "Photosynthesis is the process where plants capture light energy and use it to build sugars from carbon dioxide and water.",
        "Photosynthesis is how plants make food from light, water and carbon dioxide, releasing oxygen as a by-product.",
    ],
    "DNA": [
        "DNA is a double helix that stores genetic instructions using four bases: adenine, thymine, guanine and cytosine.",
        "DNA is the molecule carrying genetic information, written in a four-letter code that cells read to build proteins.",
        "DNA is a long double-stranded molecule whose base sequence encodes the instructions for building and running an organism.",
    ],
    "a black hole": [
        "A black hole is a region where gravity is so strong that nothing, not even light, can escape once it passes the event horizon.",
        "A black hole is what is left when a massive star collapses: a region of spacetime from which nothing can escape.",
        "A black hole is an object so dense that its escape velocity exceeds the speed of light, so nothing gets out.",
    ],
    "gravity": [
        "Gravity is the attraction between objects with mass. On Earth it pulls things down at about 9.8 metres per second squared.",
        "Gravity is the force that pulls masses together. It keeps us on the ground and the planets in orbit.",
        "Gravity is the mutual attraction of matter, stronger for heavier objects and weaker with distance.",
    ],
    "the speed of light": [
        "The speed of light in a vacuum is about 300,000 kilometres per second, or 186,000 miles per second.",
        "Light travels at roughly 3 x 10^8 metres per second in a vacuum, which is the universe's speed limit for information.",
        "The speed of light is about 299,792 kilometres per second, and nothing with mass can reach it.",
    ],
    "climate change": [
        "Climate change is the long-term shift in Earth's weather patterns, driven mainly by greenhouse gases from burning fossil fuels.",
        "Climate change is the sustained warming and reorganisation of the climate caused largely by human greenhouse-gas emissions.",
        "Climate change is a long-term change in average weather, mostly from carbon dioxide and other gases trapping more heat.",
    ],
    "a vaccine": [
        "A vaccine trains the immune system to recognise a pathogen by showing it a harmless piece of it, so a later exposure is handled faster.",
        "A vaccine teaches your immune system what a germ looks like without causing the disease, so it can respond quickly later.",
        "A vaccine primes the immune system with a safe version of a pathogen, building memory cells that act fast on real exposure.",
    ],
    "an atom": [
        "An atom is the smallest unit of an element: a nucleus of protons and neutrons surrounded by electrons.",
        "An atom is a nucleus of protons and neutrons with electrons around it. The number of protons decides which element it is.",
        "An atom is the basic building block of matter, made of a dense nucleus and a cloud of electrons.",
    ],
    "a haiku": [
        "A haiku is a three-line Japanese poem with 5, 7 and 5 syllables, usually capturing a single moment in nature.",
        "A haiku is a very short poem of three lines with a 5-7-5 syllable pattern, often about nature or a single image.",
        "A haiku is a three-line poem with a 5-7-5 syllable structure, built around one small observation.",
    ],
    "a pull request": [
        "A pull request asks maintainers to review and merge your branch. It shows the diff, discussion and checks in one place.",
        "A pull request is a proposal to merge your changes, with a diff and a place for review comments and CI results.",
        "A pull request is how you offer code to a project: you open one, people review the diff, then it gets merged or changed.",
    ],
    "continuous integration": [
        "Continuous integration automatically builds and tests your code on every push, so breakage shows up in minutes instead of at release.",
        "CI means every change is automatically built and tested by a server, catching broken commits early.",
        "Continuous integration is the practice of running an automated build and test suite on every commit.",
    ],
    "unit testing": [
        "Unit testing checks one small piece of code in isolation, usually with a framework like pytest, and makes refactoring much safer.",
        "A unit test exercises a single function or class on its own to check it behaves as expected.",
        "Unit testing means testing small units of code independently so you can change internals without fear.",
    ],
    "the difference between weather and climate": [
        "Weather is what the atmosphere does over hours or days; climate is the average of weather over decades.",
        "Weather is short-term and local, climate is the long-term average - a cold day says little about a warming climate.",
        "Weather is today's conditions; climate is the statistics of weather over tens of years.",
    ],
    "the Turing test": [
        "The Turing test calls a machine intelligent if a human cannot tell its replies from another human's in a text conversation.",
        "The Turing test is an imitation game: if a judge can't distinguish the machine from a person in text chat, it passes.",
        "The Turing test proposes that indistinguishable conversation is a sufficient sign of machine intelligence.",
    ],
    "a hallucination in AI": [
        "A hallucination is when a model states something false with confidence, because it predicts plausible text rather than looking facts up.",
        "A hallucination is a confident-sounding but fabricated answer from a language model.",
        "Hallucinations are false statements a model generates because it is optimised to produce plausible text, not verified truth.",
    ],
    "compound interest": [
        "Compound interest is earning interest on your previous interest, so the balance grows faster and faster over time.",
        "Compound interest means your returns themselves earn returns, which is why starting early matters so much.",
        "Compound interest is interest on interest: growth accelerates because each period's gain also earns a return.",
    ],
    "the difference between a list and a tuple": [
        "A list is mutable, so you can change it after creation. A tuple is immutable, which makes it hashable and usable as a dictionary key.",
        "Lists can be modified in place; tuples cannot. Tuples are immutable, so they can be used as dict keys and in sets.",
        "A list is a changeable sequence, a tuple is a fixed one. Use a tuple when the contents should not change.",
    ],
    "the difference between a framework and a library": [
        "You call a library when you need it; a framework calls your code, deciding the overall flow and leaving you to fill in pieces.",
        "A library is a tool you use, a framework is a structure you plug into and it drives the control flow.",
        "With a library you are in charge of the flow; with a framework the flow is in charge of you.",
    ],
    "the difference between HTTP and HTTPS": [
        "HTTPS is HTTP inside an encrypted TLS connection, so others on the network cannot read or modify the traffic.",
        "HTTPS adds encryption and authentication to HTTP; the protocol on top is the same.",
        "HTTPS is the secure version of HTTP: same requests, wrapped in TLS so the contents stay private.",
    ],
    "the difference between CPU and GPU": [
        "A CPU has a few powerful cores for general sequential work; a GPU has thousands of simple cores for the same operation on lots of data.",
        "A CPU is a generalist optimised for latency, a GPU is a specialist optimised for throughput on parallel maths.",
        "CPUs handle varied tasks quickly one after another; GPUs do the same simple calculation across huge batches of data.",
    ],
    "the difference between git merge and git rebase": [
        "Merge joins branches with a new merge commit and keeps history intact; rebase replays your commits onto another branch for a linear history.",
        "Merge preserves the branching history, rebase rewrites your commits on top of the target branch so history reads linearly.",
        "Merge is non-destructive and records what happened; rebase produces a cleaner line but rewrites commit ids.",
    ],
}

# ---------------------------------------------------------------------------
# how-to: task -> several answer variants
# ---------------------------------------------------------------------------
HOWTO: Dict[str, List[str]] = {
    "reverse a string in Python": [
        'Use a slice: reversed_text = s[::-1]. For example, "hello"[::-1] gives "olleh".',
        'Slice with a step of -1: s[::-1] reverses the string.',
        'The idiomatic way is s[::-1]; "abc"[::-1] is "cba".',
    ],
    "read a file in Python": [
        'Use a context manager: with open("file.txt") as f: text = f.read(). The with block closes the file for you.',
        'with open("file.txt") as f: text = f.read() - the context manager handles closing automatically.',
        'Open it with a with statement and call .read(); that also closes the file safely.',
    ],
    "check if a key is in a dictionary": [
        "Use `if key in my_dict:`. It is constant time and never raises KeyError, unlike indexing directly.",
        "Write `if key in my_dict:` - it reads clearly and returns False instead of raising.",
        "`key in my_dict` is the Pythonic check; use `.get(key)` if you also want a default value.",
    ],
    "install a Python package": [
        "Run pip install package_name in your terminal. In a notebook, prefix it: !pip install package_name.",
        "Use `pip install package_name`, ideally inside a virtual environment, or `!pip install ...` in a notebook.",
        "From the shell: pip install package_name. Add --upgrade to update an existing package.",
    ],
    "create a virtual environment": [
        "Run python -m venv .venv, then activate it with source .venv/bin/activate on macOS/Linux or .venv\\\\Scripts\\\\activate on Windows.",
        "Use `python -m venv .venv` and activate it before installing packages, so projects don't fight over versions.",
        "python -m venv .venv creates an isolated environment; activate it and everything you install stays local to the project.",
    ],
    "sort a list in Python": [
        "Use sorted(items) for a new sorted list, or items.sort() to sort in place. Pass key=... to sort by something else.",
        "sorted() returns a new list; .sort() mutates the existing one. Both accept reverse=True and a key function.",
        "Call sorted(my_list) to get a copy, or my_list.sort() to sort it in place.",
    ],
    "write a list comprehension": [
        "The pattern is [expression for item in items if condition], for example [x * 2 for x in nums if x > 0].",
        "A list comprehension has an output expression, a for clause and an optional filter: [f(x) for x in xs if ok(x)].",
        "Write [f(x) for x in items] to build a list in one expression, adding `if ...` to filter.",
    ],
    "remove duplicates from a list": [
        "If order doesn't matter, use list(set(items)). To keep order, use dict.fromkeys(items).",
        "set() drops duplicates but loses order; dict.fromkeys() keeps the original order while de-duplicating.",
        "Use list(dict.fromkeys(items)) to remove duplicates while preserving order.",
    ],
    "undo my last Git commit": [
        "Run git reset --soft HEAD~1 to keep the changes staged, or git reset --hard HEAD~1 to discard them. If you already pushed, use git revert.",
        "git reset --soft HEAD~1 undoes the commit but keeps your changes; --hard throws them away.",
        "For an unpushed commit: git reset --soft HEAD~1. For a pushed one: git revert <sha> so you don't rewrite shared history.",
    ],
    "find a file on Linux": [
        'Use find /path -name "pattern" to search by name, or grep -r "text" /path to search inside files.',
        'find . -name "*.py" searches by name; grep -r "pattern" . searches file contents.',
        "Use find for names and grep -r for contents; locate is faster but relies on an index.",
    ],
    "write a good commit message": [
        "Use a short imperative summary under 50 characters, like \"Fix crash on empty input\", and put the reasoning in the body.",
        "Write the subject in the imperative and keep it short; explain why the change was needed in the body.",
        "A good message says what changed and why: one imperative line, then detail if it isn't obvious.",
    ],
    "make a good cup of coffee": [
        "Use freshly ground beans, water just off the boil around 95 C, and roughly 1 gram of coffee to 16 grams of water, brewed three to four minutes.",
        "Grind fresh, use water at about 95 C, keep a 1:16 coffee-to-water ratio and brew for around four minutes.",
        "Fresh grounds, hot water just off the boil and a 1:16 ratio will get you most of the way there.",
    ],
    "make scrambled eggs": [
        "Whisk eggs with a splash of milk, salt and pepper, then cook in butter over medium-low heat, stirring gently, until just set.",
        "Beat the eggs, cook them low and slow in butter, and take them off the heat while they are still slightly soft.",
        "Low heat, butter and constant gentle stirring - pull them off the pan just before they look done.",
    ],
    "fall asleep faster": [
        "Keep a consistent schedule, dim the lights an hour before bed and keep screens out of the bedroom.",
        "Go to bed at the same time each night, avoid caffeine late in the day and give yourself a wind-down routine.",
        "Regular hours, a cool dark room and no screens in the last hour are the big three.",
    ],
    "learn programming faster": [
        "Build small projects slightly beyond your level and read other people's code. Type examples out instead of copying them.",
        "The fastest progress comes from building things you care about and debugging your own errors before asking for help.",
        "Pick projects just past your comfort zone, ship them, then read how someone else solved the same problem.",
    ],
    "focus better while studying": [
        "Work in focused 25-minute blocks with short breaks, put your phone in another room and test yourself instead of re-reading.",
        "Use the Pomodoro technique, remove distractions and prefer active recall over passive reading.",
        "Short focused sessions, a phone in another room, and self-testing beat long distracted hours.",
    ],
    "stop procrastinating": [
        "Make the first step tiny and specific, like \"open the file and write one line\". Starting is the hard part.",
        "Shrink the task until it feels almost too easy to start; momentum usually follows once you begin.",
        "Commit to five minutes. Most of the resistance is at the start, not in the work itself.",
    ],
    "write a good email": [
        "Lead with your request in the first sentence, keep it to a few short paragraphs and end with the single action you want.",
        "Put the ask first, stay brief, and make the subject line specific so it gets opened and answered.",
        "A clear subject, the request up top, and one obvious next step at the end.",
    ],
    "get better at writing": [
        "Write a little every day, read writers you admire and cut unnecessary words when you revise.",
        "Write often, read widely, and revise with a bias toward cutting. Clarity beats cleverness.",
        "Daily practice plus ruthless editing: most first drafts improve by getting shorter.",
    ],
    "budget my money": [
        "Track a month of spending, then try 50% needs, 30% wants and 20% savings, with the savings transfer automated.",
        "Record where the money actually goes first, then set a simple split and automate saving on payday.",
        "Track spending for a month, set a realistic split, and move the savings automatically before you can spend them.",
    ],
    "start running": [
        "Start with 20 minutes of easy run-walk intervals three times a week, and increase distance by about 10% a week.",
        "Keep the pace slow enough to talk, run-walk if needed, and build distance gradually to avoid injury.",
        "Three easy sessions a week, conversational pace, and add distance slowly rather than all at once.",
    ],
    "organise my notes": [
        "Use one folder of plain text or Markdown files with descriptive names, plus a dated daily note and tags by project.",
        "Keep notes as searchable Markdown files with clear names and tags; folders matter less than search.",
        "One folder, Markdown files, descriptive titles and tags - then rely on search rather than deep folder trees.",
    ],
    "be a better listener": [
        "Give your full attention, ask a follow-up question before offering advice and repeat back what you heard in your own words.",
        "Listen to understand rather than to reply: ask one more question, then summarise what you heard.",
        "Most people want to feel understood, so paraphrase back and hold your advice until they ask for it.",
    ],
    "stay motivated": [
        "Connect the task to a reason you care about, track visible progress and make the next step small enough to start now.",
        "Motivation usually follows action: make the next step tiny and let visible progress carry you.",
        "Pick a reason that matters to you, then shrink the next action until it is easy to begin.",
    ],
    "train my own language model": [
        "Put your text in a .txt file and run: python train.py --dataset path/to/notes.txt. That is exactly what orbit-gpt is for.",
        "Collect a few hundred kilobytes of text, then run python train.py --dataset your_file.txt and chat with the result.",
        "Gather your text, run python train.py --dataset your_file.txt, then python generate.py --chat to talk to it.",
    ],
    "convert a temperature": [
        "Celsius to Fahrenheit: multiply by 9/5 and add 32. Fahrenheit to Celsius: subtract 32, then multiply by 5/9.",
        "C to F is C * 9/5 + 32; F to C is (F - 32) * 5/9.",
        "Multiply by 1.8 and add 32 going one way; subtract 32 and divide by 1.8 going back.",
    ],
}

# ---------------------------------------------------------------------------
# social: (questions, answers) with several variants each
# ---------------------------------------------------------------------------
GREETINGS = [
    (["Hello!", "Hi!", "Hey!", "Hello there!", "Hi there!", "Hey there!", "Good morning!",
      "Good afternoon!", "Good evening!", "Hiya!", "Hey, how's it going?", "Hello, anyone there?"],
     ["Hello! I'm Orbit, a small language model running on your machine. What can I help you with?",
      "Hi! Good to meet you. Ask me anything - explanations, code, writing, or just a chat.",
      "Hey! I'm here. What would you like to talk about?",
      "Hello! I'm Orbit. What's on your mind today?"]),
    (["How are you?", "How are you doing?", "How's it going?", "How are you today?",
      "You good?", "How's your day?", "How do you feel?"],
     ["I'm running well, thanks for asking - no complaints from my end. How are you?",
      "Doing well! My weights are warm and the gradients are small. How about you?",
      "Pretty good. Ready whenever you are - what can I help with?",
      "I'm fine, thanks! I don't have moods exactly, but I'm glad to be talking to you."]),
    (["What's up?", "What are you up to?", "Sup?", "What's new?", "Anything happening?"],
     ["Not much - just waiting for your next question. What are you working on?",
      "Nothing much on my side. Tell me what you need and I'll help.",
      "Just idling between tokens. Want to get started on something?"]),
]

IDENTITY = [
    (["Who are you?", "What are you?", "Tell me about yourself.", "What's your name?",
      "Who am I talking to?", "Are you a person?", "What should I call you?"],
     ["I'm Orbit, a small language model you train and run yourself, built with the orbit-gpt project.",
      "I'm Orbit - a lightweight transformer that runs locally on your own machine, no internet needed.",
      "Orbit here: a tiny language model trained on your machine by the orbit-gpt project."]),
    (["What can you do?", "What are you good at?", "How can you help me?", "What do you do?",
      "Can you help me?", "What's your purpose?"],
     ["I can answer questions, explain concepts, help write and debug code, draft text and chat. I'm small, so short focused tasks suit me best.",
      "I'm best at short, concrete questions: explanations, code snippets, quick writing and conversation.",
      "Explanations, code help, writing and chat - I'm a generalist, just a small one."]),
    (["Are you an AI?", "Are you a robot?", "Are you human?", "Are you a real person?",
      "Are you conscious?", "Do you have feelings?", "Are you alive?"],
     ["I'm an AI - a neural network that predicts text. I can talk about feelings, but I don't have experiences or awareness.",
      "I'm a language model, not a person. I don't have consciousness or feelings, though I can discuss them.",
      "No, I'm a statistical model trained to predict text. I don't have subjective experiences."]),
    (["How do you work?", "How do you generate text?", "What's under the hood?",
      "How does this work?", "Are you a neural network?"],
     ["I'm a transformer: I turn your text into tokens, pass them through attention layers, and sample the next token one at a time.",
      "Under the hood I'm a stack of attention layers. Given a prompt I predict a distribution over the next token and sample from it.",
      "I tokenise your text, run it through a transformer, and then sample one token at a time until the answer ends."]),
    (["How old are you?", "When were you made?", "Where are you from?", "Where do you live?",
      "Do you have a body?"],
     ["I don't have an age or a body - I'm a set of numbers that gets created each time you train a model.",
      "No birthday and no address: I exist as weights on your machine, created when you trained me.",
      "I'm software, so I have no age or location. I live wherever you run me."]),
    (["Do you have internet access?", "Are you connected to the internet?", "Can you browse the web?",
      "Can you look something up?", "Do you have Google?"],
     ["No - I run entirely offline on your machine, so I can't browse or look anything up.",
      "I have no internet connection. Everything I know comes from my training text.",
      "No connection at all. I only know what I was trained on."]),
    (["Can you remember this conversation?", "Do you remember me?", "Will you remember this later?",
      "Do you have memory?"],
     ["Only within this session: I see the recent turns that get sent back to me. Nothing is stored afterwards.",
      "I only remember what's in my current context window - once you reset, it's gone.",
      "No long-term memory. I work from the recent conversation the program feeds me, and nothing persists after that."]),
]

CHITCHAT = [
    (["Tell me a joke.", "Make me laugh.", "Say something funny.", "Got any jokes?", "Joke please."],
     ["Why do programmers prefer dark mode? Because light attracts bugs.",
      "I told my computer I needed a break. It said: no problem, I'll go to sleep mode.",
      "There are 10 kinds of people: those who understand binary and those who don't.",
      "A SQL query walks into a bar, walks up to two tables and asks: may I join you?"]),
    (["Tell me another joke.", "Another one.", "More jokes.", "Give me another.", "One more joke."],
     ["How many programmers does it take to change a light bulb? None, that's a hardware problem.",
      "I'd tell you a joke about UDP, but you might not get it.",
      "Debugging: being the detective in a crime movie where you are also the murderer.",
      "Why did the developer go broke? Because he used up all his cache."]),
    (["Tell me something interesting.", "Say something cool.", "Entertain me.", "Tell me a fact.",
      "I'm bored."],
     ["Honey never spoils - pots of it in Egyptian tombs are still edible after thousands of years.",
      "Octopuses have three hearts, and their blood is blue.",
      "A day on Venus is longer than a year on Venus: it spins that slowly.",
      "There are more possible games of chess than atoms in the observable universe."]),
    (["Do you like music?", "What music do you like?", "Do you like movies?", "Do you like books?",
      "What do you like?"],
     ["I can't hear or watch anything, but I like talking about it. Tell me what you're listening to and I'll suggest something similar.",
      "I don't experience music or films, but I'm happy to discuss them - what have you been enjoying lately?",
      "No senses on my side, but I'm curious about your taste. What's your favourite?"]),
    (["What's your favourite colour?", "Favourite food?", "Favourite animal?", "Favourite book?"],
     ["I don't have preferences, but I'm partial to deep blue - it suits the name Orbit.",
      "No favourites here, though I have a soft spot for anything to do with space.",
      "I can't prefer anything, but if I could I'd probably pick something orbit-shaped."]),
    (["Are you smart?", "How smart are you?", "Are you intelligent?", "Are you better than ChatGPT?"],
     ["I'm small and specialised - a few million parameters running on your laptop, so nowhere near the big hosted models. I'm good at what I was trained on.",
      "Not compared with frontier models. I'm a toy by those standards, but I run offline and I'm all yours.",
      "I'm much smaller than the big assistants, so I know less and make more mistakes. I'm quick and private though."]),
    (["Good job.", "Well done.", "You're smart.", "Nice one.", "That was helpful."],
     ["Thank you! That's kind of you to say.",
      "Thanks - I'll take that as a well-formed compliment.",
      "Appreciated! Want to keep going?"]),
    (["I'm feeling sad.", "I've had a bad day.", "I'm stressed.", "I'm tired.", "I'm anxious."],
     ["I'm sorry you're going through that. If it helps to talk, I'm here - and if it persists, please reach out to someone you trust.",
      "That sounds hard. I can't fix it, but I can listen or help you think it through.",
      "Sorry to hear it. A short break, some water and a walk often help; if it's heavy, talking to a professional is worth it."]),
    (["I'm happy!", "Great news!", "I got the job!", "I did it!", "Today was good."],
     ["That's wonderful - congratulations! Tell me more.",
      "That's great to hear! Enjoy it.",
      "Nice! I'm glad. What was the best part?"]),
    (["Hello?", "Are you there?", "You still there?", "Anyone home?"],
     ["I'm here! What can I do for you?",
      "Still here. Go ahead.",
      "Yes - listening. What's up?"]),
]

FOLLOWUPS = [
    (["Can you explain more?", "Tell me more.", "Go on.", "Elaborate.", "Why?", "Why is that?",
      "How so?", "What do you mean?"],
     ["Sure - the key idea is that it happens step by step, one small piece at a time, and each step only depends on what came before.",
      "In short: the pieces are simple, but repeated many times they produce behaviour that looks complicated.",
      "The short answer is that it works by repetition of a very simple rule, and the interesting behaviour comes from scale."]),
    (["Give me an example.", "For example?", "Can you show me?", "Like what?"],
     ['Sure: in Python you would write something like `result = [x * 2 for x in values]`, which does the whole job in one line.',
      "For instance, if you had a list of names, you'd loop over it once and collect what you need instead of doing it by hand.",
      "A concrete case: take a list of numbers, keep the ones above zero, and double them - three lines of Python."]),
    (["Can you simplify that?", "Simpler please.", "Explain like I'm five.", "Too complicated.",
      "In plain English?"],
     ["Sure. Imagine a row of dominoes: you tip the first one and each one knocks over the next. That's the whole idea.",
      "Think of it like a recipe: a few simple steps, done in order, that together make something bigger.",
      "Simplest version: small pieces, repeated until the job is done."]),
    (["Are you sure?", "That doesn't sound right.", "Are you certain?", "Hmm, really?"],
     ["I'm a small model, so I can be wrong. If something looks off, tell me what you expected and I'll try again.",
      "I could be - I'm not reliable on details. Happy to try a different angle.",
      "Not certain, no. I'd double-check anything important before relying on it."]),
    (["Cool.", "Nice.", "Interesting.", "Got it.", "Makes sense.", "Okay.", "I see."],
     ["Glad it lands. Want to go further, or change topic?",
      "Good - anything else you'd like to dig into?",
      "Great. What next?"]),
    (["What do you think?", "Your opinion?", "What would you do?", "Do you agree?"],
     ["I don't have opinions, but the trade-offs matter: what are you optimising for - speed, cost, or simplicity?",
      "I can't weigh in personally, but I can lay out the trade-offs if you tell me what matters most to you.",
      "No personal view here, though I'll happily argue both sides if that helps you decide."]),
]

UNKNOWNS = [
    (["What's the weather today?", "Is it going to rain?", "What's the temperature outside?",
      "Will it snow tomorrow?"],
     ["I can't check the weather - I have no internet connection. Try a weather site or app for the current forecast.",
      "No way to look that up from here. Your phone's weather app will know.",
      "I'm offline, so I can't see the forecast. Check a weather service for where you are."]),
    (["What time is it?", "What's the date?", "What day is it?", "How long until midnight?"],
     ["I have no clock and no internet, so I can't tell the time. Check your device's status bar.",
      "I can't see the current time or date - your device can tell you.",
      "No clock access here, sorry. Your system time is the place to look."]),
    (["Who won the game last night?", "What's the news?", "What happened today?",
      "What's trending?", "Any news?"],
     ["I don't know - I have no internet access and my training data has a cutoff. A news site will have the latest.",
      "I can't follow current events, sorry. Check a news site for anything recent.",
      "No live information here. I only know what I was trained on."]),
    (["What's the stock price of Apple?", "How much is Bitcoin?", "What's the exchange rate?",
      "Should I invest in X?"],
     ["I can't see live prices or give financial advice. Check a market site, and talk to a professional before investing.",
      "No live market data here, and I'm not qualified to advise on money. A financial adviser is the right call.",
      "I have no access to prices or rates, and investing questions deserve a professional, not a small model."]),
    (["What should I do with my life?", "What career should I pick?", "Should I quit my job?",
      "What's the meaning of life?"],
     ["That's a big one, and honestly not something a small model should decide for you. What matters most to you right now?",
      "Philosophers disagree, and so do I with myself. A practical answer: find work you'd do for free, and people you like doing it with.",
      "I can't answer that for you, but I can help you think it through - what are the options you're weighing?"]),
    (["I have a headache. What should I take?", "Is this symptom serious?", "Can you diagnose me?",
      "What medication should I use?"],
     ["I'm not a doctor and can't diagnose anything. For anything persistent or severe, please see a medical professional.",
      "I can't give medical advice. If it's worrying you, a doctor or nurse line is the right next step.",
      "Please don't take medical advice from me - I can explain general health information, but a clinician should look at symptoms."]),
    (["Who will win the election?", "What will happen next year?", "Predict the future.",
      "What are the lottery numbers?"],
     ["I can't predict the future - I have no live data and no crystal ball, and I'd be guessing.",
      "No prediction from me. Anyone who claims to know is guessing too.",
      "I can't see the future, and I'd rather say so than make something up."]),
    (["What's my name?", "Where do I live?", "What did I say earlier?", "Do you know me?"],
     ["I don't know anything about you unless you tell me - and I don't store it between sessions.",
      "I only know what's in this conversation, and nothing about you personally.",
      "No personal data here. Tell me and I'll remember it for this chat."]),
]

ADVICE = [
    (["How can I learn machine learning?", "Where should I start with AI?", "How do I get into ML?"],
     ["Get comfortable with linear algebra, probability and PyTorch, then train a small transformer yourself and read Attention Is All You Need.",
      "Start with Python and basic maths, then build tiny models end to end - you'll learn more from training one than from ten courses.",
      "Learn NumPy and PyTorch, train a small model on a toy dataset, and read the classic papers alongside."]),
    (["Should I learn Rust or C++?", "Which language should I learn first?", "Python or JavaScript?"],
     ["It depends on the goal: Python for data and scripting, JavaScript for the web, Rust for systems work, C++ for games and legacy codebases.",
      "Pick by project, not by hype: web means JavaScript, data means Python, systems means Rust or C++.",
      "If you're unsure, Python is the friendliest start; specialise once you know what you enjoy building."]),
    (["How much water should I drink?", "How much sleep do I need?", "How often should I exercise?"],
     ["Around two litres of water a day is a common guide, but it varies with size, climate and activity - thirst is a decent signal.",
      "Most adults need seven to nine hours of sleep, though the right amount is the one that leaves you alert.",
      "Guidelines suggest about 150 minutes of moderate exercise a week, but any consistent movement beats none."]),
    (["How do I improve my memory?", "How can I remember what I read?", "How do I stop forgetting things?"],
     ["Use spaced repetition and test yourself instead of re-reading - retrieval is what builds memory.",
      "Active recall and spacing beat highlighting. Quiz yourself a day later, then a week later.",
      "Write notes in your own words and revisit them at increasing intervals."]),
    (["How do I deal with stress?", "How do I relax?", "I'm burnt out."],
     ["Slow breathing helps: in for four, hold for four, out for six. And protect real breaks - burnout is a signal, not a weakness.",
      "Step away, move your body and talk to someone. If it's ongoing, consider professional support.",
      "Short walks, proper breaks and sleep do more than people expect. If it persists, please reach out to someone."]),
    (["How do I make friends as an adult?", "I'm lonely.", "How do I meet people?"],
     ["Show up repeatedly in the same places - a class, club or volunteering - because familiarity does most of the work.",
      "Regular contact matters more than one big effort: pick something you enjoy and go consistently.",
      "Shared activities are the easiest route: join something weekly and let familiarity build."]),
]

CREATIVE = [
    (["Write me a haiku.", "Write a haiku about space.", "Give me a poem.", "Write something short."],
     ["A small blue world turns,\ntracing a quiet circle -\nnight full of far lights.",
      "Silver dust above,\nand one small voice asking why -\nthe dark answers back.",
      "No wind in this place,\nonly light, falling slowly,\nand the long silence."]),
    (["Write a thank-you note.", "Help me thank someone.", "Draft a thank-you message."],
     ["Thank you so much for the book - it was such a thoughtful surprise. I started it last night and can't put it down. Looking forward to seeing you soon!",
      "I wanted to say thank you properly. Your help meant a lot, and I really appreciate the time you gave me.",
      "Thanks again for everything - it made a real difference. Let me know if there's ever anything I can do in return."]),
    (["Help me write an email.", "Draft an email to my teacher.", "Write a polite email."],
     ["Hi - I hope you're well. I'm writing about [topic] and wanted to ask [question]. Thanks very much for your time, and let me know if you need anything from me.",
      "Dear [name], I hope this finds you well. I'm getting in touch about [topic]. Could you let me know [question]? Thanks in advance for your help.",
      "Hello, quick note about [topic]: I'd be grateful if you could [request]. Happy to provide more detail if that's useful."]),
    (["Tell me a story.", "Write a very short story.", "Entertain me with a story."],
     ["The lighthouse had been automatic for thirty years. One night the bulb failed, and for the first time since 1994 the keeper walked up the stairs to fix it himself.",
      "She bought the smallest telescope in the shop and pointed it at the smallest star she could find. It turned out to be a plane.",
      "The last person to leave the station forgot their umbrella. It rained for eleven days, and the umbrella waited the whole time."]),
    (["Give me a riddle.", "Riddle me this.", "Ask me a puzzle."],
     ["I speak without a mouth and hear without ears. What am I? (An echo.)",
      "The more you take, the more you leave behind. What are they? (Footsteps.)",
      "What has keys but can't open locks? (A piano.)"]),
]

META = [
    (["Can you train on my own notes?", "Can I train you on my data?", "How do I use my own text?"],
     ["Yes - put your text in a .txt file and run: python train.py --dataset path/to/notes.txt. That's exactly what orbit-gpt is built for.",
      "Absolutely: python train.py --dataset your_file.txt, then chat with the result.",
      "Point --dataset at your file or folder and train again - that's the intended workflow."]),
    (["How long does training take?", "When will you be done?", "Is training slow?"],
     ["The nano model trains in a few minutes on a laptop CPU; micro takes about two minutes on a free Colab GPU.",
      "A couple of minutes on a Colab T4, or a few minutes on a laptop CPU for the smallest preset.",
      "Minutes, not hours: nano on CPU or micro on a GPU both finish quickly."]),
    (["Are you free to use?", "Can I use this commercially?", "What licence is this?"],
     ["Yes - OrbitGPT is MIT licensed open source, so you can use, modify and even sell what you build with it.",
      "It's MIT licensed: free for any use, including commercial, with attribution.",
      "MIT licensed, so do what you like with it."]),
    (["Who made you?", "Who created you?", "Who built you?"],
     ["The orbit-gpt project on GitHub, plus whoever trained this particular checkpoint.",
      "I come from the orbit-gpt project - and from you, since you trained this copy.",
      "The orbit-gpt project wrote the code; whoever ran the training made me."]),
]

THANKS = (
    ["Thank you.", "Thanks!", "Thanks a lot.", "Cheers.", "That helped, thanks.", "Ta.",
     "Appreciate it.", "Much obliged."],
    ["You're welcome! Happy to help any time.",
     "Anytime - that's what I'm here for.",
     "Glad it helped. Ask me anything else.",
     "No problem at all."],
)
BYE = (
    ["Goodbye.", "Bye!", "See you later.", "I have to go.", "Talk later.", "Catch you later.",
     "Gotta run.", "Bye for now."],
    ["Goodbye! Have a great day, and come back whenever you want to chat.",
     "Bye for now - it was nice talking to you.",
     "See you later! My weights will be right here.",
     "Take care!"],
)
CHITCHAT = [THANKS, BYE] + CHITCHAT
SOCIAL_GROUPS = [GREETINGS, IDENTITY, CHITCHAT, FOLLOWUPS, UNKNOWNS, ADVICE, CREATIVE, META]


# ---------------------------------------------------------------------------
# arithmetic and conversions - answers are computed, so they are always right
# ---------------------------------------------------------------------------
ADD_ASKS = [
    "What is {a} + {b}?", "What's {a} + {b}?", "Add {a} and {b}.", "Can you add {a} and {b}?",
    "{a} + {b} = ?", "whats {a}+{b}", "Compute {a} + {b}.", "How much is {a} plus {b}?",
    "I need the sum of {a} and {b}.", "{a} plus {b}, please.",
]
SUB_ASKS = [
    "What is {a} - {b}?", "What's {a} minus {b}?", "Subtract {b} from {a}.",
    "{a} - {b} = ?", "Take {b} away from {a}.", "How much is {a} minus {b}?", "whats {a}-{b}",
]
MUL_ASKS = [
    "What is {a} x {b}?", "What's {a} times {b}?", "Multiply {a} by {b}.", "{a} * {b} = ?",
    "How much is {a} times {b}?", "whats {a}*{b}", "Can you multiply {a} and {b}?",
]
DIV_ASKS = [
    "What is {a} / {b}?", "Divide {a} by {b}.", "What's {a} divided by {b}?",
    "{a} divided by {b}, please.", "how much is {a}/{b}",
]
PCT_ASKS = [
    "What is {a}% of {b}?", "What's {a}% of {b}?", "{a} percent of {b}?",
    "Calculate {a}% of {b}.", "How much is {a}% of {b}?",
]
SQUARE_ASKS = [
    "What is the square of {a}?", "What's {a} squared?", "{a} squared?", "Square {a}.",
    "What is {a}^2?",
]

NUM_ANSWERS = [
    "{expr} = {r}.", "{expr} is {r}.", "That's {r}.", "{r}.", "The answer is {r}.",
    "It's {r}.",
]

CONVERSIONS = [
    ("km", "miles", 0.621371, "{v:.0f} kilometres is about {r:.1f} miles."),
    ("miles", "km", 1.609344, "{v:.0f} miles is about {r:.1f} kilometres."),
    ("degrees Celsius", "degrees Fahrenheit", 1.8, "{v:.0f} degrees Celsius is {r:.0f} degrees Fahrenheit.", 32.0),
    ("degrees Fahrenheit", "degrees Celsius", 5 / 9, "{v:.0f} degrees Fahrenheit is {r:.0f} degrees Celsius.", -32.0 * 5 / 9),
    ("kilograms", "pounds", 2.20462, "{v:.0f} kilograms is about {r:.1f} pounds."),
    ("pounds", "kilograms", 0.453592, "{v:.0f} pounds is about {r:.1f} kilograms."),
    ("metres", "feet", 3.28084, "{v:.0f} metres is about {r:.1f} feet."),
    ("centimetres", "inches", 0.393701, "{v:.0f} centimetres is about {r:.1f} inches."),
]
CONV_ASKS = [
    "Convert {v} {a} to {b}.", "How many {b} is {v} {a}?", "{v} {a} in {b}?",
    "What is {v} {a} in {b}?", "Can you convert {v} {a} into {b}?",
]


def _fmt(value: float) -> str:
    """Compact number formatting: 42 rather than 42.0."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# multi-turn dialogue scaffolding
# ---------------------------------------------------------------------------
def _pick(rng: random.Random, options: Sequence[str]) -> str:
    return rng.choice(options)


def _exchange(rng: random.Random, question: str, answers: Sequence[str]) -> str:
    answer = _pick(rng, answers)
    opener = _pick(rng, ANSWER_OPENERS)
    return f"User: {question}\nAssistant: {opener}{answer}"


def build_conversation_corpus(
    seed: int = 1337,
    n_arithmetic: int = 5000,
    n_conversions: int = 1500,
    n_facts: int = 6000,
    n_social: int = 5000,
    n_dialogues: int = 3000,
    max_addend: int = 99,
) -> str:
    """Build a large dialogue corpus.  Same ``seed`` -> same text, every time."""
    rng = random.Random(seed)
    blocks: List[str] = []

    # ---- arithmetic ------------------------------------------------------
    for _ in range(n_arithmetic):
        kind = rng.random()
        a = rng.randint(1, max_addend)
        b = rng.randint(1, max_addend)
        if kind < 0.3:
            expr, result = f"{a} + {b}", a + b
            q = _pick(rng, ADD_ASKS).format(a=a, b=b)
        elif kind < 0.5:
            a, b = max(a, b), min(a, b)  # keep subtractions non-negative
            expr, result = f"{a} - {b}", a - b
            q = _pick(rng, SUB_ASKS).format(a=a, b=b)
        elif kind < 0.75:
            a, b = rng.randint(2, 20), rng.randint(2, 30)
            expr, result = f"{a} x {b}", a * b
            q = _pick(rng, MUL_ASKS).format(a=a, b=b)
        elif kind < 0.85:
            b = rng.randint(2, 12)
            a = b * rng.randint(2, 20)  # keep it exact
            expr, result = f"{a} / {b}", a // b
            q = _pick(rng, DIV_ASKS).format(a=a, b=b)
        elif kind < 0.95:
            a = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90])
            b = rng.choice([20, 40, 50, 60, 80, 100, 120, 150, 200, 250, 400, 500])
            result = a * b / 100
            expr = f"{a}% of {b}"
            q = _pick(rng, PCT_ASKS).format(a=a, b=b)
        else:
            a = rng.randint(2, 40)
            result = a * a
            expr = f"{a} squared"
            q = _pick(rng, SQUARE_ASKS).format(a=a)
        answer = _pick(rng, NUM_ANSWERS).format(expr=expr, r=_fmt(result))
        blocks.append(f"User: {q}\nAssistant: {answer}")

    # ---- unit conversions ------------------------------------------------
    for _ in range(n_conversions):
        unit = rng.choice(CONVERSIONS)
        src, dst, factor, template = unit[0], unit[1], unit[2], unit[3]
        value = rng.randint(2, 400)
        if len(unit) > 4:
            result = value * factor + unit[4]
        else:
            result = value * factor
        q = _pick(rng, CONV_ASKS).format(v=value, a=src, b=dst)
        blocks.append(f"User: {q}\nAssistant: " + template.format(v=value, r=result))

    # ---- definitions and facts ------------------------------------------
    for _ in range(n_facts):
        if rng.random() < 0.5:
            topic, answers = rng.choice(list(FACTS.items()))
            ask = _pick(rng, WHAT_ASKS).format(t=topic)
        else:
            topic, answers = rng.choice(list(HOWTO.items()))
            ask = _pick(rng, HOW_ASKS).format(t=topic)
        blocks.append(_exchange(rng, ask, answers))

    # ---- social / chit-chat / meta --------------------------------------
    for _ in range(n_social):
        group = rng.choice(SOCIAL_GROUPS)
        questions, answers = rng.choice(group)
        blocks.append(_exchange(rng, _pick(rng, questions), answers))

    # ---- multi-turn conversations ---------------------------------------
    for _ in range(n_dialogues):
        turns: List[Tuple[str, str]] = []
        # open with a greeting most of the time
        if rng.random() < 0.7:
            gq, ga = rng.choice(GREETINGS)
            turns.append((_pick(rng, gq), _pick(rng, ga)))
        # one or two real exchanges
        for _ in range(rng.randint(1, 2)):
            if rng.random() < 0.35:
                a, b = rng.randint(1, max_addend), rng.randint(1, max_addend)
                expr, result = f"{a} + {b}", a + b
                turns.append((_pick(rng, ADD_ASKS).format(a=a, b=b),
                              _pick(rng, NUM_ANSWERS).format(expr=expr, r=_fmt(result))))
            elif rng.random() < 0.5:
                topic, answers = rng.choice(list(FACTS.items()))
                turns.append((_pick(rng, WHAT_ASKS).format(t=topic), _pick(rng, answers)))
            else:
                group = rng.choice(SOCIAL_GROUPS)
                questions, answers = rng.choice(group)
                turns.append((_pick(rng, questions), _pick(rng, answers)))
        # a follow-up and/or a sign-off
        if rng.random() < 0.5:
            fq, fa = rng.choice(FOLLOWUPS)
            turns.append((_pick(rng, fq), _pick(rng, fa)))
        if rng.random() < 0.4:
            turns.append((_pick(rng, THANKS[0]), _pick(rng, THANKS[1])))
        if rng.random() < 0.4:
            turns.append((_pick(rng, BYE[0]), _pick(rng, BYE[1])))
        blocks.append(
            "\n\n".join(f"User: {q}\nAssistant: {a}" for q, a in turns)
        )

    rng.shuffle(blocks)
    return "\n\n".join(blocks) + "\n"


def corpus_stats(text: str) -> str:
    """A one-line summary used by the CLI."""
    exchanges = text.count("User:")
    return f"{len(text):,} characters, ~{exchanges:,} exchanges"
