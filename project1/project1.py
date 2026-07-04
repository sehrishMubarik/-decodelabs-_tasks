"""
DecodeLabs - Project 1 (Advanced Edition): Human-Like Rule-Based Chatbot
-------------------------------------------------------------------------
A more "real-world" version of a rule-based bot. Still 100% deterministic
and traceable (no ML), but adds the techniques real FAQ / support bots
use to feel less robotic:

  1. Multi-phrase intents   -> many ways to say "hello" all map to one rule
  2. Fuzzy matching         -> typos like "helo" or "thnx" still match
  3. Memory / context       -> remembers your name across the conversation
  4. Time-aware responses   -> greets differently morning/afternoon/night
  5. Session logging        -> writes the chat transcript to a file,
                                like a real support bot would for QA
"""

import re
import random
import difflib
from datetime import datetime

# ---------------------------------------------------------------------
# KNOWLEDGE BASE
# Each intent has several *example patterns* (not exact-match strings).
# We fuzzy-match the user's message against all patterns to find the
# closest intent, instead of requiring a perfect string match.
# ---------------------------------------------------------------------
INTENTS = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "yo", "good morning", "good evening", "sup"],
        "responses": ["Hey{name}! Great to hear from you.", "Hello{name}! What's on your mind?"],
    },
    "wellbeing": {
        "patterns": ["how are you", "how's it going", "how are things", "you good"],
        "responses": ["I'm running smoothly, thanks for asking! How about you{name}?"],
    },
    "name_query": {
        "patterns": ["what is your name", "who are you", "what should i call you"],
        "responses": ["I'm RuleBot, DecodeLabs' support assistant."],
    },
    "capabilities": {
        "patterns": ["what can you do", "help me", "what do you know", "how can you help"],
        "responses": ["I can chat casually, remember your name, and answer a few FAQs. "
                      "Try asking about pricing, hours, or just say hi!"],
    },
    "thanks": {
        "patterns": ["thank you", "thanks", "appreciate it", "thx", "cheers"],
        "responses": ["You're very welcome{name}!", "Anytime{name}, happy to help."],
    },
    "frustration": {
        "patterns": ["this is annoying", "you are useless", "not helpful", "this is bad", "stupid bot"],
        "responses": ["I'm sorry this hasn't been helpful{name}. Want me to connect you to a human agent?"],
    },
    "pricing": {
        "patterns": ["how much does it cost", "pricing", "price", "what's the cost"],
        "responses": ["Our plans start at $9/month. Want details on a specific tier?"],
    },
    "hours": {
        "patterns": ["what are your hours", "when are you open", "business hours"],
        "responses": ["We're online 24/7 — I never sleep!"],
    },
    "joke": {
        "patterns": ["tell me a joke", "make me laugh", "say something funny"],
        "responses": ["Why do programmers prefer dark mode? Because light attracts bugs."],
    },
}

EXIT_COMMANDS = {"bye", "exit", "quit", "goodbye", "see you"}
FUZZY_THRESHOLD = 0.72  # how close a match needs to be (0-1) to count

# Flatten all patterns into one lookup list: pattern -> intent name
PATTERN_TO_INTENT = {
    pattern: intent
    for intent, data in INTENTS.items()
    for pattern in data["patterns"]
}


def sanitize(raw_input: str) -> str:
    """Lowercase, strip whitespace, and collapse extra spaces."""
    text = raw_input.lower().strip()
    return re.sub(r"\s+", " ", text)


def extract_name(text: str) -> str | None:
    """Very light 'entity extraction': looks for 'my name is X' or 'i'm X'."""
    match = re.search(r"(?:my name is|i am|i'm)\s+([a-zA-Z]+)", text)
    return match.group(1).capitalize() if match else None


def time_based_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"


def match_intent(text: str) -> str | None:
    """Fuzzy-match the input against known patterns; return the best intent."""
    close = difflib.get_close_matches(text, PATTERN_TO_INTENT.keys(), n=1, cutoff=FUZZY_THRESHOLD)
    if close:
        return PATTERN_TO_INTENT[close[0]]

    # Fallback: check if any pattern's keywords appear inside the sentence
    for pattern, intent in PATTERN_TO_INTENT.items():
        if pattern in text:
            return intent
    return None


class ChatBot:
    def __init__(self):
        self.user_name = None
        self.turn_count = 0
        self.transcript = []

    def name_tag(self) -> str:
        """Returns ', Name' if we know the user's name, else ''. Used to
        personalize responses like 'Hey, Sam!' vs 'Hey!'."""
        return f", {self.user_name}" if self.user_name else ""

    def respond(self, raw_input: str) -> str:
        self.turn_count += 1
        clean_input = sanitize(raw_input)
        self.transcript.append(("user", raw_input))

        # Try to learn the user's name at any point in the conversation.
        possible_name = extract_name(clean_input)
        if possible_name:
            self.user_name = possible_name
            reply = f"Nice to meet you, {self.user_name}!"
            self.transcript.append(("bot", reply))
            return reply

        if "what is my name" in clean_input or "what's my name" in clean_input:
            reply = (f"Your name is {self.user_name}!" if self.user_name
                      else "I don't think you've told me your name yet!")
            self.transcript.append(("bot", reply))
            return reply

        intent = match_intent(clean_input)
        if intent:
            template = random.choice(INTENTS[intent]["responses"])
            reply = template.format(name=self.name_tag())
        else:
            reply = ("I'm not sure I understand. Could you rephrase that, "
                      "or ask me something like pricing, hours, or just say hi?")

        self.transcript.append(("bot", reply))
        return reply

    def save_log(self, path: str = "chat_log.txt"):
        with open(path, "w") as f:
            f.write(f"Session on {datetime.now()}\n")
            for speaker, message in self.transcript:
                f.write(f"{speaker}: {message}\n")


def run_chatbot():
    bot = ChatBot()
    print(f"RuleBot: {time_based_greeting()}! I'm RuleBot. Type 'bye' to exit.")

    while True:
        raw_input_text = input("You: ")
        clean_input = sanitize(raw_input_text)

        if clean_input in EXIT_COMMANDS:
            name = f", {bot.user_name}" if bot.user_name else ""
            print(f"RuleBot: It was great chatting with you{name}! "
                  f"We exchanged {bot.turn_count} messages. Goodbye!")
            bot.save_log()
            break

        print(f"RuleBot: {bot.respond(raw_input_text)}")


if __name__ == "__main__":
    run_chatbot()