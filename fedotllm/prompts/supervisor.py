def choose_next_prompt(messages: str) -> str:
    return f"""
You are a supervisor.
You are given a conversation between a user and an agents.
You need to decide who should act next. Or should we FINISH?
'''
user = '''
[Conversation history]
{messages}
"""
