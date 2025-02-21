import os
import re
import whisper
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import ollama


def load_whisper_model(model_name="base"):
    return whisper.load_model(model_name)

def process_voice_input(audio_file_path, whisper_model):
    result = whisper_model.transcribe(audio_file_path)
    return result["text"]

game_state = [] # history

# gen story w DeepSeek
def generate_story(input_dict):
    user_input = input_dict.get("input", "")
    
    context = "\n".join(f"{item[0]}, {item[1]}" for item in game_state[-5:]) # keep last 5 interactions
    full_prompt = f"Context: {context}\n The player says: {user_input}.\nContinue the story given the option the player chooses. Give 2 new options that keep the story going.\n"
    
    response = ollama.generate(model='deepseek-r1:1.5b', prompt=full_prompt)
    
    # take out <think> and </think> from response
    cleaned_response = re.sub(r'<think>.*?</think>', '', response['response'], flags=re.DOTALL).strip()
    
    # update history
    game_state.append((f"Player: {user_input}", f"Game: {cleaned_response}"))
    
    return cleaned_response

# image processing with Qwen-VL
def process_image(image_path):
    if not os.path.exists(image_path):
        return "Image not found. Please check the path."
    
    # gen description of image
    response = ollama.generate(
        model='bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh',
        prompt=f"Describe the image located at {image_path} and suggest how it could add a new story element."
    )
    return response['response']

# LangChain Runnable to chain tasks
generate_story_runnable = RunnableLambda(generate_story)

# init message history
message_history = ChatMessageHistory()

# init RunnableWithMessageHistory
conversation = RunnableWithMessageHistory(
    generate_story_runnable,
    lambda session_id: message_history,  # TODO: why is it blanked out, Use ChatMessageHistory for memory
    input_messages_key="input",
    history_messages_key="history",
)

# sidekick for advice
def ask_sidekick(question):
    context = "\n".join(f"{item[0]}, {item[1]}" for item in game_state[-5:]) # keep last 5 interactions
    full_prompt = f"Context: {context}\nYou're a helpful sidekick. The player asks you: {question}\nAnswer the question given the context"
    
    response = ollama.generate(model='deepseek-r1:1.5b', prompt=full_prompt)
    
    # take out <think> and </think> from response
    cleaned_response = re.sub(r'<think>.*?</think>', '', response['response'], flags=re.DOTALL).strip()
    
    # update history
    game_state.append((f"Player: {question}", f"Sidekick: {cleaned_response}"))
    return cleaned_response

# game loop
def start_game():
    print("Welcome to The Game")
    print ("This took me forever so please enjoy it!")
    
    # load Whisper model
    whisper_model = load_whisper_model("base")
    
    # init story gen
    intro_story = generate_story({"input": "Create a text-based choose-your-own-adventure game. You are communicating with the player. Begin by introducing the game and explaining how the player will make choices. Each response should continue the adventure and present the player with two clear options for what to do next. The game should be engaging, allowing the player to explore freely and make impactful decisions that shape the outcome. The story should be immersive, with unexpected twists, challenges, and a clear path to victory. Now, introduce the game and explain the rules before presenting 2 options for the player to decide their next action."})
    print(intro_story)
    
    while True:
        # take in user input (text or voice)
        user_input = input("What do you want to do?\nType 'help' for info on voice and image input and to speak with your sidekick, 'exit' to end The Game, or 'voice' to speak instructions: ")
        
        if user_input.lower() == 'exit':
            print("Thanks for playing! Goodbye.")
            break
        elif user_input.lower() == 'help':
            print("To add voice input after submitting 'voice', type the path to an mp3 file.\nTo submit an image type 'image:' followed by the image path.\nIf you need further guidance, summon your sidekick by typing 'ask sidekick:' followed by your question")
            continue
        elif user_input.lower() == 'voice':
            # take in voice input
            audio_file_path = user_input
            if os.path.exists(audio_file_path):
                voice_input = process_voice_input(audio_file_path, whisper_model)
                print("You said:", voice_input)
                response = conversation.invoke({"input": voice_input}, config={"configurable": {"session_id": "game_session"}})
                print("Game response:", response) # TODO: does this feed into game logic?
            else:
                print("Audio file not found. Please check the path.")
        elif user_input.lower().startswith("image:"):
            # process an image
            image_path = user_input[6:].strip()  # extract image path after "image:"
            if os.path.exists(image_path):
                image_description = process_image(image_path)
                print("Image description:", image_description)
                response = conversation.invoke({"input": f"The player sees: {image_description}"}, config={"configurable": {"session_id": "game_session"}})
                print("Game response based on your image:", response)
            else:
                print("Image not found. Please check the path.")
        elif user_input.lower().startswith("ask sidekick:"):
            # ask the sidekick for advice
            question = user_input[13:]
            sidekick_response = ask_sidekick(question)
            print("Sidekick says:", sidekick_response)
        else:
            # handle text input
            response = conversation.invoke({"input": user_input}, config={"configurable": {"session_id": "game_session"}})
            print("Game response:", response)

# run the game
if __name__ == "__main__":
    start_game()