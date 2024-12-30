from openai import OpenAI
    
if __name__ == '__main__':
    # init openai client, ensure you run ollama server first (ollama server -p 11434)
    client = OpenAI(base_url="http://localhost:11434/v1/",api_key="ollama")
    # init chat history
    chat_history = []
    # start chat loop
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        # optional to add system role content
        chat_history.append({"role":"user","content":user_input})
        try:
            chat_complition = client.chat.completions.create(messages=chat_history,model="llama3")
            model_response = chat_complition.choices[0]
            print("Assistant Response: ",model_response.message.content)
            chat_history.append({"role":"assistant","content":model_response.message.content})
        except Exception as e:
            raise e