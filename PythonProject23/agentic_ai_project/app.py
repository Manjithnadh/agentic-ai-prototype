from agents.rag_agent import build_qa

if __name__ == "__main__":
    qa = build_qa("uploads/Frontend_JD.pdf")   # pass your file here
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        print("AI:", qa.run(query))
