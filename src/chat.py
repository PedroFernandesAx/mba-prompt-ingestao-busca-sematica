from search import search_prompt


def main():
    print("Chat iniciado. Digite 'sair' para encerrar.\n")

    while True:
        question = input("Você: ").strip()

        if not question:
            continue
        if question.lower() == "sair":
            break

        search_prompt(question)


if __name__ == "__main__":
    main()