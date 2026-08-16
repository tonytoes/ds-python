words = input("Enter a word: ")
n = int(input("Enter a value: "))

result = words[:n].lower() + words[n:]
print(result)
