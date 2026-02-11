import numpy as np
import random

#M Examples
#N Features -> N Dimensions
#X.shape(M, N)
#Y.shape(M, 1)

def askQuestion(question, answer):
	print(question)

	guess = ""
	while guess != answer:
		guess = input()
		if guess != answer:
			print("Try again")
	print("Correct!")

if __name__ == "__main__":
	print("Welcome to the Shape Guess Game")
	while (1):
		features = random.randint(1, 100)
		examples = random.randint(1, 100)
		print(f"For: {features} Features and {examples} Examples")

		askQuestion("What's the shape of X(M, N)?", f"({examples}, {features})")
		print("\n")

		askQuestion("What's the shape of Y(M, N)?", f"({examples}, 1)")
		print("\n")

		askQuestion("What is the type of a Vector of shape(M, 1)?", f"Column Vector")
		print("\n")

		askQuestion("What is the type of a Vector of shape(1, N)?", f"Row Vector")
		print("\n")

		askQuestion("What is the type of a Vector of shape(M, N)?", f"Matrix")
		print("\n\n\n")
