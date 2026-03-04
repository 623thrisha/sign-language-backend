from collections import deque

sentence = ""
buffer = deque(maxlen=10)
last_confirmed = ""

OUTPUT_FILE = "output.txt"

def update_sentence(letter):
    global sentence, last_confirmed

    if letter is None:
        buffer.clear()
        return sentence

    buffer.append(letter)

    if buffer.count(letter) >= 7 and letter != last_confirmed:
        # ✅ Handle SPACE and DELETE gestures
        if letter == "SPACE":
            sentence += " "
        elif letter == "DELETE":
            sentence = sentence[:-1] if sentence else ""  # avoid error if empty
        else:
            sentence += letter

        last_confirmed = letter
        buffer.clear()

        # 🔴 Write the updated sentence to file
        with open(OUTPUT_FILE, "w") as f:
            f.write(sentence)

    return sentence


def get_sentence():
    try:
        with open(OUTPUT_FILE, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""
