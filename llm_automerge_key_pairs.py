from openai import OpenAI

client = OpenAI()

def generate_pairs(input_keys: list[str]) -> list[tuple[str, str]]:
    return [(input_keys[i], input_keys[i+1]) for i in range(len(input_keys) - 1)]

def vote_on_merge(item_a: str, item_b: str) -> bool:
    prompt = f"Do '{item_a}' and '{item_b}' refer to the same organization? Reply ONLY 'YES' or 'NO'."
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    except Exception as e:
        print(f"Error calling API: {e}")
        return False

def run(input_list: list[str]) -> list[tuple[str, str]]:
    pairs = generate_pairs(input_list)
    confirmed_merges = []

    for a, b in pairs:
        if vote_on_merge(a, b):
            confirmed_merges.append((a, b))
            
    return confirmed_merges
#EOF