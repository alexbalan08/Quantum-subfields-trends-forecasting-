import anthropic
import json
import os
import time

client = anthropic.Anthropic(api_key=api_key)

LABEL_INSTRUCTIONS = """You are a domain expert in quantum computing.

Your task is to assign one label to each research paper based on its title and abstract.

Choose from the following categories:

- quantum algorithms  
- quantum error correction  
- quantum hardware  
- quantum communication  
- quantum simulation  
- quantum machine learning  
- quantum information theory  
- quantum cryptography  
- quantum foundations  
- quantum control  
- quantum compilation  
- quantum sensing  
- quantum metrology  
- quantum resource theory  
- quantum software tools  
- not quantum computing

Return only the label.

"""

INPUT_FILE = "quantum_computing_filtered_stream.jsonl"  # JSONL file 
OUTPUT_FILE = "quantum_computing_labeled.jsonl"

def load_already_processed():
    processed = set()
    if not os.path.exists(OUTPUT_FILE):
        return processed
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                processed.add((d["title"], d["summary"]))
            except Exception:
                continue
    return processed

def label_paper(paper):
    prompt = (
        LABEL_INSTRUCTIONS
        + f"\n\nTitle: {paper['title']}\n\nAbstract: {paper['summary']}"
    )
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    label = response.content[0].text.strip().lower()
    return label

def main():
    processed = load_already_processed()
    print(f"processed {len(processed)} papers")

    with open(INPUT_FILE, "r", encoding="utf-8") as in_f, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:

        for i, line in enumerate(in_f):
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
            except Exception as e:
                print(f"bad JSON line {i}: {e}")
                continue

			# here skip for already labeled, if need to pause the script and resume later
            key = (paper.get("title", ""), paper.get("summary", "")) # summary is abstract 
            if key in processed:
                print(f"[{i}] skipping: {paper.get('title', '<no title>')}")
                continue

            print(f"[{i}] labeling: {paper.get('title', '<no title>')}")
            try:
                label = label_paper(paper)
            except Exception as e:
                print(f"error labeling paper {i}: {e}")
                continue

            out_f.write(json.dumps({
                "title": paper.get("title", ""),
                "summary": paper.get("summary", ""),
                "label": label
            }) + "\n")
            out_f.flush()

    print("done")

if __name__ == "__main__":
    main()
