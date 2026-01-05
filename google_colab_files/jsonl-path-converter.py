import json
import os

# -------- CONFIG --------
INPUT_JSONL  = "/home/troy/qwen-nrp-proj/test.jsonl"
OUTPUT_JSONL = "/home/troy/qwen-nrp-proj/test-colab.jsonl"

OLD_PREFIX = "/workspace/data/jam-causing-material/test/images/"
NEW_PREFIX = "/content/drive/MyDrive/test/images/"
# ------------------------

def rewrite_jsonl_paths():
    with open(INPUT_JSONL, "r") as fin, open(OUTPUT_JSONL, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            record = json.loads(line)

            for msg in record.get("messages", []):
                if msg.get("role") == "user":
                    for item in msg.get("content", []):
                        if item.get("type") == "image":
                            old_path = item["image"]

                            if not old_path.startswith(OLD_PREFIX):
                                raise ValueError(
                                    f"Line {line_num}: unexpected image path:\n{old_path}"
                                )

                            filename = os.path.basename(old_path)
                            item["image"] = os.path.join(NEW_PREFIX, filename)

            fout.write(json.dumps(record) + "\n")

    print(f"Done. Colab JSONL written to:\n{OUTPUT_JSONL}")

if __name__ == "__main__":
    rewrite_jsonl_paths()
