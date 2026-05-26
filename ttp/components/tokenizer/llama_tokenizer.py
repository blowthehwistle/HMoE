from transformers import AutoTokenizer

# 7b, 13b, 70b 모두 토크나이저는 동일합니다.
model_id = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)

# 테스트
print(tokenizer.encode("Hello World"))