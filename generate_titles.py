import torch
from transformers import BertTokenizer, BertForMaskedLM

def generate_titles(text):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    # Tokenize the input text
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([tokens])
    
    # Generate predicted token IDs
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs[0]
    
    # Decode the predicted token IDs into titles
    predicted_titles = []
    for prediction in predictions[0]:
        predicted_token = torch.argmax(prediction).item()
        predicted_title = tokenizer.decode([predicted_token], skip_special_tokens=True)
        predicted_titles.append(predicted_title)
    
    return predicted_titles

# Test the title generation function
text = "This is an abstract about the application of BERT model."
titles = generate_titles(text)
print("Generated Titles:")
for title in titles:
    print(title)
